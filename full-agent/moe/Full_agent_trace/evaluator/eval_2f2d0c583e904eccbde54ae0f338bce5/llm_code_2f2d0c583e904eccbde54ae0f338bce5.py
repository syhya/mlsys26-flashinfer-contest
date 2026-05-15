import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


H = 7168
I = 2048
BLOCK = 128
H_BLKS = H // BLOCK          # 56
I2 = 2 * I
I2_BLKS = I2 // BLOCK        # 32
I_BLKS = I // BLOCK          # 16
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = 256 // N_GROUP
E_LOCAL = 32


def _dequant_hidden_rows(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    x = hidden_states.index_select(0, token_ids).float().view(-1, H_BLKS, BLOCK)
    s = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous().view(-1, H_BLKS, 1)
    return (x * s).view(-1, H)


def _dequant_hidden_rows_to_bf16(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    x = hidden_states.index_select(0, token_ids).float().view(-1, H_BLKS, BLOCK)
    s = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous().view(-1, H_BLKS, 1)
    return (x * s).view(-1, H).to(torch.bfloat16)


def _dequant_gemm1_weight_one(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    e: int,
) -> torch.Tensor:
    w = gemm1_weights[e].float().view(I2_BLKS, BLOCK, H_BLKS, BLOCK)
    s = gemm1_weights_scale[e].float().view(I2_BLKS, 1, H_BLKS, 1)
    return (w * s).reshape(I2, H)


def _dequant_gemm1_weight_one_bf16(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    e: int,
) -> torch.Tensor:
    w = gemm1_weights[e].float().view(I2_BLKS, BLOCK, H_BLKS, BLOCK)
    s = gemm1_weights_scale[e].float().view(I2_BLKS, 1, H_BLKS, 1)
    return (w * s).reshape(I2, H).to(torch.bfloat16)


def _dequant_gemm2_weight_one(
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    e: int,
) -> torch.Tensor:
    w = gemm2_weights[e].float().view(H_BLKS, BLOCK, I_BLKS, BLOCK)
    s = gemm2_weights_scale[e].float().view(H_BLKS, 1, I_BLKS, 1)
    return (w * s).reshape(H, I)


def _dequant_gemm2_weight_one_bf16(
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    e: int,
) -> torch.Tensor:
    w = gemm2_weights[e].float().view(H_BLKS, BLOCK, I_BLKS, BLOCK)
    s = gemm2_weights_scale[e].float().view(H_BLKS, 1, I_BLKS, 1)
    return (w * s).reshape(H, I).to(torch.bfloat16)


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_global = routing_logits.shape[1]

    # 1) Exact routing
    scores = torch.sigmoid(routing_logits.float())
    scores_bias = scores + routing_bias.float().view(1, -1)

    grouped = scores_bias.view(T, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, group_idx, True)
    expert_mask_groups = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
    pruned = torch.where(expert_mask_groups, scores_bias, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    topk_scores = scores.gather(1, topk_idx)
    topk_weights = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * float(routed_scaling_factor)

    # 2) Build local dispatch
    local_start = int(local_expert_offset)
    local_end = local_start + E_LOCAL

    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    flat_token_idx, flat_slot_idx = torch.nonzero(local_mask, as_tuple=True)
    Tk_total = flat_token_idx.numel()

    if Tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    local_expert_idx = (topk_idx[flat_token_idx, flat_slot_idx] - local_start).to(torch.int64)
    route_w = topk_weights[flat_token_idx, flat_slot_idx].float()

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int64)
    sorted_experts = local_expert_idx[order]
    sorted_route_w = route_w[order]

    expert_counts = torch.bincount(sorted_experts, minlength=E_LOCAL)
    expert_offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Speed-oriented but correctness-safe:
    # use BF16 matmuls with FP32 accumulation behavior from torch.mm backend;
    # keep final accumulation/output in FP32 and never requantize SwiGLU to FP8.
    #
    # Cache dequantized BF16 weights lazily for active experts only.
    w1_cache_bf16 = [None] * E_LOCAL
    w2_cache_bf16 = [None] * E_LOCAL

    # Small-batch exact fallback threshold
    small_threshold = 4

    for e in range(E_LOCAL):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok = sorted_tokens[start:end]
        rw = sorted_route_w[start:end]
        tk = end - start

        if tk <= small_threshold:
            a_e = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok)
            w1 = _dequant_gemm1_weight_one(gemm1_weights, gemm1_weights_scale, e)
            g1 = torch.mm(a_e, w1.t())
            x1 = g1[:, :I]
            x2 = g1[:, I:]
            c = F.silu(x2) * x1
            w2 = _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, e)
            o = torch.mm(c, w2.t())
            o.mul_(rw[:, None])
            output.index_add_(0, tok, o)
            continue

        a_bf16 = _dequant_hidden_rows_to_bf16(hidden_states, hidden_states_scale, tok)

        w1_bf16 = w1_cache_bf16[e]
        if w1_bf16 is None:
            w1_bf16 = _dequant_gemm1_weight_one_bf16(gemm1_weights, gemm1_weights_scale, e)
            w1_cache_bf16[e] = w1_bf16

        g1 = torch.mm(a_bf16, w1_bf16.t()).float()
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        c = F.silu(x2) * x1

        w2_bf16 = w2_cache_bf16[e]
        if w2_bf16 is None:
            w2_bf16 = _dequant_gemm2_weight_one_bf16(gemm2_weights, gemm2_weights_scale, e)
            w2_cache_bf16[e] = w2_bf16

        o = torch.mm(c.to(torch.bfloat16), w2_bf16.t()).float()
        o.mul_(rw[:, None])
        output.index_add_(0, tok, o)

    return output.to(torch.bfloat16)