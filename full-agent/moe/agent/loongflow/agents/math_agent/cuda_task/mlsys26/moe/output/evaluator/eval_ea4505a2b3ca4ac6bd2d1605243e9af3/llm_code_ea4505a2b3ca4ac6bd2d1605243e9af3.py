import torch
import torch.nn.functional as F


H_DIM = 7168
I_DIM = 2048
E_LOCAL = 32
E_GLOBAL = 256
N_GROUP = 8
GROUP_SIZE = 32
TOPK_GROUP = 4
TOPK_EXPERT = 8
BLOCK = 128
H_BLOCKS = H_DIM // BLOCK          # 56
I_BLOCKS = I_DIM // BLOCK          # 16
O1_BLOCKS = (2 * I_DIM) // BLOCK   # 32


def _dequant_hidden_rows(hidden_states: torch.Tensor,
                         hidden_states_scale: torch.Tensor,
                         token_ids: torch.Tensor) -> torch.Tensor:
    # hidden_states: [T, 7168] fp8
    # hidden_states_scale: [56, T] fp32
    rows = hidden_states.index_select(0, token_ids).to(torch.float32).view(-1, H_BLOCKS, BLOCK)
    scales = hidden_states_scale[:, token_ids].transpose(0, 1).contiguous().unsqueeze(-1)
    return (rows * scales).reshape(-1, H_DIM)


def _dequant_gemm1_weight_one(gemm1_weights: torch.Tensor,
                              gemm1_weights_scale: torch.Tensor,
                              e: int) -> torch.Tensor:
    # weights[e]: [4096, 7168], scales[e]: [32, 56]
    # block layout is [o_blk, 128, k_blk, 128]
    w = gemm1_weights[e].to(torch.float32).view(O1_BLOCKS, BLOCK, H_BLOCKS, BLOCK)
    s = gemm1_weights_scale[e].to(torch.float32).view(O1_BLOCKS, 1, H_BLOCKS, 1)
    return (w * s).reshape(2 * I_DIM, H_DIM)


def _dequant_gemm2_weight_one(gemm2_weights: torch.Tensor,
                              gemm2_weights_scale: torch.Tensor,
                              e: int) -> torch.Tensor:
    # weights[e]: [7168, 2048], scales[e]: [56, 16]
    # block layout is [h_blk, 128, i_blk, 128]
    w = gemm2_weights[e].to(torch.float32).view(H_BLOCKS, BLOCK, I_BLOCKS, BLOCK)
    s = gemm2_weights_scale[e].to(torch.float32).view(H_BLOCKS, 1, I_BLOCKS, 1)
    return (w * s).reshape(H_DIM, I_DIM)


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,        # [T, 256] float32
    routing_bias: torch.Tensor,          # [256]    bfloat16
    hidden_states: torch.Tensor,         # [T, 7168] float8_e4m3fn
    hidden_states_scale: torch.Tensor,   # [56, T]  float32
    gemm1_weights: torch.Tensor,         # [32, 4096, 7168] float8_e4m3fn
    gemm1_weights_scale: torch.Tensor,   # [32, 32, 56] float32
    gemm2_weights: torch.Tensor,         # [32, 7168, 2048] float8_e4m3fn
    gemm2_weights_scale: torch.Tensor,   # [32, 56, 16] float32
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    device = hidden_states.device
    T = routing_logits.shape[0]

    # 1) DeepSeek no-aux routing
    scores = torch.sigmoid(routing_logits.to(torch.float32))
    scores_with_bias = scores + routing_bias.to(torch.float32).view(1, E_GLOBAL)

    grouped = scores_with_bias.view(T, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)
    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=device)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_GLOBAL)

    neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=device)
    pruned = torch.where(expert_mask, scores_with_bias, neg_inf)
    topk_idx = torch.topk(pruned, k=TOPK_EXPERT, dim=1, largest=True, sorted=True).indices

    sel_mask = torch.zeros((T, E_GLOBAL), dtype=torch.bool, device=device)
    sel_mask.scatter_(1, topk_idx, True)

    routed = torch.where(sel_mask, scores, torch.zeros((), dtype=torch.float32, device=device))
    routed_sum = routed.sum(dim=1, keepdim=True).clamp_min_(1e-20)
    routed = routed * (float(routed_scaling_factor) / routed_sum)

    # 2) Local expert dispatch
    local_start = int(local_expert_offset)
    local_end = local_start + E_LOCAL

    local_pick_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    flat_token_idx, flat_slot_idx = torch.nonzero(local_pick_mask, as_tuple=True)
    tk_total = flat_token_idx.numel()

    if tk_total == 0:
        return torch.zeros((T, H_DIM), dtype=torch.bfloat16, device=device)

    flat_global_experts = topk_idx[flat_token_idx, flat_slot_idx]
    flat_local_experts = (flat_global_experts - local_start).to(torch.int64)
    flat_weights = routed[flat_token_idx, flat_global_experts].to(torch.float32)

    order = torch.argsort(flat_local_experts)
    sorted_tokens = flat_token_idx[order].to(torch.int64)
    sorted_local_experts = flat_local_experts[order]
    sorted_weights = flat_weights[order]

    expert_counts = torch.bincount(sorted_local_experts, minlength=E_LOCAL)
    expert_offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # 3) Expert compute
    output = torch.zeros((T, H_DIM), dtype=torch.float32, device=device)

    w1_cache = [None] * E_LOCAL
    w2_cache = [None] * E_LOCAL

    for e in range(E_LOCAL):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok = sorted_tokens[start:end]
        rw = sorted_weights[start:end]

        a_e = _dequant_hidden_rows(hidden_states, hidden_states_scale, tok)

        w1 = w1_cache[e]
        if w1 is None:
            w1 = _dequant_gemm1_weight_one(gemm1_weights, gemm1_weights_scale, e)
            w1_cache[e] = w1

        g1 = torch.mm(a_e, w1.t())
        x1 = g1[:, :I_DIM]
        x2 = g1[:, I_DIM:]
        c = F.silu(x2) * x1

        w2 = w2_cache[e]
        if w2 is None:
            w2 = _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, e)
            w2_cache[e] = w2

        o = torch.mm(c, w2.t())
        o.mul_(rw.unsqueeze(1))
        output.index_add_(0, tok, o)

        del a_e, g1, x1, x2, c, o

    return output.to(torch.bfloat16)