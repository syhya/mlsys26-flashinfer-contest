import torch
import triton
import triton.language as tl


H = 7168
I = 2048
E_LOCAL = 32
E_GLOBAL = 256
N_GROUP = 8
GROUP_SIZE = 32
TOPK_GROUP = 4
TOP_K = 8
BLOCK = 128
H_BLK = H // BLOCK          # 56
I2_BLK = (2 * I) // BLOCK   # 32
I_BLK = I // BLOCK          # 16


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["M"],
)
@triton.jit
def _grouped_gemm1_kernel(
    A_ptr, A_scale_ptr,
    W_ptr, W_scale_ptr,
    C_ptr,
    sorted_tokens_ptr,
    expert_offsets_ptr,
    M,  # total active rows
    stride_am, stride_ak,
    stride_ask, stride_ast,
    stride_we, stride_wn, stride_wk,
    stride_wse, stride_wsn, stride_wsk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start = tl.load(expert_offsets_ptr + pid_e)
    end = tl.load(expert_offsets_ptr + pid_e + 1)
    tk = end - start
    if pid_m * BLOCK_M >= tk:
        return

    offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m_local < tk
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < (2 * I)

    row_ids = start + offs_m_local
    token_ids = tl.load(sorted_tokens_ptr + row_ids, mask=mask_m, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(0, H_BLK):
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = (
            A_ptr
            + token_ids[:, None] * stride_am
            + (kb * BLOCK_K + offs_k)[None, :] * stride_ak
        )
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

        w_ptrs = (
            W_ptr
            + pid_e.to(tl.int64) * stride_we
            + offs_n[None, :] * stride_wn
            + (kb * BLOCK_K + offs_k)[:, None] * stride_wk
        )
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)

        a_scale_ptrs = A_scale_ptr + kb * stride_ask + token_ids[:, None] * stride_ast
        a_scale = tl.load(a_scale_ptrs, mask=mask_m[:, None], other=0.0)

        n_blk = offs_n // BLOCK
        ws_ptrs = (
            W_scale_ptr
            + pid_e.to(tl.int64) * stride_wse
            + n_blk[None, :] * stride_wsn
            + kb * stride_wsk
        )
        w_scale = tl.load(ws_ptrs, mask=mask_n[None, :], other=0.0)

        dot = tl.dot(a, w, out_dtype=tl.float32)
        acc += dot * a_scale * w_scale

    c_ptrs = (
        C_ptr
        + row_ids[:, None].to(tl.int64) * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def _route(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    scores = torch.sigmoid(routing_logits.float())
    scores_bias = scores + routing_bias.float().view(1, E_GLOBAL)

    grouped = scores_bias.view(scores_bias.shape[0], N_GROUP, GROUP_SIZE)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)
    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, GROUP_SIZE).reshape(-1, E_GLOBAL)

    neg_inf = torch.finfo(scores_bias.dtype).min
    pruned = scores_bias.masked_fill(~expert_mask, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    sel = torch.zeros_like(scores)
    sel.scatter_(1, topk_idx, 1.0)
    weights = scores * sel
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return topk_idx, weights


def _dequant_gemm2_weight_one(
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    e: int,
) -> torch.Tensor:
    w = gemm2_weights[e].float().view(H_BLK, BLOCK, I_BLK, BLOCK)
    s = gemm2_weights_scale[e].float().view(H_BLK, 1, I_BLK, 1)
    return (w * s).reshape(H, I).contiguous()


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
    T = hidden_states.shape[0]

    topk_idx, weights = _route(routing_logits, routing_bias, routed_scaling_factor)

    local_start = int(local_expert_offset)
    local_end = local_start + E_LOCAL

    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    flat_tok, flat_slot = torch.nonzero(local_mask, as_tuple=True)
    tk_total = flat_tok.numel()

    if tk_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    local_expert_idx = (topk_idx[flat_tok, flat_slot] - local_start).to(torch.int32)
    perm = torch.argsort(local_expert_idx)
    sorted_tokens = flat_tok[perm].to(torch.int32)
    sorted_experts = local_expert_idx[perm]

    sorted_weights = weights[flat_tok[perm], topk_idx[flat_tok[perm], flat_slot[perm]]].float()

    expert_counts = torch.bincount(sorted_experts, minlength=E_LOCAL)
    expert_offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    g1 = torch.zeros((tk_total, 2 * I), dtype=torch.float32, device=device)

    grid = lambda META: (
        triton.cdiv(2 * I, META["BLOCK_N"]),
        triton.cdiv(int(expert_counts.max().item()) if tk_total > 0 else 1, META["BLOCK_M"]),
        E_LOCAL,
    )

    _grouped_gemm1_kernel[grid](
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        g1,
        sorted_tokens, expert_offsets,
        tk_total,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        g1.stride(0), g1.stride(1),
        BLOCK_K=BLOCK,
    )

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    w2_cache = [None] * E_LOCAL

    for e in range(E_LOCAL):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok = sorted_tokens[start:end].long()
        rw = sorted_weights[start:end]

        ge = g1[start:end]
        x1 = ge[:, :I]
        x2 = ge[:, I:]
        c = torch.nn.functional.silu(x2) * x1

        w2 = w2_cache[e]
        if w2 is None:
            w2 = _dequant_gemm2_weight_one(gemm2_weights, gemm2_weights_scale, e)
            w2_cache[e] = w2

        o = torch.mm(c, w2.t())
        o.mul_(rw[:, None])
        output.index_add_(0, tok, o)

    return output.to(torch.bfloat16)