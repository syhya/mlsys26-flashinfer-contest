import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ----------------------------
# Routing / dispatch helpers
# ----------------------------

def _route_noaux_topk8(routing_logits: torch.Tensor,
                       routing_bias: torch.Tensor,
                       routed_scaling_factor: float):
    # Exact DeepSeek-style no-aux routing:
    # sigmoid -> add bias for selection -> group top2 sum -> top4 groups -> global top8
    # final weights from unbiased sigmoid scores
    logits = routing_logits.float()
    bias = routing_bias.float().view(1, -1)

    s = torch.sigmoid(logits)
    s_bias = s + bias

    T, E = s.shape
    N_GROUP = 8
    GROUP_SIZE = E // N_GROUP
    TOPK_GROUP = 4
    TOP_K = 8

    sbg = s_bias.view(T, N_GROUP, GROUP_SIZE)
    top2 = torch.topk(sbg, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)
    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), device=logits.device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E)

    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=torch.float32)
    pruned = torch.where(expert_mask, s_bias, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    chosen = torch.zeros_like(s, dtype=torch.bool)
    chosen.scatter_(1, topk_idx, True)
    weights = torch.where(chosen, s, torch.zeros_like(s))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    topk_weights = torch.gather(weights, 1, topk_idx)
    return topk_idx, topk_weights


def _build_local_dispatch(topk_idx: torch.Tensor,
                          topk_weights: torch.Tensor,
                          local_expert_offset: int,
                          e_local: int):
    local_start = int(local_expert_offset)
    mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    tok_idx, slot_idx = torch.nonzero(mask, as_tuple=True)
    if tok_idx.numel() == 0:
        return None

    local_expert_idx = (topk_idx[tok_idx, slot_idx] - local_start).to(torch.int32)
    rw = topk_weights[tok_idx, slot_idx].to(torch.float32)

    perm = torch.argsort(local_expert_idx)
    sorted_tokens = tok_idx[perm].to(torch.int32)
    sorted_experts = local_expert_idx[perm]
    sorted_weights = rw[perm]

    counts = torch.bincount(sorted_experts.to(torch.int64), minlength=e_local)
    offsets = torch.zeros(e_local + 1, device=topk_idx.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts.to(torch.int32), dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, offsets


# ----------------------------
# Triton GEMM1
# Computes:
# hidden_states[token, H(fp8)] x gemm1_weights[e, N=4096, H(fp8)]^T -> G1
# with exact block-scale semantics and FP32 accumulation
# ----------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["N_DIM"],
)
@triton.jit
def _grouped_gemm1_kernel(
    A_ptr, A_scale_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    sorted_tokens_ptr, expert_offsets_ptr,
    H_DIM, N_DIM,
    stride_am, stride_ak,
    stride_ascale_k, stride_ascale_t,
    stride_we, stride_wn, stride_wk,
    stride_wscale_e, stride_wscale_nb, stride_wscale_kb,
    stride_om, stride_on,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    start = tl.load(expert_offsets_ptr + pid_e)
    end = tl.load(expert_offsets_ptr + pid_e + 1)
    tk = end - start

    if pid_m * BLOCK_M >= tk:
        return

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < tk
    n_mask = n_offs < N_DIM

    token_ids = tl.load(sorted_tokens_ptr + start + m_offs, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = tl.arange(0, BLOCK_K).to(tl.int64)
    n64 = n_offs.to(tl.int64)

    for k_base in range(0, H_DIM, BLOCK_K):
        kb = k_base // BLOCK_K
        k_offs = k0 + k_base

        a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        w_ptrs = (
            W_ptr
            + pid_e.to(tl.int64) * stride_we
            + n64[None, :] * stride_wn
            + k_offs[:, None] * stride_wk
        )

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        a_scale_ptrs = A_scale_ptr + kb * stride_ascale_k + token_ids[:, None] * stride_ascale_t
        a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=0.0)

        nb = n_offs // BLOCK_K
        w_scale_ptrs = (
            W_scale_ptr
            + pid_e.to(tl.int64) * stride_wscale_e
            + nb[None, :].to(tl.int64) * stride_wscale_nb
            + kb * stride_wscale_kb
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=0.0)

        acc += tl.dot(a, w, out_dtype=tl.float32) * a_scale * w_scale

    out_ptrs = Out_ptr + (start + m_offs)[:, None].to(tl.int64) * stride_om + n64[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ----------------------------
# Dequant helpers for exact GEMM2
# ----------------------------

def _dequant_gemm2_weight_expert_fp32(w_fp8_e: torch.Tensor, w_scale_e: torch.Tensor) -> torch.Tensor:
    # w_fp8_e: [H, I] fp8
    # w_scale_e: [H//128, I//128]
    # dequant semantics: block [hb, ib, 128, 128] scaled by scale[hb, ib]
    H, I = w_fp8_e.shape
    w = w_fp8_e.float().view(H // 128, 128, I // 128, 128).permute(0, 2, 1, 3).contiguous()
    w = w * w_scale_e[:, :, None, None]
    w = w.permute(0, 2, 1, 3).reshape(H, I).contiguous()
    return w


# ----------------------------
# Main entry
# ----------------------------

@torch.no_grad()
def run(
    routing_logits: torch.Tensor,        # [T, 256] float32
    routing_bias: torch.Tensor,          # [256]    bfloat16
    hidden_states: torch.Tensor,         # [T, 7168] fp8
    hidden_states_scale: torch.Tensor,   # [56, T]  float32
    gemm1_weights: torch.Tensor,         # [32, 4096, 7168] fp8
    gemm1_weights_scale: torch.Tensor,   # [32, 32, 56] float32
    gemm2_weights: torch.Tensor,         # [32, 7168, 2048] fp8
    gemm2_weights_scale: torch.Tensor,   # [32, 56, 16] float32
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    device = hidden_states.device
    T = routing_logits.shape[0]
    H = 7168
    I = 2048
    E_LOCAL = gemm1_weights.shape[0]

    # 1) Exact routing
    topk_idx, topk_weights = _route_noaux_topk8(
        routing_logits, routing_bias, routed_scaling_factor
    )

    # 2) GPU dispatch
    dispatch = _build_local_dispatch(topk_idx, topk_weights, local_expert_offset, E_LOCAL)
    if dispatch is None:
        return torch.zeros((T, H), device=device, dtype=torch.bfloat16)

    sorted_tokens, sorted_experts, sorted_weights, expert_offsets = dispatch
    tk_total = int(sorted_tokens.numel())
    if tk_total == 0:
        return torch.zeros((T, H), device=device, dtype=torch.bfloat16)

    # 3) Fast exact GEMM1 with Triton
    g1 = torch.zeros((tk_total, 2 * I), device=device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(2 * I, META["BLOCK_N"]),
        triton.cdiv(tk_total, META["BLOCK_M"]),
        E_LOCAL,
    )

    _grouped_gemm1_kernel[grid](
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        g1,
        sorted_tokens, expert_offsets,
        H, 2 * I,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        g1.stride(0), g1.stride(1),
        BLOCK_K=128,
    )

    # 4) Exact FP32 SwiGLU
    x1 = g1[:, :I]
    x2 = g1[:, I:]
    s = F.silu(x2) * x1  # [tk_total, I], fp32

    # 5) Correctness-first GEMM2 per expert
    #    Exact block dequantization of weights, FP32 matmul, routing weight applied before scatter.
    grouped_out = torch.zeros((tk_total, H), device=device, dtype=torch.float32)

    # Cache only experts that actually appear
    active_experts = torch.nonzero((expert_offsets[1:] - expert_offsets[:-1]) > 0, as_tuple=False).flatten()
    w2_cache = {}

    for e_t in active_experts.tolist():
        start = int(expert_offsets[e_t].item())
        end = int(expert_offsets[e_t + 1].item())
        if end <= start:
            continue

        if e_t not in w2_cache:
            w2_cache[e_t] = _dequant_gemm2_weight_expert_fp32(
                gemm2_weights[e_t], gemm2_weights_scale[e_t]
            )
        w2e = w2_cache[e_t]  # [H, I] fp32

        se = s[start:end]  # [tk, I]
        ye = torch.mm(se, w2e.t())  # [tk, H]
        ye.mul_(sorted_weights[start:end, None])
        grouped_out[start:end].copy_(ye)

    # 6) Scatter-add
    output = torch.zeros((T, H), device=device, dtype=torch.float32)
    output.index_add_(0, sorted_tokens.to(torch.int64), grouped_out)
    return output.to(torch.bfloat16)