import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["N_dim"],
)
@triton.jit
def grouped_gemm1_kernel(
    A_ptr,
    A_scale_ptr,
    W_ptr,
    W_scale_ptr,
    Out_ptr,
    sorted_tokens_ptr,
    expert_offsets_ptr,
    H,
    N_dim,
    stride_am,
    stride_ak,
    stride_ascale_k,
    stride_ascale_m,
    stride_we,
    stride_wn,
    stride_wk,
    stride_wscale_e,
    stride_wscale_n,
    stride_wscale_k,
    stride_outm,
    stride_outn,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    e_idx = tl.program_id(2)

    start_idx = tl.load(expert_offsets_ptr + e_idx)
    end_idx = tl.load(expert_offsets_ptr + e_idx + 1)
    Tk = end_idx - start_idx

    if m_idx * BLOCK_M >= Tk:
        return

    m_offs = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < Tk

    n_offs = (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offs < N_dim

    token_ids = tl.load(sorted_tokens_ptr + start_idx + m_offs, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_offs = tl.arange(0, BLOCK_K).to(tl.int64)

    a_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
    w_ptrs = (
        W_ptr
        + e_idx.to(tl.int64) * stride_we
        + n_offs[None, :] * stride_wn
        + k_offs[:, None] * stride_wk
    )

    scale_n_offs = n_offs // 128
    w_scale_base = (
        W_scale_ptr
        + e_idx.to(tl.int64) * stride_wscale_e
        + scale_n_offs[None, :] * stride_wscale_n
    )
    a_scale_base = A_scale_ptr + token_ids[:, None] * stride_ascale_m

    for k in range(0, H, BLOCK_K):
        k_idx = k // BLOCK_K

        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)

        a_scale = tl.load(
            a_scale_base + k_idx * stride_ascale_k,
            mask=m_mask[:, None],
            other=0.0,
        )
        w_scale = tl.load(
            w_scale_base + k_idx * stride_wscale_k,
            mask=n_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(a, w, out_dtype=tl.float32) * a_scale * w_scale

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = (
        Out_ptr
        + (start_idx + m_offs)[:, None].to(tl.int64) * stride_outm
        + n_offs[None, :] * stride_outn
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _route_noaux_topk8(routing_logits, routing_bias, routed_scaling_factor):
    T, E_global = routing_logits.shape
    N_GROUP = 8
    TOPK_GROUP = 4
    TOP_K = 8
    group_size = E_global // N_GROUP

    logits = routing_logits.float()
    bias = routing_bias.float().view(1, E_global)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)

    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=routing_logits.device)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), device=routing_logits.device, dtype=torch.float32)
    pruned = torch.where(expert_mask, s_with_bias, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    topk_scores = s.gather(1, topk_idx)
    topk_weights = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_idx, topk_weights


def _build_local_dispatch(topk_idx, topk_weights, local_expert_offset, e_local):
    local_start = int(local_expert_offset)
    mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    token_idx, slot_idx = torch.nonzero(mask, as_tuple=True)
    tk_total = token_idx.numel()
    if tk_total == 0:
        return None

    local_expert_idx = (topk_idx[token_idx, slot_idx] - local_start).to(torch.int64)
    local_weights = topk_weights[token_idx, slot_idx].to(torch.float32)

    order = torch.argsort(local_expert_idx)
    sorted_tokens = token_idx[order]
    sorted_experts = local_expert_idx[order]
    sorted_weights = local_weights[order]

    expert_counts = torch.bincount(sorted_experts, minlength=e_local)
    expert_offsets = torch.zeros(e_local + 1, dtype=torch.int64, device=topk_idx.device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return (
        sorted_tokens.to(torch.int32),
        sorted_experts.to(torch.int32),
        sorted_weights,
        expert_offsets,
        int(tk_total),
    )


def _dequant_gemm2_weight_expert(w_fp8, w_scale):
    # w_fp8: [H, I], scales: [H//128, I//128]
    H, I = w_fp8.shape
    w = w_fp8.float().view(H // 128, 128, I // 128, 128)
    s = w_scale.float().view(H // 128, I // 128)
    w = w * s[:, None, :, None]
    return w.permute(0, 2, 1, 3).reshape(H, I).contiguous()


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
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    topk_idx, topk_weights = _route_noaux_topk8(
        routing_logits, routing_bias, routed_scaling_factor
    )

    dispatch = _build_local_dispatch(topk_idx, topk_weights, local_expert_offset, E_local)
    if dispatch is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_tokens, sorted_experts, sorted_weights, expert_offsets, Tk_total = dispatch

    G1 = torch.zeros((Tk_total, 2 * I), dtype=torch.float32, device=device)

    grid_gemm1 = lambda META: (
        triton.cdiv(2 * I, META["BLOCK_N"]),
        triton.cdiv(Tk_total, META["BLOCK_M"]),
        E_local,
    )

    grouped_gemm1_kernel[grid_gemm1](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        W_ptr=gemm1_weights,
        W_scale_ptr=gemm1_weights_scale,
        Out_ptr=G1,
        sorted_tokens_ptr=sorted_tokens,
        expert_offsets_ptr=expert_offsets,
        H=H,
        N_dim=2 * I,
        stride_am=hidden_states.stride(0),
        stride_ak=hidden_states.stride(1),
        stride_ascale_k=hidden_states_scale.stride(0),
        stride_ascale_m=hidden_states_scale.stride(1),
        stride_we=gemm1_weights.stride(0),
        stride_wn=gemm1_weights.stride(1),
        stride_wk=gemm1_weights.stride(2),
        stride_wscale_e=gemm1_weights_scale.stride(0),
        stride_wscale_n=gemm1_weights_scale.stride(1),
        stride_wscale_k=gemm1_weights_scale.stride(2),
        stride_outm=G1.stride(0),
        stride_outn=G1.stride(1),
        BLOCK_K=128,
    )

    x1 = G1[:, :I]
    x2 = G1[:, I:]
    S = F.silu(x2) * x1

    O = torch.zeros((Tk_total, H), dtype=torch.float32, device=device)

    # correctness-first GEMM2 per expert, avoiding unstable dynamic requantization
    for e in range(E_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue
        s_e = S[start:end].contiguous()
        w2_e = _dequant_gemm2_weight_expert(gemm2_weights[e], gemm2_weights_scale[e])
        o_e = torch.mm(s_e, w2_e.t())
        o_e.mul_(sorted_weights[start:end].unsqueeze(1))
        O[start:end] = o_e

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    output.index_add_(0, sorted_tokens.long(), O)
    return output.to(torch.bfloat16)