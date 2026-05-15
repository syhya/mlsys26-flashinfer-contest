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

    token_idx_ptr = sorted_tokens_ptr + start_idx + m_offs
    token_ids = tl.load(token_idx_ptr, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_base = tl.arange(0, BLOCK_K).to(tl.int64)

    for k in range(0, H, BLOCK_K):
        k_offs = k_base + k
        k_idx = k // BLOCK_K

        A_ptrs = A_ptr + token_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        W_ptrs = (
            W_ptr
            + e_idx.to(tl.int64) * stride_we
            + n_offs[None, :] * stride_wn
            + k_offs[:, None] * stride_wk
        )

        a = tl.load(A_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(W_ptrs, mask=n_mask[None, :], other=0.0)

        a_scale_ptrs = (
            A_scale_ptr
            + k_idx * stride_ascale_k
            + token_ids[:, None] * stride_ascale_m
        )
        a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=0.0)

        scale_n_offs = n_offs // 128
        w_scale_ptrs = (
            W_scale_ptr
            + e_idx.to(tl.int64) * stride_wscale_e
            + scale_n_offs[None, :] * stride_wscale_n
            + k_idx * stride_wscale_k
        )
        w_scale = tl.load(w_scale_ptrs, mask=n_mask[None, :], other=0.0)

        dot_res = tl.dot(a, w, out_dtype=tl.float32)
        acc += dot_res * a_scale * w_scale

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

    s = torch.sigmoid(routing_logits.float())
    s_bias = s + routing_bias.float().view(1, E_global)

    grouped = s_bias.view(T, N_GROUP, group_size)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)
    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), dtype=torch.bool, device=routing_logits.device)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), device=routing_logits.device, dtype=torch.float32)
    scores_pruned = torch.where(expert_mask, s_bias, neg_inf)

    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    topk_mask = torch.zeros((T, E_global), dtype=torch.bool, device=routing_logits.device)
    topk_mask.scatter_(1, topk_idx, True)
    weights = torch.where(topk_mask, s, torch.zeros((), device=s.device, dtype=s.dtype))
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor
    topk_weights = torch.gather(weights, 1, topk_idx)
    return topk_idx, topk_weights


def _build_local_dispatch(topk_idx, topk_weights, local_expert_offset, e_local):
    device = topk_idx.device
    local_start = int(local_expert_offset)
    mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    token_idx, slot_idx = torch.nonzero(mask, as_tuple=True)

    if token_idx.numel() == 0:
        return (
            torch.empty((0,), device=device, dtype=torch.int32),
            torch.empty((0,), device=device, dtype=torch.int32),
            torch.empty((0,), device=device, dtype=torch.float32),
            torch.zeros((e_local + 1,), device=device, dtype=torch.int32),
        )

    local_experts = (topk_idx[token_idx, slot_idx] - local_start).to(torch.int32)
    local_weights = topk_weights[token_idx, slot_idx].to(torch.float32)

    order = torch.argsort(local_experts)
    sorted_tokens = token_idx[order].to(torch.int32)
    sorted_experts = local_experts[order]
    sorted_weights = local_weights[order]

    counts = torch.bincount(sorted_experts, minlength=e_local)
    offsets = torch.zeros((e_local + 1,), device=device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return sorted_tokens, sorted_experts, sorted_weights, offsets


def _dequant_gemm2_weight_expert(w_fp8, w_scale):
    # w_fp8: [H, I], scales: [H//128, I//128]
    H, I = w_fp8.shape
    hb = H // 128
    ib = I // 128
    w = w_fp8.reshape(hb, 128, ib, 128).permute(0, 2, 1, 3).contiguous().float()
    w = w * w_scale[:, :, None, None]
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
    T = routing_logits.shape[0]
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    topk_idx, topk_weights = _route_noaux_topk8(
        routing_logits, routing_bias, routed_scaling_factor
    )

    sorted_tokens, sorted_experts, sorted_weights, expert_offsets = _build_local_dispatch(
        topk_idx, topk_weights, local_expert_offset, E_local
    )

    Tk_total = sorted_tokens.numel()
    if Tk_total == 0:
        return torch.zeros((T, H), device=device, dtype=torch.bfloat16)

    G1 = torch.zeros((Tk_total, 2 * I), device=device, dtype=torch.float32)

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

    O_grouped = torch.zeros((Tk_total, H), device=device, dtype=torch.float32)

    for e in range(E_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        S_e = S[start:end]
        W2_e = _dequant_gemm2_weight_expert(gemm2_weights[e], gemm2_weights_scale[e])
        Y_e = torch.mm(S_e, W2_e.t())
        Y_e.mul_(sorted_weights[start:end].unsqueeze(1))
        O_grouped[start:end] = Y_e

    output = torch.zeros((T, H), device=device, dtype=torch.float32)
    output.index_add_(0, sorted_tokens.long(), O_grouped)
    return output.to(torch.bfloat16)