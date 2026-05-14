#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")

void launchFusedGatingKernel(
    void* routing_logits,
    void* routing_bias,
    float routing_scaling_factor,
    void* routing_idx,
    void* routing_weights,
    int seq_len
);

void launchCountExpertAndOffsetsKernel(
    void* routing_idx,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    int seq_len,
    int local_expert_offset
);

void launchPermuteKernel(
    void* routing_idx,
    void* routing_weight,
    void* expert_offsets,
    void* out_token_idx,
    void* out_weights,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
);

void launchCountScanPermuteKernel(
    void* routing_idx,
    void* routing_weight,
    void* expert_counts,
    void* expert_offsets,
    void* total_tokens,
    void* out_token_idx,
    void* out_weights,
    void* token2permuted_idx,     // [seq_len * 8]
    void* token_counts,
    int seq_len,
    int local_expert_offset
);


void launchMoePermuteCopyFp8WithScaleKernel(
    void* input,
    void* input_scale,
    void* permuted_token_idx,
    void* output,
    void* output_scale,
    void* offset,
    int input_seq_len
);

void launchScatterAddKernel(
    void* tmp_output,
    void* token_idx,
    void* output,
    void* offset,
    int seq_len
);

void launchActQuantKernel(
    void* input,
    void* output,
    void* scale,
    void* offset,
    int seq_len
);

void launchBuildPaddedOffsetsKernel(
    void* offsets,
    void* padded_offsets
);

void launchPadFp8HiddenScaleKernel(
    void* src_hidden,
    void* src_scale,
    void* offsets,
    void* padded_offsets,
    void* dst_hidden,
    void* dst_scale,
    int src_scale_stride,
    int dst_scale_stride,
    int dst_capacity
);

void launchTransposeScaleMnToKKernel(
    void* src,
    void* dst,
    int rows,
    int scale_blocks
);

void launchUnpadBf16RowsKernel(
    void* src,
    void* offsets,
    void* padded_offsets,
    void* dst,
    int dst_capacity
);

void launchScaleBf16RowsByWeightKernel(
    void* data,
    void* weights,
    int rows
);

bool launchCooperativeKernel(
    void* routing_logits,
    void* routing_bias,
    float routing_scaling_factor,
    void* routing_idx,
    void* routing_weights,
    void* input,
    void* input_scale,
    void* expert_counts,
    void* expert_offsets,
    // void* total_tokens,
    void* out_token_idx,
    void* out_weights,
    void* output,
    void* output_scale,
    void* token2permuted_idx,
    void* token_counts,
    int seq_len,
    int local_expert_offset
);

void fusedRoutePermuteCopyIntoWrapper(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    float routing_scaling_factor,
    torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale,
    int64_t local_expert_offset,
    torch::Tensor routing_idx,
    torch::Tensor routing_weights,
    torch::Tensor expert_counts,
    torch::Tensor expert_offsets,
    torch::Tensor total_tokens_device,
    torch::Tensor permute_token_idx,
    torch::Tensor permute_weight,
    torch::Tensor permute_hidden_states,
    torch::Tensor permute_hidden_states_scale,
    torch::Tensor token2permuted_idx,
    torch::Tensor token_counts,
    int seq_len
) {
    CHECK_CUDA(routing_logits);
    CHECK_CUDA(routing_bias);
    CHECK_CUDA(hidden_states);
    CHECK_CUDA(hidden_states_scale);
    CHECK_CUDA(routing_idx);
    CHECK_CUDA(routing_weights);
    CHECK_CUDA(expert_counts);
    CHECK_CUDA(expert_offsets);
    CHECK_CUDA(total_tokens_device);
    CHECK_CUDA(permute_token_idx);
    CHECK_CUDA(permute_weight);
    CHECK_CUDA(permute_hidden_states);
    CHECK_CUDA(permute_hidden_states_scale);

    CHECK_CONTIGUOUS(routing_logits);
    CHECK_CONTIGUOUS(routing_bias);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(hidden_states_scale);
    CHECK_CONTIGUOUS(routing_idx);
    CHECK_CONTIGUOUS(routing_weights);
    CHECK_CONTIGUOUS(expert_counts);
    CHECK_CONTIGUOUS(expert_offsets);
    CHECK_CONTIGUOUS(total_tokens_device);
    CHECK_CONTIGUOUS(permute_token_idx);
    CHECK_CONTIGUOUS(permute_weight);
    CHECK_CONTIGUOUS(permute_hidden_states);
    CHECK_CONTIGUOUS(permute_hidden_states_scale);

    CHECK_FLOAT32(routing_logits);
    CHECK_FLOAT32(hidden_states_scale);
    CHECK_INT32(routing_idx);
    CHECK_FLOAT32(routing_weights);
    CHECK_INT32(expert_counts);
    CHECK_INT32(expert_offsets);
    CHECK_INT32(total_tokens_device);
    CHECK_INT32(permute_token_idx);
    CHECK_FLOAT32(permute_weight);
    CHECK_FLOAT32(permute_hidden_states_scale);

    TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must be [seq_len, 256]");
    TORCH_CHECK(routing_logits.size(1) == 256, "routing_logits second dim must be 256");
    TORCH_CHECK(routing_bias.numel() == 256, "routing_bias must have 256 elements");
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [seq_len, 7168]");
    TORCH_CHECK(hidden_states.size(1) == 7168, "hidden_states second dim must be 7168");
    TORCH_CHECK(hidden_states.element_size() == 1, "hidden_states must be 1-byte dtype for fp8 kernel");
    TORCH_CHECK(hidden_states_scale.dim() == 2, "hidden_states_scale must be [56, seq_len]");
    TORCH_CHECK(hidden_states_scale.size(0) == 56, "hidden_states_scale first dim must be 56");
    TORCH_CHECK(hidden_states_scale.size(1) == hidden_states.size(0), "hidden_states_scale second dim must equal hidden_states seq_len");
    TORCH_CHECK(routing_logits.size(0) == hidden_states.size(0), "routing_logits and hidden_states seq_len must match");

    if (seq_len < 128) {
        launchCooperativeKernel(
            routing_logits.data_ptr<float>(),
            routing_bias.data_ptr<at::BFloat16>(),
            routing_scaling_factor,
            routing_idx.data_ptr<int>(),
            routing_weights.data_ptr<float>(),
            hidden_states.data_ptr(),
            hidden_states_scale.data_ptr<float>(),
            expert_counts.data_ptr<int>(),
            expert_offsets.data_ptr<int>(),
            // total_tokens.data_ptr<int>(),
            permute_token_idx.data_ptr<int>(),
            permute_weight.data_ptr<float>(),
            permute_hidden_states.data_ptr(),
            permute_hidden_states_scale.data_ptr<float>(),
            token2permuted_idx.data_ptr<int>(),
            token_counts.data_ptr<int>(),
            seq_len,
            static_cast<int>(local_expert_offset)
        );
        return;
    }


    launchFusedGatingKernel(
        routing_logits.data_ptr<float>(),
        routing_bias.data_ptr<at::BFloat16>(),
        routing_scaling_factor,
        routing_idx.data_ptr<int>(),
        routing_weights.data_ptr<float>(),
        static_cast<int>(seq_len)
    );

    if (seq_len > 256 && seq_len != 901) {
        launchCountExpertAndOffsetsKernel(
            routing_idx.data_ptr<int>(),
            expert_counts.data_ptr<int>(),
            expert_offsets.data_ptr<int>(),
            total_tokens_device.data_ptr<int>(),
            static_cast<int>(seq_len),
            static_cast<int>(local_expert_offset)
        );
        launchPermuteKernel(
            routing_idx.data_ptr<int>(),
            routing_weights.data_ptr<float>(),
            expert_counts.data_ptr<int>(),
            permute_token_idx.data_ptr<int>(),
            permute_weight.data_ptr<float>(),
            token2permuted_idx.data_ptr<int>(),
            token_counts.data_ptr<int>(),
            static_cast<int>(seq_len),
            static_cast<int>(local_expert_offset)
        );
    } else {
        launchCountScanPermuteKernel(
        routing_idx.data_ptr<int>(),
        routing_weights.data_ptr<float>(),
        expert_counts.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        total_tokens_device.data_ptr<int>(),
        permute_token_idx.data_ptr<int>(),
        permute_weight.data_ptr<float>(),
        token2permuted_idx.data_ptr<int>(),
        token_counts.data_ptr<int>(),
        static_cast<int>(seq_len),
        static_cast<int>(local_expert_offset)
    );
    }

    launchMoePermuteCopyFp8WithScaleKernel(
        hidden_states.data_ptr(),
        hidden_states_scale.data_ptr<float>(),
        permute_token_idx.data_ptr<int>(),
        permute_hidden_states.data_ptr(),
        permute_hidden_states_scale.data_ptr<float>(),
        expert_offsets.data_ptr<int>(),
        static_cast<int>(seq_len)
    );
}

void buildPaddedOffsetsWrapper(
    torch::Tensor offsets,
    torch::Tensor padded_offsets
) {
    CHECK_CUDA(offsets);
    CHECK_CUDA(padded_offsets);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(padded_offsets);
    CHECK_INT32(offsets);
    CHECK_INT32(padded_offsets);
    TORCH_CHECK(offsets.numel() == 33, "offsets must have 33 elements");
    TORCH_CHECK(padded_offsets.numel() == 33, "padded_offsets must have 33 elements");

    launchBuildPaddedOffsetsKernel(
        offsets.data_ptr<int>(),
        padded_offsets.data_ptr<int>()
    );
}

void padFp8HiddenScaleWrapper(
    torch::Tensor src_hidden,
    torch::Tensor src_scale,
    torch::Tensor offsets,
    torch::Tensor padded_offsets,
    torch::Tensor dst_hidden,
    torch::Tensor dst_scale,
    int64_t src_scale_stride,
    int64_t dst_scale_stride,
    int64_t dst_capacity
) {
    CHECK_CUDA(src_hidden);
    CHECK_CUDA(src_scale);
    CHECK_CUDA(offsets);
    CHECK_CUDA(padded_offsets);
    CHECK_CUDA(dst_hidden);
    CHECK_CUDA(dst_scale);
    CHECK_CONTIGUOUS(src_hidden);
    CHECK_CONTIGUOUS(src_scale);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(padded_offsets);
    CHECK_CONTIGUOUS(dst_hidden);
    CHECK_CONTIGUOUS(dst_scale);
    CHECK_FLOAT32(src_scale);
    CHECK_FLOAT32(dst_scale);
    CHECK_INT32(offsets);
    CHECK_INT32(padded_offsets);
    TORCH_CHECK(src_hidden.element_size() == 1, "src_hidden must be fp8-sized");
    TORCH_CHECK(dst_hidden.element_size() == 1, "dst_hidden must be fp8-sized");

    launchPadFp8HiddenScaleKernel(
        src_hidden.data_ptr(),
        src_scale.data_ptr<float>(),
        offsets.data_ptr<int>(),
        padded_offsets.data_ptr<int>(),
        dst_hidden.data_ptr(),
        dst_scale.data_ptr<float>(),
        static_cast<int>(src_scale_stride),
        static_cast<int>(dst_scale_stride),
        static_cast<int>(dst_capacity)
    );
}

void transposeScaleMnToKWrapper(
    torch::Tensor src,
    torch::Tensor dst,
    int64_t rows,
    int64_t scale_blocks
) {
    CHECK_CUDA(src);
    CHECK_CUDA(dst);
    CHECK_CONTIGUOUS(src);
    CHECK_CONTIGUOUS(dst);
    CHECK_FLOAT32(src);
    CHECK_FLOAT32(dst);
    TORCH_CHECK(rows >= 0, "rows must be non-negative");
    TORCH_CHECK(scale_blocks > 0, "scale_blocks must be positive");
    TORCH_CHECK(src.numel() >= rows * scale_blocks, "src is too small");
    TORCH_CHECK(dst.numel() >= rows * scale_blocks, "dst is too small");

    launchTransposeScaleMnToKKernel(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(scale_blocks)
    );
}

void unpadBf16RowsWrapper(
    torch::Tensor src,
    torch::Tensor offsets,
    torch::Tensor padded_offsets,
    torch::Tensor dst,
    int64_t dst_capacity
) {
    CHECK_CUDA(src);
    CHECK_CUDA(offsets);
    CHECK_CUDA(padded_offsets);
    CHECK_CUDA(dst);
    CHECK_CONTIGUOUS(src);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(padded_offsets);
    CHECK_CONTIGUOUS(dst);
    CHECK_INT32(offsets);
    CHECK_INT32(padded_offsets);
    TORCH_CHECK(src.scalar_type() == torch::kBFloat16, "src must be bf16");
    TORCH_CHECK(dst.scalar_type() == torch::kBFloat16, "dst must be bf16");

    launchUnpadBf16RowsKernel(
        src.data_ptr<at::BFloat16>(),
        offsets.data_ptr<int>(),
        padded_offsets.data_ptr<int>(),
        dst.data_ptr<at::BFloat16>(),
        static_cast<int>(dst_capacity)
    );
}

void scaleBf16RowsByWeightWrapper(
    torch::Tensor data,
    torch::Tensor weights,
    int64_t rows
) {
    CHECK_CUDA(data);
    CHECK_CUDA(weights);
    CHECK_CONTIGUOUS(data);
    CHECK_CONTIGUOUS(weights);
    CHECK_FLOAT32(weights);
    TORCH_CHECK(data.scalar_type() == torch::kBFloat16, "data must be bf16");
    TORCH_CHECK(data.dim() == 2, "data must be [rows, hidden]");
    TORCH_CHECK(data.size(1) == 7168, "data hidden dim must be 7168");
    TORCH_CHECK(rows >= 0, "rows must be non-negative");
    TORCH_CHECK(data.size(0) >= rows, "data has fewer rows than requested");
    TORCH_CHECK(weights.numel() >= rows, "weights has fewer elements than requested");

    launchScaleBf16RowsByWeightKernel(
        data.data_ptr(),
        weights.data_ptr<float>(),
        static_cast<int>(rows)
    );
}

void scatterAddWrapper(
    torch::Tensor tmp_output,
    torch::Tensor token_idx,
    torch::Tensor output,
    torch::Tensor offset,
    int seq_len
) {
    CHECK_CUDA(tmp_output);
    CHECK_CUDA(token_idx);
    CHECK_CUDA(output);
    CHECK_CUDA(offset);
    CHECK_CONTIGUOUS(tmp_output);
    CHECK_CONTIGUOUS(token_idx);
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(offset);
    CHECK_FLOAT32(tmp_output);
    CHECK_INT32(token_idx);
    CHECK_INT32(offset);

    TORCH_CHECK(tmp_output.dim() == 2, "tmp_output must be [TotalValidTokens, HiddenDim]");
    TORCH_CHECK(token_idx.dim() == 1, "token_idx must be [TotalValidTokens]");
    TORCH_CHECK(output.dim() == 2, "output must be [SeqLen, HiddenDim]");
    // TORCH_CHECK(offset.numel() == 1, "offset must have 1 element");
    TORCH_CHECK(tmp_output.size(0) == token_idx.size(0), "tmp_output and token_idx first dim must match");
    TORCH_CHECK(tmp_output.size(1) == output.size(1), "tmp_output and output second dim must match");

    launchScatterAddKernel(
        tmp_output.data_ptr<float>(),
        token_idx.data_ptr<int>(),
        output.data_ptr<float>(),
        offset.data_ptr<int>(),
        seq_len
    );
}

void actQuantWrapper(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    torch::Tensor offset,
    int seq_len
) {
    launchActQuantKernel(
        input.data_ptr(),
        output.data_ptr(),
        scale.data_ptr(),
        offset.data_ptr(),
        seq_len
    );
}

void launchReduceAddKernel(
    void* tmp_output,
    void* token2permuted_idx,
    void* token_counts,
    void* output,
    int seq_len
);

void reduceAddWrapper(
    torch::Tensor tmp_output,
    torch::Tensor token2permuted_idx,
    torch::Tensor token_counts,
    torch::Tensor output,
    int seq_len
) {
    launchReduceAddKernel(
        tmp_output.data_ptr(),
        token2permuted_idx.data_ptr<int>(),
        token_counts.data_ptr<int>(),
        output.data_ptr<at::BFloat16>(),
        seq_len
    );
}
