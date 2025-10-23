#include <metal_stdlib>
using namespace metal;

kernel void tensorAddition(const device float* tensorA [[buffer(0)]],
                           const device float* tensorB [[buffer(1)]],
                           device float* tensorC [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    tensorC[gid] = tensorA[gid] + tensorB[gid];
}

kernel void tensorElementWiseMultiplication(const device float* tensorA [[buffer(0)]],
                                            const device float* tensorB [[buffer(1)]],
                                            device float* tensorC [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    tensorC[gid] = tensorA[gid] * tensorB[gid];
}

struct TensorDimensions {
    uint Batch;
    uint M;
    uint N;
    uint K;
};

kernel void tensorMultiplication(const device float* tensorA [[buffer(0)]],
                                 const device float* tensorB [[buffer(1)]],
                                 device float* tensorC [[buffer(2)]],
                                 constant const TensorDimensions& dims [[buffer(3)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= dims.N || gid.y >= dims.M || gid.z >= dims.Batch) {
        return;
    }

    float sum = 0.0f;

    uint matrix_size_A = dims.M * dims.K;
    uint matrix_size_B = dims.K * dims.N;
    uint batch_offset_A = gid.z * matrix_size_A;
    uint batch_offset_B = gid.z * matrix_size_B;
    
    for (uint k = 0; k < dims.K; ++k) {
        uint index_A = batch_offset_A + gid.y * dims.K + k;
        uint index_B = batch_offset_B + k * dims.N + gid.x;
        
        sum += tensorA[index_A] * tensorB[index_B];
    }
    
    uint matrix_size_result = dims.M * dims.N;
    uint result_offset = gid.z * matrix_size_result;
    uint result_index = result_offset + gid.y * dims.N + gid.x;
    
    tensorC[result_index] = sum;
}

kernel void tensorReLU(const device float* tensorA [[buffer(0)]],
                       device float* tensorB [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    tensorB[gid] = max(0.0f, tensorA[gid]);
}
