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

//kernel void tensorMultiplication(const device float* tensorA [[buffer(0)]],
//                                 const device float* tensorB [[buffer(1)]],
//                                 device float* tensorC [[buffer(2)]],
//                                 constant const TensorDimensions& dims [[buffer(3)]],
//                                 uint3 gid [[thread_position_in_grid]]) {
//    if (gid.x >= dims.N || gid.y >= dims.M || gid.z >= dims.Batch) {
//        return;
//    }
//
//    float sum = 0.0f;
//
//    uint matrix_size_A = dims.M * dims.K;
//    uint matrix_size_B = dims.K * dims.N;
//    uint batch_offset_A = gid.z * matrix_size_A;
//    uint batch_offset_B = gid.z * matrix_size_B;
//    
//    for (uint k = 0; k < dims.K; ++k) {
//        uint index_A = batch_offset_A + gid.y * dims.K + k;
//        uint index_B = batch_offset_B + k * dims.N + gid.x;
//        
//        sum += tensorA[index_A] * tensorB[index_B];
//    }
//    
//    uint matrix_size_result = dims.M * dims.N;
//    uint result_offset = gid.z * matrix_size_result;
//    uint result_index = result_offset + gid.y * dims.N + gid.x;
//    
//    tensorC[result_index] = sum;
//}

constant uint TILE_SIZE = 16;

kernel void tiledTensorMultiplication(
    const device float* tensorA [[buffer(0)]],
    const device float* tensorB [[buffer(1)]],
    device float* tensorC [[buffer(2)]],
    constant const TensorDimensions& dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    // 1. Allocate ultra-fast shared memory for this specific threadgroup
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    // Pre-calculate batch offsets
    uint batch_offset_A = gid.z * dims.M * dims.K;
    uint batch_offset_B = gid.z * dims.K * dims.N;
    
    uint row = gid.y;
    uint col = gid.x;
    
    float sum = 0.0f;

    // Calculate how many tiles we need to slide across the K dimension
    uint num_tiles = (dims.K + TILE_SIZE - 1) / TILE_SIZE;

    // 2. Slide the tiles across the matrices
    for (uint t = 0; t < num_tiles; ++t) {
        
        // --- COLLABORATIVE LOADING ---
        // Each thread loads exactly ONE element into Tile A
        uint tiled_col_A = t * TILE_SIZE + tid.x;
        if (row < dims.M && tiled_col_A < dims.K) {
            tileA[tid.y][tid.x] = tensorA[batch_offset_A + row * dims.K + tiled_col_A];
        } else {
            tileA[tid.y][tid.x] = 0.0f; // Boundary padding
        }

        // Each thread loads exactly ONE element into Tile B
        uint tiled_row_B = t * TILE_SIZE + tid.y;
        if (tiled_row_B < dims.K && col < dims.N) {
            tileB[tid.y][tid.x] = tensorB[batch_offset_B + tiled_row_B * dims.N + col];
        } else {
            tileB[tid.y][tid.x] = 0.0f; // Boundary padding
        }

        // 3. SYNCHRONIZE
        // Ensure all threads have finished loading the tile before doing math
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- COMPUTE ---
        // Do the dot product using the ultra-fast threadgroup memory
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }

        // 4. SYNCHRONIZE AGAIN
        // Ensure all math is done before the next loop iteration overwrites the tiles
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. WRITE RESULT
    if (row < dims.M && col < dims.N) {
        uint result_offset = gid.z * dims.M * dims.N;
        tensorC[result_offset + row * dims.N + col] = sum;
    }
}

kernel void tensorReLU(const device float* tensorA [[buffer(0)]],
                       device float* tensorB [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    tensorB[gid] = max(0.0f, tensorA[gid]);
}
