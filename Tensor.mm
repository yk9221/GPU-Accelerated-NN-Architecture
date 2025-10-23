#include "Tensor.hpp"
#import <Metal/Metal.h>

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    t.printData();
    return os;
}

void Tensor::printDataRecursive(std::ostream& os, std::vector<int>& indices, int current_dim) const {
    os << "[";

    for (int i = 0; i < shape[current_dim]; ++i) {
        indices[current_dim] = i;

        if (current_dim == getRank() - 1) {
            os << getValue(indices);
        } else {
            printDataRecursive(os, indices, current_dim + 1);
        }

        if (i < shape[current_dim] - 1) {
            os << ", ";
            if (current_dim < getRank() - 1) {
                os << "\n" << std::string(current_dim + 1, ' ');
            }
        }
    }

    os << "]";
}

void Tensor::printShape() const {
    for(int i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if(i != shape.size() - 1) {
            std::cout << "x";
        }
    }
    std::cout << std::endl;
}

void Tensor::printData() const {
    if (getRank() == 0 || data.empty()) {
        std::cout << (data.empty() ? "[]" : std::to_string(data[0])) << std::endl;
        return;
    }
    
    std::vector<int> initial_indices(getRank(), 0);
    
    printDataRecursive(std::cout, initial_indices, 0);
    std::cout << std::endl;
}

static size_t calculateSize(const std::vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<long long>());
}

Tensor::Tensor() : shape({}), data({}) {}

Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {
    size_t total_size = calculateSize(shape);
    
    data.resize(total_size);
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape(shape), data(data) {}

Tensor::~Tensor() {}

Tensor Tensor::zeros(const std::vector<int>& shape) {
    size_t total_size = calculateSize(shape);
    std::vector<float> data(total_size, 0.0f);
    return Tensor(shape, data);
}

Tensor Tensor::ones(const std::vector<int>& shape) {
    size_t total_size = calculateSize(shape);
    std::vector<float> data(total_size, 1.0f);
    return Tensor(shape, data);
}

Tensor Tensor::full(const std::vector<int>& shape, float value) {
    size_t total_size = calculateSize(shape);
    std::vector<float> data(total_size, value);
    return Tensor(shape, data);
}

Tensor Tensor::random::randn(const std::vector<int>& shape, float mean, float stddev) {
    size_t total_size = calculateSize(shape);
    std::vector<float> data(total_size);
    
    static std::mt19937 generator(std::random_device{}());
    std::normal_distribution<float> distribution(mean, stddev);
    
    for(int i = 0; i < total_size; i++) {
        data[i] = distribution(generator);
    }
    
    return Tensor(shape, data);
}

Tensor Tensor::random::uniform(const std::vector<int>& shape, float low, float high) {
    size_t total_size = calculateSize(shape);
    std::vector<float> data(total_size);
    
    static std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(low, high);
    
    for(int i = 0; i < total_size; i++) {
        data[i] = distribution(generator);
    }
    
    return Tensor(shape, data);
}

int Tensor::getRank() const {
    return static_cast<int>(shape.size());
}

int Tensor::getSize() const {
    return static_cast<int>(data.size());
}

const std::vector<int> Tensor::getShape() const {
    return this->shape;
}

const std::vector<float> Tensor::getData() const {
    return this->data;
}

float* Tensor::getDataPtr() {
    return this->data.data();
}

size_t Tensor::getFlatIndex(const std::vector<int>& indices) const {
    if (indices.size() != getRank()) {
        throw std::runtime_error("Index dimension mismatch.");
    }

    size_t index = 0;
    size_t stride = 1;

    for (int i = getRank() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index is out of bounds.");
        }
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

float Tensor::getValue(const std::vector<int>& indices) const {
    return data[getFlatIndex(indices)];
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    const int rank = getRank();

    if(dim1 < 0 || dim1 >= rank || dim2 < 0 || dim2 >= rank) {
        throw std::runtime_error("Transpose dimensions out of bounds.");
    }
    
    std::vector<int> new_shape = shape;
    std::swap(new_shape[dim1], new_shape[dim2]);
    
    Tensor result(new_shape);
    
    std::vector<int> new_indicies(rank);
    for(int i = 0; i < getSize(); i++) {
        int temp_i = i;
        for(int d = rank - 1; d >= 0; d--) {
            new_indicies[d] = temp_i % new_shape[d];
            temp_i /= new_shape[d];
        }
        
        std::vector<int> original_indicies = new_indicies;
        std::swap(original_indicies[dim1], original_indicies[dim2]);
        
        float value = getValue(original_indicies);
        result.data[i] = value;
    }
    
    return result;
}

Tensor Tensor::sum(int axis, bool keep_dims) const {
    const int rank = this->getRank();
    
    if (axis < 0 || axis >= rank) {
        throw std::runtime_error("Sum axis is out of bounds.");
    }
    
    std::vector<int> result_shape;
    for (int i = 0; i < rank; ++i) {
        if (i == axis) {
            if (keep_dims) {
                result_shape.push_back(1);
            }
        } else {
            result_shape.push_back(this->shape[i]);
        }
    }
    
    if (result_shape.empty()) {
        result_shape.push_back(1);
    }

    Tensor result = Tensor::zeros(result_shape);
    
    for (int i = 0; i < this->getSize(); ++i) {
        std::vector<int> original_indices(rank);
        size_t temp_i = i;
        for (int d = rank - 1; d >= 0; --d) {
            original_indices[d] = temp_i % this->shape[d];
            temp_i /= this->shape[d];
        }
        
        std::vector<int> result_indices;
        for (int d = 0; d < rank; ++d) {
            if (d == axis) {
                if (keep_dims) {
                    result_indices.push_back(0);
                }
            } else {
                result_indices.push_back(original_indices[d]);
            }
        }
        
        size_t result_flat_index = result.getFlatIndex(result_indices);
        result.data[result_flat_index] += this->getValue(original_indices);
    }
    
    return result;
}

Tensor Tensor::add(const Tensor& t1, const Tensor& t2, bool gpu_acceleration) {
    if(t1.getShape() != t2.getShape()) {
        throw std::runtime_error("Shape mismatch");
    }
    
    if(!gpu_acceleration) {
        Tensor result(t1.getShape());
        
        for(int i = 0; i < t1.getSize(); i++) {
            result.data[i] = t1.data[i] + t2.data[i];
        }
        
        return result;
    }
    
    return elementWiseOperations(t1, t2, "tensorAddition");
}

Tensor Tensor::mul(const Tensor& t1, const Tensor& t2, bool gpu_acceleration) {
    if(t1.getShape() != t2.getShape()) {
        throw std::runtime_error("Shape mismatch");
    }
    
    if(!gpu_acceleration) {
        Tensor result(t1.getShape());
        
        for(int i = 0; i < t1.getSize(); i++) {
            result.data[i] = t1.data[i] * t2.data[i];
        }
        
        return result;
    }
    
    return elementWiseOperations(t1, t2, "tensorElementWiseMultiplication");
}

Tensor Tensor::matmul(const Tensor& t1, const Tensor& t2, bool gpu_acceleration) {
    const auto& shape1 = t1.getShape();
    const auto& shape2 = t2.getShape();
    const size_t rank1 = shape1.size();
    const size_t rank2 = shape2.size();

    if (rank1 < 2 || rank2 < 2) {
        throw std::runtime_error("matmul requires tensors of at least rank 2.");
    }

    const int M = shape1[rank1 - 2];
    const int K1 = shape1[rank1 - 1];
    const int K2 = shape2[rank2 - 2];
    const int N = shape2[rank2 - 1];

    if (K1 != K2) {
        throw std::runtime_error("Inner matrix dimensions must agree for matmul.");
    }
    const int K = K1;

    std::vector<int> batch_shape1(shape1.begin(), shape1.end() - 2);
    std::vector<int> batch_shape2(shape2.begin(), shape2.end() - 2);
    if (batch_shape1 != batch_shape2) {
         throw std::runtime_error("Batch dimensions must be identical for matmul.");
    }

    std::vector<int> result_shape = batch_shape1;
    result_shape.push_back(M);
    result_shape.push_back(N);
    
    long long total_batches = std::accumulate(batch_shape1.begin(), batch_shape1.end(), 1LL, std::multiplies<long long>());
    if (total_batches == 0 && !batch_shape1.empty()) total_batches = 1;
    
    Tensor result = Tensor::zeros(result_shape);

    if (!gpu_acceleration) {
        const size_t matrix_size_A = M * K;
        const size_t matrix_size_B = K * N;
        const size_t matrix_size_C = M * N;

        #pragma omp parallel for
        for (int b = 0; b < total_batches; b++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        size_t indexA = b * matrix_size_A + i * K + k;
                        size_t indexB = b * matrix_size_B + k * N + j;
                        sum += t1.getData()[indexA] * t2.getData()[indexB];
                    }
                    size_t indexResult = b * matrix_size_C + i * N + j;
                    result.getDataPtr()[indexResult] = sum;
                }
            }
        }
        return result;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to get Metal device, falling back to CPU." << std::endl;
            matmul(t1, t2, false);
            return result;
        }

        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        id<MTLLibrary> library = [device newDefaultLibrary];
        id<MTLFunction> function = [library newFunctionWithName:@"tensorMultiplication"];
        
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
        
        id<MTLBuffer> bufferA = [device newBufferWithBytes:t1.getData().data() length:t1.getSize() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:t2.getData().data() length:t2.getSize() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:result.getSize() * sizeof(float) options:MTLResourceStorageModeShared];

        struct TensorDimensions {
            uint Batch, M, N, K;
        };
        TensorDimensions dims = { (uint)total_batches, (uint)M, (uint)N, (uint)K };
        id<MTLBuffer> buffer_dims = [device newBufferWithBytes:&dims length:sizeof(TensorDimensions) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        [compute_encoder setComputePipelineState:pipeline_state];
        [compute_encoder setBuffer:bufferA offset:0 atIndex:0];
        [compute_encoder setBuffer:bufferB offset:0 atIndex:1];
        [compute_encoder setBuffer:bufferC offset:0 atIndex:2];
        [compute_encoder setBuffer:buffer_dims offset:0 atIndex:3];

        // The grid depth is the total number of independent matrix multiplications to perform.
        MTLSize grid_size = MTLSizeMake(N, M, total_batches);

        NSUInteger w = [pipeline_state threadExecutionWidth];
        NSUInteger h = [pipeline_state maxTotalThreadsPerThreadgroup] / w;
        MTLSize thread_group_size = MTLSizeMake(w, h, 1);

        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [compute_encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(result.getDataPtr(), [bufferC contents], result.getSize() * sizeof(float));
    }
    
    return result;
}

Tensor Tensor::relu(const Tensor& t, bool gpu_acceleration) {
    if(!gpu_acceleration) {
        Tensor result(t.getShape());
        
        for(int i = 0; i < t.getSize(); i++) {
            result.data[i] = std::max(0.0f, t.data[i]);
        }
        
        return result;
    }
    
    return unaryOperations(t, "tensorReLU");
    
}

Tensor Tensor::reluDerivative(const Tensor& t, bool gpu_acceleration) {
    if(!gpu_acceleration) {
        Tensor result(t.getShape());
        
        for(int i = 0; i < t.getSize(); i++) {
            if(t.data[i] > 0) {
                result.data[i] = 1.0f;
            }
            else {
                result.data[i] = 0.0f;
            }
        }
        
        return result;
    }
    
    return unaryOperations(t, "tensorReLU");
    
}

Tensor Tensor::unaryOperations(const Tensor& t, const char* kernel_name) {
    Tensor result(t.getShape());
    
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to get Metal device, falling back to CPU." << std::endl;
            if(strcmp(kernel_name, "tensorReLU") == 0) {
                return relu(t, false);
            }
            else {
                throw std::runtime_error("Unknown kernel.");
            }
        }
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        
        id<MTLLibrary> library = [device newDefaultLibrary];
        NSError* error = nil;

        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
        
        const size_t data_size = t.getSize() * sizeof(float);

        id<MTLBuffer> bufferA = [device newBufferWithBytes:t.getData().data() length:data_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithLength:data_size options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        [compute_encoder setComputePipelineState:pipeline_state];
        [compute_encoder setBuffer:bufferA offset:0 atIndex:0];
        [compute_encoder setBuffer:bufferB offset:0 atIndex:1];

        MTLSize grid_size = MTLSizeMake(t.getSize(), 1, 1);
        NSUInteger thread_group_size_max = [pipeline_state maxTotalThreadsPerThreadgroup];
        if (thread_group_size_max > t.getSize()) {
            thread_group_size_max = t.getSize();
        }
        MTLSize thread_group_size = MTLSizeMake(thread_group_size_max, 1, 1);

        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [compute_encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(result.getDataPtr(), [bufferB contents], data_size);
    }
    
    return result;
}

Tensor Tensor::elementWiseOperations(const Tensor& t1, const Tensor& t2, const char* kernel_name) {
    Tensor result(t1.getShape());
    
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to get Metal device, falling back to CPU." << std::endl;
            if(strcmp(kernel_name, "tensorAddition") == 0) {
                return add(t1, t2, false);
            }
            else if(strcmp(kernel_name, "tensorElementWiseMultiplication") == 0) {
                return mul(t1, t2, false);
            }
            else {
                throw std::runtime_error("Unknown kernel.");
            }
            return add(t1, t2, false);
        }
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        
        id<MTLLibrary> library = [device newDefaultLibrary];
        NSError* error = nil;

        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
        
        const size_t data_size = t1.getSize() * sizeof(float);

        id<MTLBuffer> bufferA = [device newBufferWithBytes:t1.getData().data() length:data_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:t2.getData().data() length:data_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:data_size options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        [compute_encoder setComputePipelineState:pipeline_state];
        [compute_encoder setBuffer:bufferA offset:0 atIndex:0];
        [compute_encoder setBuffer:bufferB offset:0 atIndex:1];
        [compute_encoder setBuffer:bufferC offset:0 atIndex:2];

        MTLSize grid_size = MTLSizeMake(t1.getSize(), 1, 1);
        NSUInteger thread_group_size_max = [pipeline_state maxTotalThreadsPerThreadgroup];
        if (thread_group_size_max > t1.getSize()) {
            thread_group_size_max = t1.getSize();
        }
        MTLSize thread_group_size = MTLSizeMake(thread_group_size_max, 1, 1);

        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [compute_encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(result.getDataPtr(), [bufferC contents], data_size);
    }
    
    return result;
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return Tensor::add(lhs, rhs, true);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return Tensor::mul(lhs, rhs, true);
}
