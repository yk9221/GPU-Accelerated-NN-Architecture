#include "ActivationFunction.hpp"

Tensor ReLU::forward(Tensor input) {
    return Tensor::relu(input, true);
}

Tensor ReLU::backward(Tensor input, Tensor grad) {
    return grad * Tensor::reluDerivative(input, true);
}

Tensor Softmax::forward(Tensor input) {
    return Tensor::relu(input, true);
}

Tensor Softmax::backward(Tensor input, Tensor grad) {
    return grad * Tensor::reluDerivative(input, true);
}
