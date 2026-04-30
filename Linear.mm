#include "Linear.hpp"

#include <cmath>

Linear::Linear(int input_dim, int output_dim) {
    weight = Tensor::random::randn({output_dim, input_dim}) * Tensor::full({output_dim, input_dim}, std::sqrt(2.0 / input_dim));
    bias = Tensor::zeros({output_dim, 1});
    
    grad_weight = Tensor::zeros(weight.getShape());
    grad_bias = Tensor::zeros(bias.getShape());
}

Tensor Linear::forward(Tensor input) {
    return Tensor::matmul(weight, input, true) + bias;
}

Tensor Linear::backward(Tensor input, Tensor grad) {
    grad_weight = grad_weight + Tensor::matmul(input.transpose(), grad);
    grad_bias = grad_bias + grad.sum(0, true);
    
    return Tensor::matmul(grad, weight.transpose());
}

void Linear::zeroGrad() {
    grad_weight = Tensor::zeros(weight.getShape());
    grad_weight = Tensor::zeros(bias.getShape());
}
