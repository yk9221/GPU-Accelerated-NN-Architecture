#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "Tensor.hpp"

class Linear {
private:
    Tensor weight;
    Tensor bias;
    Tensor grad_weight;
    Tensor grad_bias;
public:
    Linear(int input_dim, int output_dim);
    Tensor forward(Tensor input);
    Tensor backward(Tensor input, Tensor grad);
    void zeroGrad();
};

#endif
