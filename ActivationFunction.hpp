#ifndef ACTIVATIONFUNCTION_HPP
#define ACTIVATIONFUNCTION_HPP

#include "Tensor.hpp"

class ReLU {
public:
    Tensor forward(Tensor input);
    Tensor backward(Tensor input, Tensor grad);
};

class Softmax {
public:
    Tensor forward(Tensor input);
    Tensor backward(Tensor input, Tensor grad);
};

#endif
