#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <numeric>
#include <functional>
#include <iostream>
#include <random>

class Tensor {
friend class Linear;
private:
    std::vector<int> shape;
    std::vector<float> data;
    
    // --- Constructors ---
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);
    
    // --- Print ---
    void printDataRecursive(std::ostream& os, std::vector<int>& indices, int current_dim) const;
    
    // --- Getter ---
    size_t getFlatIndex(const std::vector<int>& indices) const;
    float getValue(const std::vector<int>& indices) const;
    
    // --- Operations ---
    static Tensor unaryOperations(const Tensor& t, const char* kernel_name);
    static Tensor elementWiseOperations(const Tensor& t1, const Tensor& t2, const char* kernel_name);
public:
    // --- Initializers ---
    static Tensor zeros(const std::vector<int>& shape);
    static Tensor ones(const std::vector<int>& shape);
    static Tensor full(const std::vector<int>& shape, float value=0.0);
    struct random {
        static Tensor randn(const std::vector<int>& shape, float mean=0.0, float stddev=1.0);
        static Tensor uniform(const std::vector<int>& shape, float low=0.0, float high=1.0);
    };
    
    // --- Destructor ---
    ~Tensor();
    
    // --- Print ---
    void printShape() const;
    void printData() const;
    
    // --- Getter ---
    int getRank() const;
    int getSize() const;
    const std::vector<int> getShape() const;
    const std::vector<float> getData() const;
    float* getDataPtr();
    
    // --- Operations ---
    Tensor transpose(int dim1=0, int dim2=1) const;
    Tensor sum(int axis, bool keep_dims=false) const;
    static Tensor add(const Tensor& t1, const Tensor& t2, bool gpu_acceleration=false);
    static Tensor mul(const Tensor& t1, const Tensor& t2, bool gpu_acceleration=false);
    static Tensor matmul(const Tensor& t1, const Tensor& t2, bool gpu_acceleration=false);
    static Tensor relu(const Tensor& t, bool gpu_acceleration=false);
    static Tensor reluDerivative(const Tensor& t, bool gpu_acceleration=false);
    
};

// --- Print Overload ---
std::ostream& operator<<(std::ostream& os, const Tensor& t);
Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);

#endif
