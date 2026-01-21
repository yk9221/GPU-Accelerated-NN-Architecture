# GPU-Accelerated Deep Learning Framework

## ⚡ Overview
High-performance deep learning library built from scratch in C++. Unlike standard educational implementations, it is designed for speed and scalability, leveraging Apple's **Metal API** to offload computationally intensive linear algebra operations to the GPU.

The core philosophy of this project is **zero-abstraction**: implementing every component—from the matrix multiplication kernels (MSL) to the automatic differentiation engine—without relying on high-level frameworks like PyTorch or TensorFlow.

## 🚀 Key Features

### 🖥️ High-Performance Backend
* **Dual-Backend Architecture:** Seamlessly toggles between **CPU (Host)** execution for small-scale debugging and **GPU (Device)** execution for large-scale training.
* **Metal Integration:** Uses an **Objective-C++ Bridge** to interface standard C++ tensors with Apple's Metal Performance Shaders and custom compute kernels.
* **Custom Compute Kernels:** Hand-written **Metal Shading Language (MSL)** kernels for $O(N^3)$ operations like Matrix Multiplication (GEMM) and Convolution.

### 🧠 Core Deep Learning Primitives
* **Tensor Engine:** N-dimensional array manipulation with strided memory views and automatic broadcasting.
* **Autograd System:** Reverse-mode automatic differentiation engine that builds dynamic computational graphs.
* **Optimizers:** First-principle implementations of **SGD**, **Momentum**, and **Adam** that manage state on the GPU to minimize host-to-device data transfer overhead.

## 🏗️ System Architecture
The library is designed with a clear separation of concerns to handle the complexity of mixed-language programming (C++ and Objective-C).

1.  **Frontend (C++17):** The user-facing API. Handles tensor shapes, graph construction, and high-level logic.
2.  **Bridge (Objective-C++):** A `.mm` translation layer that manages `MTLDevice`, `MTLCommandQueue`, and `MTLBuffer` references.
3.  **Backend (Metal MSL):** `.metal` shader files containing the parallelized math logic executed by the GPU threads.

## 💻 Installation

### Prerequisites
* macOS (10.13+) with Metal-supported GPU.
* Clang/LLVM compiler.
* CMake (3.10+).

## 🧠 Usage Example

### Defining a Network (C++ API)
```cpp
#include "metalnet.hpp"

int main() {
    // Initialize the GPU context
    Device::set_default(DeviceType::GPU);

    // Create tensors on the GPU
    Tensor inputs = Tensor::randn({64, 784}); 
    Tensor weights = Tensor::randn({784, 10}, /*requires_grad=*/true);
    
    // Forward Pass (Executed on GPU)
    Tensor logits = Tensor::matmul(inputs, weights);
    Tensor probs = Tensor::softmax(logits);
    
    // Backward Pass (Gradients computed via Metal kernels)
    Tensor loss = CrossEntropy(probs, targets);
    loss.backward();
    
    // Optimizer Step
    Adam optimizer(weights, 0.001);
    optimizer.step();

    return 0;
}
```

## 🗺️ Roadmap & Status

### 🏗️ Core Infrastructure
- [x] **Tensor Engine:** N-dimensional strided memory views, broadcasting, and zero-copy slicing.
- [x] **Autograd Engine:** Reverse-mode automatic differentiation with dynamic computation graphs (DAG).
- [x] **Metal Backend (GPU):**
    - [x] Shared memory management & zero-copy buffer transfer (`MTLBuffer`).
    - [x] Asynchronous command dispatch (`MTLCommandQueue`).
    - [x] Kernel fusion engine for element-wise operations (Add + ReLU).

### 🚀 Neural Network Primitives (GPU Accelerated)
- [ ] **Layers (`nn.Module`):**
    - [x] `nn.Linear` (Dense Matrix Multiplication).
    - [ ] `nn.Conv2d` (Implementing Winograd/GEMM-based convolution in MSL).
    - [ ] `nn.BatchNorm2d` & `nn.LayerNorm` (Parallelized reduction kernels).
    - [ ] `nn.MaxPool2d` & `nn.AvgPool2d`.
    - [ ] `nn.Dropout` (Parallelized random number generation).
- [ ] **Optimizers:**
    - [x] SGD (Stochastic Gradient Descent).
    - [ ] **Fused AdamW** (Kernel-level implementation for weight decay + updates).
    - [ ] RMSProp.

### 🛠️ Utilities & Ecosystem
- [ ] **DataLoaders:** Multi-threaded pre-fetching and batching pipeline.
- [ ] **Serialization:** Saving and loading model state dicts (binary format).
- [ ] **ONNX Export:** Utility to export computational graphs to ONNX format for inference.
