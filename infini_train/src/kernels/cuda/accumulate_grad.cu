#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

// 真实运行在 GPU 上的算子： tensor = tensor + lr * grad
__global__ void AccumulateGradKernel(const float* grad_ptr, float rate,
                                     float* tensor_ptr, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    tensor_ptr[idx] += rate * grad_ptr[idx];
  }
}
// 负责准备启动算子
void AccumulateGrad(const std::shared_ptr<Tensor>& gradient, float rate,
                    const std::shared_ptr<Tensor>& tensor) {
  size_t num_elements = gradient->NumElements();

  const float* grad_ptr = static_cast<const float*>(gradient->DataPtr());
  float* tensor_ptr = static_cast<float*>(tensor->DataPtr());

  int threads_per_block = 256;
  int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

  AccumulateGradKernel<<<num_blocks, threads_per_block>>>(
      grad_ptr, rate, tensor_ptr, num_elements);
}

// 一个 thread 处理d多个个元素
__global__ void AdamAccumulateGradKernel(const float* grad_data_ptr,
                                         float* param_data_ptr,
                                         float* m_data_ptr, float* v_data_ptr,
                                         float learning_rate, float beta1,
                                         float beta2, float eps, int64_t t,
                                         const size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    m_data_ptr[i] = beta1 * m_data_ptr[i] + (1 - beta1) * grad_data_ptr[i];
    v_data_ptr[i] = beta2 * v_data_ptr[i] +
                    (1 - beta2) * grad_data_ptr[i] * grad_data_ptr[i];
    float m_hat = m_data_ptr[i] / (1 - pow(beta1, t));
    float v_hat = v_data_ptr[i] / (1 - pow(beta2, t));
    param_data_ptr[i] =
        param_data_ptr[i] - learning_rate * m_hat / (sqrt(v_hat) + eps);
  }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor>& grad,
                        const std::shared_ptr<Tensor>& param,
                        const std::shared_ptr<Tensor>& m,
                        const std::shared_ptr<Tensor>& v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
  // ===================================作业===================================
  // TODO：实现Adam优化器的梯度累积和参数更新
  // REF:
  // ===================================作业===================================
  const float* grad_data_ptr = static_cast<const float*>(grad->DataPtr());
  float* param_data_ptr = static_cast<float*>(param->DataPtr());
  float* m_data_ptr = static_cast<float*>(m->DataPtr());
  float* v_data_ptr = static_cast<float*>(v->DataPtr());

  size_t num_elements = param->NumElements();
  // TODO：合理调整 block 和 thread 的数量
  int thread_per_block = 256;
  int num_blocks = (num_elements + thread_per_block - 1) / thread_per_block;
  AdamAccumulateGradKernel<<<num_blocks, thread_per_block>>>(
      grad_data_ptr, param_data_ptr, m_data_ptr, v_data_ptr, learning_rate,
      beta1, beta2, eps, t, num_elements);
}
}  // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)       \
  REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, \
                  infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
