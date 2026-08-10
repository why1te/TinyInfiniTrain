#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate,
                    const std::shared_ptr<Tensor> &tensor) {
  for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
    static_cast<float *>(tensor->DataPtr())[idx] +=
        rate * static_cast<const float *>(gradient->DataPtr())[idx];
  }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad,
                        const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m,
                        const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
  // ===================================作业===================================
  // TODO：实现Adam优化器的梯度累积和参数更新
  // REF:
  // ===================================作业===================================
  auto *grad_data = static_cast<const float *>(grad->DataPtr());
  auto *param_data = static_cast<float *>(param->DataPtr());
  auto *m_data = static_cast<float *>(m->DataPtr());
  auto *v_data = static_cast<float *>(v->DataPtr());
  int64_t n = param->NumElements();
  for (int64_t i = 0; i < n; ++i) {
    m_data[i] = beta1 * m_data[i] + (1 - beta1) * grad_data[i];
    v_data[i] = beta2 * v_data[i] + (1 - beta2) * grad_data[i] * grad_data[i];
    auto m_hat = m_data[i] / (1 - pow(beta1, t));
    auto v_hat = v_data[i] / (1 - pow(beta2, t));
    param_data[i] = param_data[i] - learning_rate * m_hat / (sqrt(v_hat) + eps);
  }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                       \
  REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name,                 \
                  infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
