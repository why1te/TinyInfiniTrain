#include "infini_train/include/autograd/function.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
namespace {
class AccumulateGrad final : public Function {
public:
  explicit AccumulateGrad(std::shared_ptr<Tensor> grad) : grad_(grad) {}

  std::vector<std::shared_ptr<Tensor>>
  Forward(const std::vector<std::shared_ptr<Tensor>> &) override {
    LOG(FATAL) << "AccumulateGrad::Forward shall not be called directly!";
    return {};
  }

  std::vector<std::shared_ptr<Tensor>>
  Backward(const std::vector<std::shared_ptr<Tensor>> &) override {
    LOG(FATAL) << "AccumulateGrad::Backward shall not be called directly!";
    return {};
  }

  void BackwardPartial(const std::shared_ptr<Tensor> &grad_output,
                       int) override {
    if (grad_output) {
      auto device = grad_->GetDevice().Type();
      auto kernel =
          Dispatcher::Instance().GetKernel({device, "AccumulateGrad"});
      kernel.Call<void>(grad_output, 1.0f, grad_);
    }
  }

private:
  std::shared_ptr<Tensor> grad_ = nullptr;
};
} // namespace
std::vector<std::shared_ptr<Tensor>>
Function::Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  // * 范式：输入 Tensors → 当前 Function → 输出 Tensors
  // 比如 y = x ^ 2, 计算当前节点的op为 2 ^ x
  // * 执行前向计算
  auto output_tensors = Forward(input_tensors);
  // * 保存反向传播需要的信息
  SetupContext(input_tensors, output_tensors);

  bool output_requires_grad = false;
  // * 连接计算图：遍历当前算子的每个输入，记录梯度下一步传播的位置
  for (int idx = 0; idx < input_tensors.size(); ++idx) {
    const auto &input_tensor = input_tensors[idx];
    // 处理叶子节点
    if (input_tensor->requires_grad() && input_tensor->is_leaf()) {
      // emplace_back(反向传播的下一个算子，梯度对应第几个输出)
      next_functions_.emplace_back(
          std::make_shared<AccumulateGrad>(input_tensor->grad()), 0);
    } else {
      next_functions_.emplace_back(input_tensor->grad_fn(),
                                   input_tensor->output_idx());
      // * 增加依赖数量
      if (input_tensor->grad_fn()) {
        input_tensor->grad_fn()->IncreaseDependenciesNumber();
      }
    }
    output_requires_grad |= input_tensor->requires_grad();
  }

  grad_outputs_reached_ = 0;
  grad_outputs_.resize(output_tensors.size(), nullptr);
  for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
    auto &output_tensor = output_tensors[output_idx];
    output_tensor->set_requires_grad(output_requires_grad);
    output_tensor->set_is_leaf(false);
    output_tensor->set_grad_fn(output_requires_grad ? shared_from_this()
                                                    : nullptr);
    // * 标记 input_tensor 是当前算子的第几个输出
    output_tensor->set_output_idx(output_idx);
  }

  // 返回前向计算结果
  return output_tensors;
}

void Function::BackwardPartial(const std::shared_ptr<Tensor> &grad_output,
                               int grad_output_idx) {
  // 保存第一次反向传播到达的梯度
  if (!grad_outputs_[grad_output_idx]) {
    grad_outputs_[grad_output_idx] = grad_output;
    ++grad_outputs_reached_;
  } else {
    // 累加梯度：AcccumulatedGrad 重写了 BackwardPartial()
    // 累加梯度保存在 grad_output
    auto accumulate_function =
        std::make_shared<AccumulateGrad>(grad_outputs_[grad_output_idx]);
    // 参数 0 没有实际作用
    accumulate_function->BackwardPartial(grad_output, 0);
  }

  ++dependencies_reached_;

  // 当前算子所需的输出梯度已全部到达
  if (grad_outputs_reached_ == grad_outputs_.size() &&
      (dependencies_reached_ == dependencies_number_ ||
       dependencies_number_ == 0)) {
    // 分别计算对每个输入的梯度，并将梯度回传给对应的上游算子
    // 计算输入梯度，由算子单独实现
    auto grad_inputs = Backward(grad_outputs_);
    saved_tensors_.clear();
    grad_outputs_.clear();
    CHECK_EQ(grad_inputs.size(), next_functions_.size());
    // 反向传播
    for (int idx = 0; idx < grad_inputs.size(); ++idx) {
      auto &grad_input = grad_inputs[idx];
      auto &[next_function, output_idx] = next_functions_[idx];
      if (grad_input && next_function) {
        // 表明调用一次当前函数即可启动对整个计算图的反向传播
        next_function->BackwardPartial(grad_input, output_idx);
      }
    }
  }
}
void Function::IncreaseDependenciesNumber() { ++dependencies_number_; }
} // namespace infini_train::autograd
