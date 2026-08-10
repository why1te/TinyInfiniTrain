# TinyInfiniTrain QA

## CUDA Adam 编译成功，但运行测试时发生 Segmentation Fault

### 问题现象

`test_adam_cuda` 可以正常编译和链接，但运行
`AdamOptimizerTest.BasicParameterUpdateCuda` 时发生段错误：

```text
test_adam_cuda ... ***Exception: SegFault
```

### 问题原因

CUDA Tensor 的 `DataPtr()` 返回设备内存地址。该地址可以作为参数传给
CUDA kernel，但不能在普通 CPU 函数中通过 `for` 循环直接解引用。

下面的代码仍然运行在 CPU 上，因此访问 `m_data[i]`、`v_data[i]`、
`grad_data[i]` 或 `tensor_data[i]` 时可能直接触发段错误：

```cpp
for (int64_t i = 0; i < n; ++i) {
  m_data[i] = beta1 * m_data[i] + (1 - beta1) * grad_data[i];
  // ...
}
```

此外，调用

```cpp
AccumulateGrad(param, learning_rate, tensor);
```

执行的是 `tensor += learning_rate * param`，并不会更新 `param`。即使没有
发生段错误，这个临时 Tensor 随后也会被销毁，Adam 参数仍不会得到正确更新。

### 当前解决方法

最终采用一个独立的 `__global__` Adam kernel，不再创建临时 Tensor，也不再
复用只能执行 `tensor += rate * grad` 的 `AccumulateGradKernel`。

普通 C++ 入口函数 `AdamAccumulateGrad()` 保持原签名不变，只负责：

1. 从 `grad`、`param`、`m` 和 `v` 中取得设备指针。
2. 通过 `NumElements()` 取得需要更新的元素总数。
3. 按每个 block 256 个线程计算启动规模并启动 CUDA kernel。

每个 GPU 线程根据自己的 `idx` 更新一个元素：

```cpp
__global__ void AdamAccumulateGradKernel(
    const float* grad_data_ptr, float* param_data_ptr,
    float* m_data_ptr, float* v_data_ptr, float learning_rate,
    float beta1, float beta2, float eps, int64_t t,
    size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }

  const float g = grad_data_ptr[idx];
  m_data_ptr[idx] = beta1 * m_data_ptr[idx] + (1.0f - beta1) * g;
  v_data_ptr[idx] =
      beta2 * v_data_ptr[idx] + (1.0f - beta2) * g * g;

  const float m_hat =
      m_data_ptr[idx] / (1.0f - powf(beta1, t));
  const float v_hat =
      v_data_ptr[idx] / (1.0f - powf(beta2, t));

  param_data_ptr[idx] -=
      learning_rate * m_hat / (sqrtf(v_hat) + eps);
}
```

入口函数按框架内其他 CUDA 算子的方式启动 kernel：

```cpp
size_t num_elements = param->NumElements();
int threads_per_block = 256;
int num_blocks =
    (num_elements + threads_per_block - 1) / threads_per_block;

AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
    grad_data_ptr, param_data_ptr, m_data_ptr, v_data_ptr,
    learning_rate, beta1, beta2, eps, t, num_elements);
```

`num_blocks` 使用向上取整，保证启动的线程总数不少于 Tensor 的元素数；最后
一个 block 中多出来的线程由 `idx < num_elements` 边界判断过滤。

CPU 版本使用显式 `for` 循环；CUDA 版本由大量 GPU 线程并行完成相同的逐元素
更新。Tensor 即使是多维的，也可以通过 `NumElements()` 扁平遍历。

公式中的 `1 - beta1^t` 和 `1 - beta2^t` 是 Adam 对零初始化移动平均的偏差
修正，不是模型中新增加的 bias 参数。当前最小实现直接在 kernel 中计算它们；
后续如需优化，可以在 CPU 入口中各计算一次后传入 kernel，避免每个线程重复
执行 `powf`。

### 验证结果

当前实现已经重新构建并通过全部 Adam 测试：

```text
test_adam:      2/2 passed
test_adam_cuda: 2/2 passed
```

### 补充说明

`Cannot find file: DartConfiguration.tcl` 是 CTest dashboard 配置警告，不是本次
段错误的直接原因。真正的问题发生在 CUDA Adam 运行时访问设备内存的方式上。
