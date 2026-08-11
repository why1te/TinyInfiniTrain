#pragma once

#include <cstdint>
#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
/// @brief 通用函数指针包装器
//      1. 把不同签名的 kernel 函数统一保存为 void *
//      2. 调用时再由调用者指定函数签名转换恢复
class KernelFunction {
public:
  // * 构造时：统一将函数指针存储为 void *
  template <typename FuncT>
  explicit KernelFunction(FuncT &&func)
      : func_ptr_(reinterpret_cast<void *>(func)) {}
  // * Call：根据提供的返回值和参数类型，恢复指针类型并执行
  template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    // =================================== 作业
    // ===================================
    // TODO：实现通用kernel调用接口
    // 功能描述：将存储的函数指针转换为指定类型并调用
    // =================================== 作业
    // ===================================

    using FuncT = RetT (*)(ArgsT...);
    auto func = reinterpret_cast<FuncT>(func_ptr_);
    return func(args...);
  }

private:
  void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
  using KeyT = std::pair<DeviceType, std::string>;

  // * 静态成员函数，可以通过类名调用
  static Dispatcher &Instance() {
    static Dispatcher instance;
    return instance;
  }

  const KernelFunction &GetKernel(KeyT key) const {
    CHECK(key_to_kernel_map_.contains(key))
        << "Kernel not found: " << key.second
        << " on device: " << static_cast<int>(key.first);
    return key_to_kernel_map_.at(key);
  }

  template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    // =================================== 作业
    // ===================================
    // TODO：实现kernel注册机制
    // 功能描述：将kernel函数与设备类型、名称绑定
    // =================================== 作业
    // ===================================
    CHECK(!key_to_kernel_map_.contains((key)))
        << "Kernel already registered: " << key.second << " on device "
        << static_cast<int>(key.first);
    key_to_kernel_map_.emplace(key,
                               KernelFunction(std::forward<FuncT>(kernel)));
  }

private:
  std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                      \
  static const bool kernel_name##_##registered = []() {                        \
    infini_train::Dispatcher::Instance().Register({device, #kernel_name},      \
                                                  kernel_func);                \
    return true;                                                               \
  }();                                                                         \
// =================================== 作业 ===================================
// TODO：实现自动注册宏
// 功能描述：在全局静态区注册kernel，避免显式初始化代码
// =================================== 作业 ===================================
