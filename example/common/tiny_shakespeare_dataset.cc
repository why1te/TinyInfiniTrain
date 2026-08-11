#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes,
                                                  std::ifstream *ifs) {
  std::vector<uint8_t> result(num_bytes);
  ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
  return result;
}

template <typename T>
T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
  static_assert(std::is_trivially_copyable<T>::value,
                "T must be trivially copyable.");
  T value;
  std::memcpy(&value, &bytes[offset], sizeof(T));
  return value;
}
/**
 * @brief 解析二进制数据集文件
 *
 * @param path 文件路径
 * @param sequence_length 一条训练样本的长度
 *
 * @return TinyShakespeareFile 文件结构体
 */
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path,
                                            size_t sequence_length) {
  /*
    ===================================作业===================================
    TODO：实现二进制数据集文件解析
    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                                     | DATA (tokens) |
    | magic(4B)| version(4B) | num_toks(4B) | reserved(1012B) |   token数据   |
    ----------------------------------------------------------------------------------
    ===================================作业===================================
  */
  std::ifstream ifs(path, std::ios::binary);
  CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;
  CHECK_GT(sequence_length, 0);
  constexpr size_t header_size = 1024;
  auto header = ReadSeveralBytesFromIfstream(header_size, &ifs);
  CHECK(ifs.good()) << "Failed to read dataset header: " << path;
  const int32_t magic = BytesToType<int32_t>(header, 0);
  const int32_t num_toks = BytesToType<int32_t>(header, 8);

  const auto type_it = kTypeMap.find(magic);
  CHECK(type_it != kTypeMap.end()) << "Unsupported dataset magic: " << magic;
  CHECK_GT(num_toks, 0) << "Dataset contains no tokens";
  const TinyShakespeareType type = type_it->second;

  // ! WARNING: 舍弃末尾 token
  const size_t num_seqs = static_cast<size_t>(num_toks) / sequence_length;
  std::vector<int64_t> dims = {static_cast<int64_t>(num_seqs),
                               static_cast<int64_t>(sequence_length)};
  TinyShakespeareFile result;
  result.type = type;
  result.dims = std::move(dims);
  result.tensor = infini_train::Tensor(result.dims, DataType::kINT64);
  const size_t num_elements = num_seqs * sequence_length;
  std::variant<std::vector<uint16_t>, std::vector<uint32_t>> buffer;
  switch (type) {
  case TinyShakespeareType::kUINT16:
    buffer = std::vector<uint16_t>(num_elements);
    break;
  case TinyShakespeareType::kUINT32:
    buffer = std::vector<uint32_t>(num_elements);
    break;
  default:
    LOG(FATAL) << "Unsupported dataset token type";
  }

  int64_t *dst = static_cast<int64_t *>(result.tensor.DataPtr());
  std::visit(
      // lamba 表达式： [捕获列表](参数列表) -> 返回类型 { 函数体 }
      [&](auto &tokens) {
        const size_t data_size_in_bytes = tokens.size() * sizeof(tokens[0]);
        ifs.read(reinterpret_cast<char *>(tokens.data()), data_size_in_bytes);
        CHECK_EQ(static_cast<size_t>(ifs.gcount()), data_size_in_bytes)
            << "dataset token data is truncated: " << path;

        for (size_t i = 0; i < tokens.size(); ++i) {
          dst[i] = static_cast<int64_t>(tokens[i]);
        }
      },
      buffer);
  return result;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath,
                                               size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(static_cast<size_t>(text_file_.dims[0]) - 1) {
  // ===================================作业===================================
  // TODO：初始化数据集实例
  // HINT: 调用ReadTinyShakespeareFile加载数据文件
  // ===================================作业===================================
}

std::pair<std::shared_ptr<infini_train::Tensor>,
          std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
  CHECK_LT(idx, text_file_.dims[0] - 1);
  std::vector<int64_t> dims =
      std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
  // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
  return {std::make_shared<infini_train::Tensor>(
              text_file_.tensor, idx * sequence_size_in_bytes_, dims),
          std::make_shared<infini_train::Tensor>(
              text_file_.tensor,
              idx * sequence_size_in_bytes_ + sizeof(int64_t), dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
