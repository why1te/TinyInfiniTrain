#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
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

unsigned int RandomU32(uint64_t &state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
  return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from RandomF32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}
// Tokenizer 的构造函数
Tokenizer::Tokenizer(const std::string &filepath) {
  /* =====================================作业=====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                                      |  VOCAB TABLE|
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) |token词表数据|
    ----------------------------------------------------------------------------------
    =====================================作业=====================================
  */
  std::ifstream ifs(filepath, std::ios::binary);
  CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;
  constexpr size_t header_size = 1024;
  auto header = ReadSeveralBytesFromIfstream(header_size, &ifs);
  CHECK(ifs.good()) << "Failed to read tokenizer header: " << filepath;

  magic_number_ = BytesToType<uint32_t>(header, 0);
  const auto version = BytesToType<uint32_t>(header, 4);
  vocab_size_ = BytesToType<uint32_t>(header, 8);

  CHECK(version == static_cast<uint32_t>(Version::kV1) ||
        version == static_cast<uint32_t>(Version::kV2))
      << "Unsupported tokenizer version: " << version;
  CHECK_GT(vocab_size_, 0) << "Tokenizer vocabulary must not be empty";

  const auto eot_it = kEotMap.find(magic_number_);
  CHECK(eot_it != kEotMap.end())
      << "Unsupported tokenizer magic: " << magic_number_;
  eot_token_ = eot_it->second;

  token_table_.reserve(vocab_size_);

  // * NOTE：词表区采用“长度 + token 原始字节”的变长格式
  // token i: | length(1 byte) | token 内容(length bytes) |
  for (uint32_t token_id = 0; token_id < vocab_size_; token_id++) {
    uint8_t token_len = 0;
    // std::ifstream::read() 的第一个参数类型固定为 char * 类型，也就是指针类型
    ifs.read(reinterpret_cast<char *>(&token_len), sizeof(token_len));
    CHECK_EQ(static_cast<size_t>(ifs.gcount()), sizeof(token_len))
        << "Failed to read token length: " << token_id;
    CHECK_GT(token_len, 0) << "Token length must be greater than zero";

    // '\0' 是填充
    std::string token(token_len, '\0');
    ifs.read(token.data(), token_len);
    CHECK_EQ(static_cast<size_t>(ifs.gcount()), static_cast<size_t>(token_len))
        << "Failed to read token: " << token_id;

    token_table_.push_back(std::move(token));
  }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
  /* =====================================作业=====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    =====================================作业=====================================
  */
  CHECK_LT(token_id, token_table_.size()) << "Invalid token id: " << token_id;
  return token_table_[token_id];
}
/**
 * @brief 使用模型进行自回归文本生成
 *
 * @param model 用于生成文本的模型
 * @param batch_size 一次输入的序列条数
 * @param sequence_length 模型输入序列的长度（等于模型训练样本的长度）
 * @param text_length 模型生成的文本长度
 * @param device 模型推理所在的设备
 *
 * @return void 不存储模型生成内容，直接打印输出
 */
void Tokenizer::GenerateText(infini_train::nn::Module &model,
                             uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
  // assign()适合： 1. vector 已经存在；2. 需要整体替换内容
  std::vector<int64_t> dims = {static_cast<int64_t>(batch_size),
                               static_cast<int64_t>(sequence_length)};
  CHECK_GT(sequence_length, 0);
  CHECK_GT(text_length, 0);
  CHECK_LE(text_length, sequence_length)
      << "Text length exceeds sequence length";
  // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(2, 64)
  infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
  int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
  for (int i = 0; i < batch_size * sequence_length; ++i) {
    x_buff[i] = eot_token_;
  }

  // Give some contexts: "The meaning of life is "
  // ! WARNING: 实际上 bs = 1
  const auto prompt_it = kPromptMap.find(magic_number_);
  CHECK(prompt_it != kPromptMap.end())
      << "Unsupported tokenizer magic: " << magic_number_;
  const auto &prompt = prompt_it->second;
  const auto prompt_len = prompt.size();
  CHECK_LE(prompt_len, sequence_length)
      << "Prompt length exceeds sequence length";
  for (size_t i = 0; i < prompt_len; ++i) {
    x_buff[i] = prompt[i];
  }
  std::cout << "The meaning of life is";

  auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
  uint64_t rng_state = kRngState;
  LOG(INFO) << "start generate text:";
  for (int t = prompt_len; t < text_length; t++) {
    /* =====================================作业=====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
       =====================================作业=====================================
    */
    // * GPU 执行
    auto logits = model.Forward({x})[0];
    auto probabilities =
        nn::function::Softmax(logits, -1)->To(Device(DeviceType::kCPU, 0));
    // * CPU 执行
    float *probabilities_data_ptr =
        static_cast<float *>(probabilities.DataPtr());
    // 采样计算 token t，需要位置 t-1 的 logits
    size_t offset = static_cast<size_t>(t - 1) * vocab_size_;
    float *next_token_probabilities = probabilities_data_ptr + offset;
    float coin = RandomF32(rng_state);
    int next_token_id = SampleMult(next_token_probabilities, vocab_size_, coin);
    x_buff[t] = next_token_id;
    std::cout << Decode(next_token_id);
    // * GPU 执行
    x = std::make_shared<Tensor>(x_tensor.To(device));
  }
  std::cout << std::endl;
}
} // namespace infini_train
