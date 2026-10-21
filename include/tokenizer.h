#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Tokenizer {
public:
    Tokenizer();
    bool load(const std::string& tokenizer_json_path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& tokens) const;

    int eos_token_id() const { return eos_id_; }
    int im_start_id() const { return im_start_id_; }
    int im_end_id() const { return im_end_id_; }
    size_t vocab_size() const { return vocab_.size(); }

private:
    void init_byte_mappings();

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> merges_rank_;
    std::unordered_map<std::string, int> special_tokens_;

    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    int eos_id_ = 151643;      // <|endoftext|>
    int im_start_id_ = 151644;  // <|im_start|>
    int im_end_id_ = 151645;    // <|im_end|>
};