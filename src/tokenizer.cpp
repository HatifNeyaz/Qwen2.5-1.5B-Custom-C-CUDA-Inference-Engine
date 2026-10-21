#include "tokenizer.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

using json = nlohmann::json;

Tokenizer::Tokenizer() {
    init_byte_mappings();
}

void Tokenizer::init_byte_mappings() {
    // Exact GPT-2 / Qwen byte-to-unicode mapping
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            std::string s(1, static_cast<char>(b));
            byte_to_unicode_[static_cast<uint8_t>(b)] = s;
            unicode_to_byte_[s] = static_cast<uint8_t>(b);
        } else {
            int code = 256 + n;
            std::string s;
            s += static_cast<char>(0xC0 | ((code >> 6) & 0x1F));
            s += static_cast<char>(0x80 | (code & 0x3F));
            byte_to_unicode_[static_cast<uint8_t>(b)] = s;
            unicode_to_byte_[s] = static_cast<uint8_t>(b);
            n++;
        }
    }
}

bool Tokenizer::load(const std::string& tokenizer_json_path) {
    std::cout << "[+] Loading tokenizer from: " << tokenizer_json_path << "..." << std::endl;
    std::ifstream f(tokenizer_json_path);
    if (!f.is_open()) {
        std::cerr << "[-] Error opening " << tokenizer_json_path << std::endl;
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cerr << "[-] JSON parse error: " << e.what() << std::endl;
        return false;
    }

    // 1. Load Vocab
    if (j.contains("model") && j["model"].contains("vocab")) {
        const auto& vocab_obj = j["model"]["vocab"];
        for (auto it = vocab_obj.begin(); it != vocab_obj.end(); ++it) {
            std::string token = it.key();
            int id = it.value().get<int>();
            vocab_[token] = id;
            id_to_token_[id] = token;
        }
    }

    // 2. Load Merges
    if (j.contains("model") && j["model"].contains("merges")) {
        int rank = 0;
        for (const auto& item : j["model"]["merges"]) {
            if (item.is_string()) {
                merges_rank_[item.get<std::string>()] = rank++;
            } else if (item.is_array() && item.size() == 2) {
                std::string pair_str = item[0].get<std::string>() + " " + item[1].get<std::string>();
                merges_rank_[pair_str] = rank++;
            }
        }
    }

    // 3. Load Added / Special Tokens
    if (j.contains("added_tokens")) {
        for (const auto& item : j["added_tokens"]) {
            int id = item["id"].get<int>();
            std::string content = item["content"].get<std::string>();
            vocab_[content] = id;
            id_to_token_[id] = content;
            special_tokens_[content] = id;

            if (content == "<|endoftext|>") eos_id_ = id;
            if (content == "<|im_start|>")  im_start_id_ = id;
            if (content == "<|im_end|>")    im_end_id_ = id;
        }
    }

    std::cout << "[✓] Tokenizer loaded: " << vocab_.size() << " vocabulary tokens, "
              << merges_rank_.size() << " merge rules.\n";
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    size_t pos = 0;

    while (pos < text.size()) {
        // Check for direct special token matches
        bool matched_special = false;
        for (const auto& sp : special_tokens_) {
            const std::string& token_str = sp.first;
            if (pos + token_str.size() <= text.size() &&
                text.compare(pos, token_str.size(), token_str) == 0) {
                tokens.push_back(sp.second);
                pos += token_str.size();
                matched_special = true;
                break;
            }
        }
        if (matched_special) continue;

        // Find boundary of next special token
        size_t next_special_pos = std::string::npos;
        for (const auto& sp : special_tokens_) {
            size_t p = text.find(sp.first, pos);
            if (p != std::string::npos && (next_special_pos == std::string::npos || p < next_special_pos)) {
                next_special_pos = p;
            }
        }

        std::string chunk = (next_special_pos == std::string::npos) 
                            ? text.substr(pos) 
                            : text.substr(pos, next_special_pos - pos);
        pos += chunk.size();

        if (chunk.empty()) continue;

        // Byte-level conversion
        std::vector<std::string> words;
        words.reserve(chunk.size());
        for (unsigned char c : chunk) {
            auto it = byte_to_unicode_.find(c);
            if (it != byte_to_unicode_.end()) {
                words.push_back(it->second);
            } else {
                words.push_back(std::string(1, static_cast<char>(c)));
            }
        }

        // Iterative BPE merges
        while (words.size() > 1) {
            int best_rank = 1e9;
            int best_idx = -1;

            for (size_t i = 0; i < words.size() - 1; ++i) {
                std::string pair_str = words[i] + " " + words[i + 1];
                auto it = merges_rank_.find(pair_str);
                if (it != merges_rank_.end()) {
                    if (it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = static_cast<int>(i);
                    }
                }
            }
            if (best_idx == -1) break;

            std::vector<std::string> new_words;
            new_words.reserve(words.size() - 1);
            for (size_t i = 0; i < words.size(); ++i) {
                if (i == static_cast<size_t>(best_idx)) {
                    new_words.push_back(words[i] + words[i + 1]);
                    i++;
                } else {
                    new_words.push_back(words[i]);
                }
            }
            words = std::move(new_words);
        }

        // Map merged tokens to IDs
        for (const auto& w : words) {
            auto it = vocab_.find(w);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            }
        }
    }
    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it == id_to_token_.end()) return "";

    const std::string& token_str = it->second;

    // Suppress internal control tokens from console output
    if (special_tokens_.count(token_str)) {
        return "";
    }

    std::string raw_bytes = "";
    for (size_t i = 0; i < token_str.size();) {
        bool matched = false;
        for (size_t len = 2; len >= 1; --len) {
            if (i + len <= token_str.size()) {
                std::string sub = token_str.substr(i, len);
                auto u_it = unicode_to_byte_.find(sub);
                if (u_it != unicode_to_byte_.end()) {
                    raw_bytes.push_back(static_cast<char>(u_it->second));
                    i += len;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            raw_bytes.push_back(token_str[i]);
            i++;
        }
    }
    return raw_bytes;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int tid : tokens) {
        result += decode(tid);
    }
    return result;
}