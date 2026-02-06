#include "utils.h"
#include <algorithm>
#include <sstream>

namespace utils {

const int DEFAULT_HASH_SEED = 42;
constexpr int HASH_MULTIPLIER = 31;

std::string StringHelper::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

std::vector<std::string> StringHelper::split(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        result.push_back(trim(token));
    }
    
    return result;
}

int calculate_hash(const std::string& input) {
    int hash = DEFAULT_HASH_SEED;
    for (char c : input) {
        hash = hash * HASH_MULTIPLIER + static_cast<int>(c);
    }
    return hash;
}

template<typename T>
T max_value(T a, T b) {
    return (a > b) ? a : b;
}

// Explicit instantiations
template int max_value<int>(int, int);
template double max_value<double>(double, double);

} // namespace utils
