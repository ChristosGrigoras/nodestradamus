#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

#define MAX_BUFFER_SIZE 1024
#define MIN_VALUE 0

namespace utils {

class StringHelper {
public:
    static std::string trim(const std::string& str);
    static std::vector<std::string> split(const std::string& str, char delimiter);
};

int calculate_hash(const std::string& input);

template<typename T>
T max_value(T a, T b);

} // namespace utils

#endif // UTILS_H
