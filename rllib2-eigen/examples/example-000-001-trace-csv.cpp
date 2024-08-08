#include "trace-utils.hpp"
#include <fstream>
#include <array>
#include <string>

int main() {
    std::ofstream file {"test.csv"};
    utils::trace::csv trace{file};

    std::array<std::string, 4> header {"a", "b", "c", "d"};
    trace += header;
    std::array<double, 4> line {1, 2, 3, 4};
    trace += line;

    return 0;
}
