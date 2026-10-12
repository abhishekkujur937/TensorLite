#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

// Prints "error: <file>:<line>:<col>: <message>" to stderr and exits with code 1.
// [[noreturn]] tells the compiler this function never comes back.
[[noreturn]] inline void report_error(const std::string& file, int line, int col, const std::string& message)
{
    std::cerr << "error: " << file << ":" << line << ":" << col << ": " << message << "\n";
    std::exit(1);
}
