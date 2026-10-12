#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "token.h"

// Turns the source text of a .tl file into a list of tokens.
class Lexer {
public:
    Lexer(const std::string& source, const std::string& filename);

    // Returns all tokens. The last token is always End.
    std::vector<Token> tokenize();

private:
    bool at_end() const;
    char peek() const;
    char peek_next() const;
    char advance();
    void skip_whitespace_and_comments();
    Token lex_identifier();
    Token lex_number();

    std::string source_;
    std::string filename_;
    std::size_t pos_ = 0;    // index of the next character to read
    int line_ = 1;           // line of the next character (starts at 1)
    int col_ = 1;            // column of the next character (starts at 1)
};
