#include "lexer.h"

#include <cctype>

#include "error.h"

// std::isalpha / std::isdigit need an unsigned char value, so we cast first.
static bool is_letter(char c)
{
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

static bool is_digit(char c)
{
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

Lexer::Lexer(const std::string& source, const std::string& filename)
    : source_(source), filename_(filename)
{
}

bool Lexer::at_end() const
{
    return pos_ >= source_.size();
}

// Current character, without consuming it. '\0' at end of file.
char Lexer::peek() const
{
    if (at_end()) {
        return '\0';
    }
    return source_[pos_];
}

// Character after the current one, without consuming anything.
char Lexer::peek_next() const
{
    if (pos_ + 1 >= source_.size()) {
        return '\0';
    }
    return source_[pos_ + 1];
}

// Consumes one character and updates line/col.
char Lexer::advance()
{
    char c = source_[pos_];
    pos_++;
    if (c == '\n') {
        line_++;
        col_ = 1;
    } else {
        col_++;
    }
    return c;
}

// Skips spaces, tabs, \r, \n and # comments.
void Lexer::skip_whitespace_and_comments()
{
    while (!at_end()) {
        char c = peek();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
        } else if (c == '#') {
            while (!at_end() && peek() != '\n') {
                advance();
            }
        } else {
            return;
        }
    }
}

// Reads [A-Za-z_][A-Za-z0-9_]* and checks if it is a keyword.
Token Lexer::lex_identifier()
{
    int start_line = line_;
    int start_col = col_;
    std::string text;

    while (is_letter(peek()) || is_digit(peek())) {
        text += advance();
    }

    TokenKind kind = TokenKind::Ident;
    if (text == "input") {
        kind = TokenKind::KwInput;
    } else if (text == "weight") {
        kind = TokenKind::KwWeight;
    } else if (text == "const") {
        kind = TokenKind::KwConst;
    } else if (text == "output") {
        kind = TokenKind::KwOutput;
    }

    return Token{kind, text, start_line, start_col};
}

// Reads -?[0-9]+ or -?[0-9]+.[0-9]+
Token Lexer::lex_number()
{
    int start_line = line_;
    int start_col = col_;
    std::string text;

    if (peek() == '-') {
        text += advance();
    }

    while (is_digit(peek())) {
        text += advance();
    }

    // Only take the '.' if a digit follows it, so "12." is not a number.
    if (peek() == '.' && is_digit(peek_next())) {
        text += advance();
        while (is_digit(peek())) {
            text += advance();
        }
    }

    return Token{TokenKind::Number, text, start_line, start_col};
}

std::vector<Token> Lexer::tokenize()
{
    std::vector<Token> tokens;

    while (true) {
        skip_whitespace_and_comments();

        if (at_end()) {
            tokens.push_back(Token{TokenKind::End, "", line_, col_});
            break;
        }

        char c = peek();

        if (is_letter(c)) {
            tokens.push_back(lex_identifier());
            continue;
        }

        if (is_digit(c) || (c == '-' && is_digit(peek_next()))) {
            tokens.push_back(lex_number());
            continue;
        }

        // Single-character symbols.
        TokenKind kind;
        if (c == '(') {
            kind = TokenKind::LParen;
        } else if (c == ')') {
            kind = TokenKind::RParen;
        } else if (c == '[') {
            kind = TokenKind::LBracket;
        } else if (c == ']') {
            kind = TokenKind::RBracket;
        } else if (c == ',') {
            kind = TokenKind::Comma;
        } else if (c == '=') {
            kind = TokenKind::Equals;
        } else {
            report_error(filename_, line_, col_, std::string("unexpected character '") + c + "'");
        }

        int start_line = line_;
        int start_col = col_;
        advance();
        tokens.push_back(Token{kind, std::string(1, c), start_line, start_col});
    }

    return tokens;
}
