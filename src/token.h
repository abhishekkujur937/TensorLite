#pragma once

#include <string>

enum class TokenKind {
    Ident,
    Number,
    KwInput,
    KwWeight,
    KwConst,
    KwOutput,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Equals,
    End
};

struct Token {
    TokenKind kind;
    std::string text;
    int line;
    int col;
};

// Returns the name of a token kind, used by --dump-tokens.
inline const char* token_kind_name(TokenKind kind)
{
    switch (kind) {
        case TokenKind::Ident:    return "Ident";
        case TokenKind::Number:   return "Number";
        case TokenKind::KwInput:  return "KwInput";
        case TokenKind::KwWeight: return "KwWeight";
        case TokenKind::KwConst:  return "KwConst";
        case TokenKind::KwOutput: return "KwOutput";
        case TokenKind::LParen:   return "LParen";
        case TokenKind::RParen:   return "RParen";
        case TokenKind::LBracket: return "LBracket";
        case TokenKind::RBracket: return "RBracket";
        case TokenKind::Comma:    return "Comma";
        case TokenKind::Equals:   return "Equals";
        case TokenKind::End:      return "End";
    }
    return "?";
}
