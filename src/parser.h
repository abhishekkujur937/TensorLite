#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ast.h"
#include "token.h"

// Recursive descent parser: one function per grammar rule (CLAUDE.md Section 6).
class Parser {
public:
    Parser(const std::vector<Token>& tokens, const std::string& filename);

    // program := stmt* EOF
    Program parse_program();

private:
    const Token& peek() const;
    const Token& advance();
    const Token& expect(TokenKind kind, const std::string& what);

    Stmt parse_stmt();
    Stmt parse_decl();
    Stmt parse_assign();
    Stmt parse_output();
    std::unique_ptr<Expr> parse_expr();
    std::vector<int> parse_dims();

    std::vector<Token> tokens_;
    std::string filename_;
    std::size_t pos_ = 0;    // index of the current token
};
