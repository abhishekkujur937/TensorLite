#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

struct Expr {
    std::string name;                          // variable name, or op name if it is a call
    std::vector<std::unique_ptr<Expr>> args;   // empty → plain variable reference
    int line;
    int col;
};

enum class StmtKind { Decl, Assign, Output };

struct Stmt {
    StmtKind kind;
    std::string decl_kind;        // "input" | "weight" | "const"   (Decl only)
    std::string name;
    std::vector<int> dims;        // Decl only
    float const_value = 0.0f;     // const Decl only
    std::unique_ptr<Expr> expr;   // Assign only
    int line;
    int col;
};

struct Program {
    std::vector<Stmt> stmts;      // keeps source order
};

// Prints the AST as an indented tree (--dump-ast). Defined in ast_dump.cpp.
void dump_ast(const Program& program, std::ostream& os);
