#include "parser.h"

#include <string>
#include <utility>

#include "error.h"

// Text used in error messages for the token we found, e.g. "','" or "end of file".
static std::string describe(const Token& tok)
{
    if (tok.kind == TokenKind::End) {
        return "end of file";
    }
    return "'" + tok.text + "'";
}

Parser::Parser(const std::vector<Token>& tokens, const std::string& filename)
    : tokens_(tokens), filename_(filename)
{
}

// Current token, without consuming it.
const Token& Parser::peek() const
{
    return tokens_[pos_];
}

// Consumes the current token and returns it. Never moves past End.
const Token& Parser::advance()
{
    const Token& tok = tokens_[pos_];
    if (tok.kind != TokenKind::End) {
        pos_++;
    }
    return tok;
}

// Consumes the current token if it has the given kind, otherwise reports
// "expected <what> but found <token>".
const Token& Parser::expect(TokenKind kind, const std::string& what)
{
    const Token& tok = peek();
    if (tok.kind != kind) {
        report_error(filename_, tok.line, tok.col, "expected " + what + " but found " + describe(tok));
    }
    return advance();
}

// program := stmt* EOF
Program Parser::parse_program()
{
    Program program;
    while (peek().kind != TokenKind::End) {
        program.stmts.push_back(parse_stmt());
    }
    return program;
}

// stmt := decl | assign | output
// The first token decides which rule to use.
Stmt Parser::parse_stmt()
{
    TokenKind kind = peek().kind;

    if (kind == TokenKind::KwInput || kind == TokenKind::KwWeight || kind == TokenKind::KwConst) {
        return parse_decl();
    }
    if (kind == TokenKind::KwOutput) {
        return parse_output();
    }
    if (kind == TokenKind::Ident) {
        return parse_assign();
    }

    const Token& tok = peek();
    report_error(filename_, tok.line, tok.col, "expected a statement but found " + describe(tok));
}

// decl := ('input' | 'weight') IDENT '[' dims ']'
//       | 'const' IDENT '[' dims ']' '=' NUMBER
Stmt Parser::parse_decl()
{
    const Token& keyword = advance();

    Stmt stmt;
    stmt.kind = StmtKind::Decl;
    stmt.decl_kind = keyword.text;
    stmt.line = keyword.line;
    stmt.col = keyword.col;

    stmt.name = expect(TokenKind::Ident, "a name").text;
    expect(TokenKind::LBracket, "'['");
    stmt.dims = parse_dims();
    expect(TokenKind::RBracket, "']'");

    if (keyword.kind == TokenKind::KwConst) {
        expect(TokenKind::Equals, "'='");
        const Token& value = expect(TokenKind::Number, "a number");
        stmt.const_value = std::stof(value.text);
    }

    return stmt;
}

// dims := NUMBER (',' NUMBER)?
std::vector<int> Parser::parse_dims()
{
    std::vector<int> dims;

    while (true) {
        const Token& tok = expect(TokenKind::Number, "a dimension");

        // A dimension must be a whole number. More than 9 digits would not fit in an int.
        if (tok.text.find('.') != std::string::npos || tok.text.size() > 9) {
            report_error(filename_, tok.line, tok.col, "expected an integer dimension but found " + describe(tok));
        }
        dims.push_back(std::stoi(tok.text));

        // At most 2 dimensions: stop after the second one, or if no ',' follows.
        if (dims.size() == 2 || peek().kind != TokenKind::Comma) {
            break;
        }
        advance();   // the ','
    }

    return dims;
}

// assign := IDENT '=' expr
Stmt Parser::parse_assign()
{
    const Token& name = advance();

    Stmt stmt;
    stmt.kind = StmtKind::Assign;
    stmt.name = name.text;
    stmt.line = name.line;
    stmt.col = name.col;

    expect(TokenKind::Equals, "'='");
    stmt.expr = parse_expr();

    return stmt;
}

// expr := IDENT
//       | IDENT '(' expr (',' expr)* ')'
// Calls itself for every argument: this recursion is what builds the tree.
std::unique_ptr<Expr> Parser::parse_expr()
{
    const Token& name = expect(TokenKind::Ident, "a name");

    std::unique_ptr<Expr> expr = std::make_unique<Expr>();
    expr->name = name.text;
    expr->line = name.line;
    expr->col = name.col;

    if (peek().kind == TokenKind::LParen) {
        advance();   // the '('
        expr->args.push_back(parse_expr());
        while (peek().kind == TokenKind::Comma) {
            advance();   // the ','
            expr->args.push_back(parse_expr());
        }
        expect(TokenKind::RParen, "')'");
    }

    return expr;
}

// output := 'output' IDENT
Stmt Parser::parse_output()
{
    const Token& keyword = advance();

    Stmt stmt;
    stmt.kind = StmtKind::Output;
    stmt.line = keyword.line;
    stmt.col = keyword.col;
    stmt.name = expect(TokenKind::Ident, "a name").text;

    return stmt;
}
