#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "error.h"
#include "ir.h"

// Maps a name from the .tl file to the node that computes it.
using SymbolTable = std::unordered_map<std::string, Node*>;

// Turns one expression into nodes and returns the node for its result.
// result_name: the user's name for the top node of an assignment,
// or "" for inner calls (they get "t<id>").
static Node* lower_expr(Graph& graph, const SymbolTable& symbols, const std::string& filename,
                        const Expr& expr, const std::string& result_name)
{
    // Plain variable: no new node, just look up what it refers to.
    if (expr.args.empty()) {
        auto found = symbols.find(expr.name);
        if (found == symbols.end()) {
            report_error(filename, expr.line, expr.col, "undefined name '" + expr.name + "'");
        }
        return found->second;
    }

    // Call: check the op name and argument count first.
    OpKind op;
    std::size_t expected_args;
    if (expr.name == "matmul") {
        op = OpKind::MatMul;
        expected_args = 2;
    } else if (expr.name == "add") {
        op = OpKind::Add;
        expected_args = 2;
    } else if (expr.name == "mul") {
        op = OpKind::Mul;
        expected_args = 2;
    } else if (expr.name == "relu") {
        op = OpKind::Relu;
        expected_args = 1;
    } else {
        report_error(filename, expr.line, expr.col, "unknown op '" + expr.name + "'");
    }

    if (expr.args.size() != expected_args) {
        std::string plural = "";
        if (expected_args != 1) {
            plural = "s";
        }
        report_error(filename, expr.line, expr.col,
                     expr.name + " expects " + std::to_string(expected_args) + " argument" + plural
                     + " but got " + std::to_string(expr.args.size()));
    }

    // Lower the arguments first (recursion), then create this node.
    std::vector<Node*> inputs;
    for (const std::unique_ptr<Expr>& arg : expr.args) {
        inputs.push_back(lower_expr(graph, symbols, filename, *arg, ""));
    }

    Node* node = graph.add_node(op, inputs, result_name);
    node->line = expr.line;
    node->col = expr.col;
    return node;
}

Graph lower(const Program& program, const std::string& filename)
{
    Graph graph;
    SymbolTable symbols;

    for (const Stmt& stmt : program.stmts) {
        if (stmt.kind == StmtKind::Decl) {
            if (symbols.count(stmt.name) > 0) {
                report_error(filename, stmt.line, stmt.col, "redefinition of '" + stmt.name + "'");
            }

            OpKind op = OpKind::Input;
            if (stmt.decl_kind == "weight") {
                op = OpKind::Weight;
            } else if (stmt.decl_kind == "const") {
                op = OpKind::Const;
            }

            Node* node = graph.add_node(op, {}, stmt.name);
            node->shape = stmt.dims;   // declared shape; checked in S4
            node->const_value = stmt.const_value;
            node->line = stmt.line;
            node->col = stmt.col;
            symbols[stmt.name] = node;
        } else if (stmt.kind == StmtKind::Assign) {
            if (symbols.count(stmt.name) > 0) {
                report_error(filename, stmt.line, stmt.col, "redefinition of '" + stmt.name + "'");
            }
            Node* node = lower_expr(graph, symbols, filename, *stmt.expr, stmt.name);
            symbols[stmt.name] = node;
        } else {
            auto found = symbols.find(stmt.name);
            if (found == symbols.end()) {
                report_error(filename, stmt.line, stmt.col, "undefined name '" + stmt.name + "'");
            }
            Node* node = found->second;
            if (!node->is_output) {
                node->is_output = true;
                graph.outputs.push_back(node);
            }
        }
    }

    graph.rebuild_users();
    return graph;
}
