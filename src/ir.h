#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "ast.h"

enum class OpKind {
    Input,
    Weight,
    Const,
    MatMul,
    Add,
    Mul,
    Relu,
    FusedMatMulAddRelu    // inputs: A, B, bias
};

struct Node {
    int id;
    OpKind op;
    std::string name;             // user name if given, else "t<id>"
    std::vector<int> shape;
    std::vector<Node*> inputs;    // non-owning
    std::vector<Node*> users;     // non-owning, rebuilt by Graph::rebuild_users()
    float const_value = 0.0f;     // Const only
    bool is_output = false;
    int line = 0;                 // for error messages
    int col = 0;
};

struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;   // owns all nodes
    std::vector<Node*> outputs;
    int next_id = 0;

    Node* add_node(OpKind op, std::vector<Node*> inputs, std::string name);
    void rebuild_users();
    std::vector<Node*> topo_order() const;      // Kahn's algorithm
    void replace_all_uses(Node* old_node, Node* new_node);
    void erase_unreachable();                   // used by DCE
    void print(std::ostream& os, const std::string& title) const;
};

// Name of an op as shown in --dump-ir, e.g. "matmul". Defined in ir.cpp.
const char* op_name(OpKind op);

// AST → IR. Reports undefined names, redefinitions, unknown ops and wrong
// argument counts. Defined in lower.cpp.
Graph lower(const Program& program, const std::string& filename);
