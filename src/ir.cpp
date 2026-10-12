#include "ir.h"

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

const char* op_name(OpKind op)
{
    switch (op) {
        case OpKind::Input:              return "input";
        case OpKind::Weight:             return "weight";
        case OpKind::Const:              return "const";
        case OpKind::MatMul:             return "matmul";
        case OpKind::Add:                return "add";
        case OpKind::Mul:                return "mul";
        case OpKind::Relu:               return "relu";
        case OpKind::FusedMatMulAddRelu: return "fused_matmul_add_relu";
    }
    return "?";
}

// Creates a node, gives it the next id, and stores it in the graph.
// The graph owns the node; the returned pointer is only a reference to it.
Node* Graph::add_node(OpKind op, std::vector<Node*> inputs, std::string name)
{
    std::unique_ptr<Node> node = std::make_unique<Node>();
    node->id = next_id;
    next_id++;
    node->op = op;
    node->inputs = std::move(inputs);
    if (name.empty()) {
        node->name = "t" + std::to_string(node->id);
    } else {
        node->name = std::move(name);
    }

    Node* raw = node.get();
    nodes.push_back(std::move(node));
    return raw;
}

// Recomputes every node's users list from the inputs lists.
// If B uses A twice (add(a, a)), A gets B as a user twice, once per edge.
void Graph::rebuild_users()
{
    for (const std::unique_ptr<Node>& node : nodes) {
        node->users.clear();
    }
    for (const std::unique_ptr<Node>& node : nodes) {
        for (Node* input : node->inputs) {
            input->users.push_back(node.get());
        }
    }
}

// Kahn's algorithm. Returns nodes so that every node comes after all its inputs.
// Needs users to be up to date (call rebuild_users() after changing edges).
std::vector<Node*> Graph::topo_order() const
{
    // 1. in-degree = number of inputs that are not placed yet
    std::unordered_map<const Node*, int> in_degree;
    for (const std::unique_ptr<Node>& node : nodes) {
        in_degree[node.get()] = static_cast<int>(node->inputs.size());
    }

    // 2. start with the nodes that have no inputs
    std::queue<Node*> ready;
    for (const std::unique_ptr<Node>& node : nodes) {
        if (in_degree[node.get()] == 0) {
            ready.push(node.get());
        }
    }

    // 3. place a ready node; each user loses one waiting input
    std::vector<Node*> order;
    while (!ready.empty()) {
        Node* node = ready.front();
        ready.pop();
        order.push_back(node);

        for (Node* user : node->users) {
            in_degree[user]--;
            if (in_degree[user] == 0) {
                ready.push(user);
            }
        }
    }

    // 4. if some nodes were never placed, they wait on each other: a cycle
    if (order.size() != nodes.size()) {
        std::cerr << "internal error: cycle in graph\n";
        std::exit(1);
    }

    return order;
}

// Every node that had old_node as an input now has new_node instead.
void Graph::replace_all_uses(Node* old_node, Node* new_node)
{
    for (const std::unique_ptr<Node>& node : nodes) {
        if (node.get() == new_node) {
            continue;   // never make a node its own input
        }
        for (std::size_t i = 0; i < node->inputs.size(); i++) {
            if (node->inputs[i] == old_node) {
                node->inputs[i] = new_node;
            }
        }
    }
    rebuild_users();
}

// Deletes every node that the outputs do not depend on.
void Graph::erase_unreachable()
{
    // Mark: walk backwards from the outputs through inputs.
    std::unordered_set<Node*> reachable;
    std::vector<Node*> stack = outputs;
    while (!stack.empty()) {
        Node* node = stack.back();
        stack.pop_back();
        if (reachable.count(node) > 0) {
            continue;
        }
        reachable.insert(node);
        for (Node* input : node->inputs) {
            stack.push_back(input);
        }
    }

    // Sweep: keep only marked nodes. A kept node only points to kept nodes,
    // so no dangling pointer is left. Unmarked nodes are freed when the old
    // vector is destroyed at the end of this function.
    std::vector<std::unique_ptr<Node>> kept;
    for (std::unique_ptr<Node>& node : nodes) {
        if (reachable.count(node.get()) > 0) {
            kept.push_back(std::move(node));
        }
    }
    nodes = std::move(kept);

    rebuild_users();
}

// "[64,128]", or "?" when the shape is not known yet.
static std::string shape_text(const std::vector<int>& shape)
{
    if (shape.empty()) {
        return "?";
    }
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); i++) {
        if (i > 0) {
            text += ",";
        }
        text += std::to_string(shape[i]);
    }
    text += "]";
    return text;
}

// "matmul %0, %1" or "const 0.5": the op column of the dump.
static std::string op_text(const Node* node)
{
    std::ostringstream text;
    text << op_name(node->op);
    if (node->op == OpKind::Const) {
        text << " " << node->const_value;
    }
    for (std::size_t i = 0; i < node->inputs.size(); i++) {
        if (i == 0) {
            text << " ";
        } else {
            text << ", ";
        }
        text << "%" << node->inputs[i]->id;
    }
    return text.str();
}

// Prints the graph in topo order, in the format of CLAUDE.md Section 9.
void Graph::print(std::ostream& os, const std::string& title) const
{
    os << "=== IR " << title << " ===\n";

    // Name column is at least 4 wide, wider if a name is longer.
    // Op column is at least 22 wide, wider if an op text is longer
    // (+1 so there is always a space before the shape).
    std::size_t name_width = 4;
    std::size_t op_width = 22;
    for (const std::unique_ptr<Node>& node : nodes) {
        if (node->name.size() > name_width) {
            name_width = node->name.size();
        }
        std::size_t op_length = op_text(node.get()).size();
        if (op_length + 1 > op_width) {
            op_width = op_length + 1;
        }
    }

    for (Node* node : topo_order()) {
        std::string id_text = "%" + std::to_string(node->id);
        os << std::left << std::setw(4) << id_text
           << std::setw(static_cast<int>(name_width)) << node->name << " = "
           << std::setw(static_cast<int>(op_width)) << op_text(node)
           << shape_text(node->shape) << "\n";
    }

    os << "outputs:";
    for (std::size_t i = 0; i < outputs.size(); i++) {
        if (i > 0) {
            os << ",";
        }
        os << " " << outputs[i]->name << "(%" << outputs[i]->id << ")";
    }
    os << "\n";
}
