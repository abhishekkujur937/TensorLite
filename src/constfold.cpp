#include <cstddef>

#include "passes.h"

// True if every input of the node is a Const node.
static bool all_inputs_const(const Node* node)
{
    for (const Node* input : node->inputs) {
        if (input->op != OpKind::Const) {
            return false;
        }
    }
    return true;
}

// Constant folding: a node whose inputs are all constants is computed now,
// at compile time, and replaced by a new Const node.
// Every Const is uniform (one value fills the whole tensor), so one float is
// enough to describe the result. Returns the number of folded nodes.
int run_constfold(Graph& graph)
{
    int folded = 0;

    // Topo order: when a node is folded, its users see a Const input and can
    // be folded in the same walk (a chain folds completely).
    for (Node* node : graph.topo_order()) {
        if (node->op == OpKind::Input || node->op == OpKind::Weight || node->op == OpKind::Const) {
            continue;
        }
        if (!all_inputs_const(node)) {
            continue;
        }

        float value = 0.0f;
        if (node->op == OpKind::Add) {
            value = node->inputs[0]->const_value + node->inputs[1]->const_value;
        } else if (node->op == OpKind::Mul) {
            value = node->inputs[0]->const_value * node->inputs[1]->const_value;
        } else if (node->op == OpKind::Relu) {
            float a = node->inputs[0]->const_value;
            if (a > 0.0f) {
                value = a;
            } else {
                value = 0.0f;
            }
        } else if (node->op == OpKind::MatMul) {
            // Each output element is a sum of K equal products a*b.
            float K = static_cast<float>(node->inputs[0]->shape[1]);
            value = node->inputs[0]->const_value * node->inputs[1]->const_value * K;
        } else {
            continue;   // FusedMatMulAddRelu only exists after fusion
        }

        // New Const node with the same name and shape.
        Node* constant = graph.add_node(OpKind::Const, {}, node->name);
        constant->shape = node->shape;
        constant->const_value = value;
        constant->line = node->line;
        constant->col = node->col;

        if (node->is_output) {
            constant->is_output = true;
            node->is_output = false;
            for (std::size_t i = 0; i < graph.outputs.size(); i++) {
                if (graph.outputs[i] == node) {
                    graph.outputs[i] = constant;
                }
            }
        }

        graph.replace_all_uses(node, constant);
        folded++;
    }

    // The old nodes and unused constants are left for the final DCE.
    return folded;
}
