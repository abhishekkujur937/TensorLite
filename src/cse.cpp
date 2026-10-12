#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "passes.h"

// Key that is equal for two nodes that compute the same thing:
// op name + input ids, e.g. "matmul:0,1".
// add and mul are commutative (a+b == b+a), so their ids are sorted.
// matmul is not (A*B != B*A), so its ids keep their order.
static std::string cse_key(const Node* node)
{
    std::vector<int> ids;
    for (const Node* input : node->inputs) {
        ids.push_back(input->id);
    }
    if (node->op == OpKind::Add || node->op == OpKind::Mul) {
        std::sort(ids.begin(), ids.end());
    }

    std::string key = op_name(node->op);
    key += ":";
    for (std::size_t i = 0; i < ids.size(); i++) {
        if (i > 0) {
            key += ",";
        }
        key += std::to_string(ids[i]);
    }
    return key;
}

// Common subexpression elimination: if two nodes compute the same thing,
// keep the first one and let all users of the second use the first.
// Returns the number of merged nodes.
int run_cse(Graph& graph)
{
    std::map<std::string, Node*> seen;   // key → first node with that key
    int merged = 0;

    // Topo order: when we reach a node, its inputs were already merged,
    // so its key already uses the surviving input ids.
    for (Node* node : graph.topo_order()) {
        if (node->op == OpKind::Input || node->op == OpKind::Weight || node->op == OpKind::Const) {
            continue;
        }

        std::string key = cse_key(node);
        auto found = seen.find(key);
        if (found == seen.end()) {
            seen[key] = node;
            continue;
        }

        Node* first = found->second;

        if (node->is_output) {
            if (first->is_output) {
                // Both are outputs with their own names and files: keep both.
                continue;
            }
            // Move the output to the first node. It takes the output's name,
            // because the output name is used in the generated C and the
            // <name>.out.bin file.
            first->is_output = true;
            first->name = node->name;
            for (std::size_t i = 0; i < graph.outputs.size(); i++) {
                if (graph.outputs[i] == node) {
                    graph.outputs[i] = first;
                }
            }
            node->is_output = false;
        }

        graph.replace_all_uses(node, first);
        merged++;
    }

    // The duplicates now have no users: DCE deletes them.
    run_dce(graph);
    return merged;
}
