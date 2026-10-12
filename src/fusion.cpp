#include <cstddef>
#include <vector>

#include "passes.h"

// Operator fusion: relu(add(matmul(A, B), bias)) → fused_matmul_add_relu(A, B, bias).
// The fused kernel adds the bias and applies relu right after computing each
// matmul element, so the matmul and add results never go to memory.
FusionStats run_fusion(Graph& graph)
{
    FusionStats stats;
    stats.fused = 0;
    stats.refused = 0;

    for (Node* r : graph.topo_order()) {
        // 1. Find the pattern: relu ← add ← matmul
        if (r->op != OpKind::Relu) {
            continue;
        }
        Node* a = r->inputs[0];
        if (a->op != OpKind::Add) {
            continue;
        }

        Node* m = nullptr;
        Node* bias = nullptr;
        if (a->inputs[0]->op == OpKind::MatMul) {
            m = a->inputs[0];
            bias = a->inputs[1];
        } else if (a->inputs[1]->op == OpKind::MatMul) {
            m = a->inputs[1];
            bias = a->inputs[0];
        } else {
            continue;
        }

        // 2. Legality: the pattern matched, now check it is safe to fuse.
        // bias must be [N] or the same shape as the matmul result.
        int N = m->shape[1];
        bool bias_ok = (bias->shape == std::vector<int>{N}) || (bias->shape == m->shape);

        // matmul and add results disappear after fusion, so nobody else may
        // need them: exactly one user each, and not an output.
        bool m_ok = (m->users.size() == 1) && !m->is_output;
        bool a_ok = (a->users.size() == 1) && !a->is_output;

        if (!bias_ok || !m_ok || !a_ok) {
            stats.refused++;
            continue;
        }

        // 3. Build the fused node and let it take r's place.
        Node* fused = graph.add_node(OpKind::FusedMatMulAddRelu,
                                     {m->inputs[0], m->inputs[1], bias}, r->name);
        fused->shape = r->shape;
        fused->line = r->line;
        fused->col = r->col;

        if (r->is_output) {
            fused->is_output = true;
            r->is_output = false;
            for (std::size_t i = 0; i < graph.outputs.size(); i++) {
                if (graph.outputs[i] == r) {
                    graph.outputs[i] = fused;
                }
            }
        }

        graph.replace_all_uses(r, fused);
        stats.fused++;
    }

    // The old relu, add and matmul nodes now have no users.
    run_dce(graph);
    return stats;
}
