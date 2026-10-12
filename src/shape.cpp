#include <cstddef>
#include <string>
#include <vector>

#include "error.h"
#include "passes.h"

// "[64,128]" for error messages.
static std::string shape_str(const std::vector<int>& shape)
{
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

// Declared shapes: every dimension must be > 0.
static void check_declared(const Node* node, const std::string& filename)
{
    for (int dim : node->shape) {
        if (dim <= 0) {
            report_error(filename, node->line, node->col,
                         "dimension must be positive in '" + node->name + "' " + shape_str(node->shape));
        }
    }
}

// matmul(A[M,K], B[K,N]) → [M,N]
static std::vector<int> matmul_shape(const Node* node, const std::vector<int>& a,
                                     const std::vector<int>& b, const std::string& filename)
{
    if (a.size() != 2 || b.size() != 2) {
        report_error(filename, node->line, node->col,
                     "matmul expects 2D inputs but got " + shape_str(a) + " and " + shape_str(b));
    }
    if (a[1] != b[0]) {
        report_error(filename, node->line, node->col,
                     "matmul inner dims mismatch: " + shape_str(a) + " x " + shape_str(b));
    }
    return {a[0], b[1]};
}

// add / mul: same shape, or A[M,N] with B[N] (B is repeated for every row) → shape of A
static std::vector<int> elementwise_shape(const Node* node, const std::vector<int>& a,
                                          const std::vector<int>& b, const std::string& filename)
{
    if (a == b) {
        return a;
    }
    if (a.size() == 2 && b.size() == 1 && b[0] == a[1]) {
        return a;
    }
    report_error(filename, node->line, node->col,
                 std::string("shape mismatch in ") + op_name(node->op) + ": "
                 + shape_str(a) + " and " + shape_str(b));
}

void infer_shapes(Graph& graph, const std::string& filename)
{
    // Topo order: when we reach a node, its inputs already have shapes.
    for (Node* node : graph.topo_order()) {
        switch (node->op) {
            case OpKind::Input:
            case OpKind::Weight:
            case OpKind::Const:
                check_declared(node, filename);
                break;
            case OpKind::MatMul:
                node->shape = matmul_shape(node, node->inputs[0]->shape, node->inputs[1]->shape, filename);
                break;
            case OpKind::Add:
            case OpKind::Mul:
                node->shape = elementwise_shape(node, node->inputs[0]->shape, node->inputs[1]->shape, filename);
                break;
            case OpKind::Relu:
                node->shape = node->inputs[0]->shape;
                break;
            case OpKind::FusedMatMulAddRelu:
                // Created only by fusion, after shapes are known: inputs are A, B, bias.
                node->shape = matmul_shape(node, node->inputs[0]->shape, node->inputs[1]->shape, filename);
                break;
        }
    }

    if (graph.outputs.empty()) {
        report_error(filename, 1, 1, "no output: the program needs at least one 'output' statement");
    }
}
