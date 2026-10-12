#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "passes.h"

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// Number of floats in a tensor: [64,128] → 8192, [256] → 256.
static int num_elements(const std::vector<int>& shape)
{
    int count = 1;
    for (int dim : shape) {
        count *= dim;
    }
    return count;
}

// "[64,128]" for comments in the generated C.
static std::string shape_comment(const std::vector<int>& shape)
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

// A float as a C literal: 0.5 → "0.5f", 2 → "2.0f".
// "2f" would not be valid C, so we add ".0" when there is no '.' or exponent.
static std::string float_literal(float value)
{
    std::ostringstream ss;
    ss.precision(9);   // 9 significant digits are enough to store any float exactly
    ss << value;
    std::string text = ss.str();
    if (text.find('.') == std::string::npos && text.find('e') == std::string::npos) {
        text += ".0";
    }
    return text + "f";
}

// True for nodes that compute something (not input / weight / const).
static bool is_computed(const Node* node)
{
    return node->op != OpKind::Input && node->op != OpKind::Weight && node->op != OpKind::Const;
}

// The C variable that holds a node's data:
//   input / weight    → its name (a tl_forward parameter)
//   output            → its name (a tl_forward parameter)
//   const             → c_<name> (static array)
//   other             → t<id> (static buffer)
static std::string c_name(const Node* node)
{
    if (node->op == OpKind::Input || node->op == OpKind::Weight) {
        return node->name;
    }
    if (node->is_output) {
        return node->name;
    }
    if (node->op == OpKind::Const) {
        return "c_" + node->name;
    }
    return "t" + std::to_string(node->id);
}

// Inputs and weights in declaration order. Node ids grow in source order,
// and graph.nodes keeps that order.
static std::vector<const Node*> param_nodes(const Graph& graph)
{
    std::vector<const Node*> params;
    for (const std::unique_ptr<Node>& node : graph.nodes) {
        if (node->op == OpKind::Input || node->op == OpKind::Weight) {
            params.push_back(node.get());
        }
    }
    return params;
}

// ---------------------------------------------------------------------------
// Kernels: one function per op, each writes a loop nest
// ---------------------------------------------------------------------------

// C[M,N] = A[M,K] x B[K,N], loop order i, j, k
static void emit_matmul(std::ostream& os, const Node* node)
{
    const Node* a = node->inputs[0];
    const Node* b = node->inputs[1];
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];
    std::string A = c_name(a);
    std::string B = c_name(b);
    std::string C = c_name(node);

    os << "    for (int i = 0; i < " << M << "; i++) {\n";
    os << "        for (int j = 0; j < " << N << "; j++) {\n";
    os << "            float acc = 0.0f;\n";
    os << "            for (int k = 0; k < " << K << "; k++) {\n";
    os << "                acc += " << A << "[i * " << K << " + k] * " << B << "[k * " << N << " + j];\n";
    os << "            }\n";
    os << "            " << C << "[i * " << N << " + j] = acc;\n";
    os << "        }\n";
    os << "    }\n";
}

// C[M,N] = relu(A[M,K] x B[K,N] + bias), loop order i, j, k.
// Same as matmul, but bias and relu are applied to acc before storing,
// so the matmul and add results never need their own buffers.
static void emit_fused(std::ostream& os, const Node* node)
{
    const Node* a = node->inputs[0];
    const Node* b = node->inputs[1];
    const Node* bias = node->inputs[2];
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];
    std::string A = c_name(a);
    std::string B = c_name(b);
    std::string BIAS = c_name(bias);
    std::string C = c_name(node);

    // bias [N] is the same for every row; bias [M,N] has one value per element.
    std::string bias_index = "[j]";
    if (bias->shape.size() == 2) {
        bias_index = "[i * " + std::to_string(N) + " + j]";
    }

    os << "    for (int i = 0; i < " << M << "; i++) {\n";
    os << "        for (int j = 0; j < " << N << "; j++) {\n";
    os << "            float acc = 0.0f;\n";
    os << "            for (int k = 0; k < " << K << "; k++) {\n";
    os << "                acc += " << A << "[i * " << K << " + k] * " << B << "[k * " << N << " + j];\n";
    os << "            }\n";
    os << "            acc += " << BIAS << bias_index << ";\n";
    os << "            " << C << "[i * " << N << " + j] = acc > 0.0f ? acc : 0.0f;\n";
    os << "        }\n";
    os << "    }\n";
}

// add / mul. Same shape → one loop over all elements.
// A[M,N] with B[N] → B[j] is reused for every row i.
static void emit_elementwise(std::ostream& os, const Node* node, const char* symbol)
{
    const Node* a = node->inputs[0];
    const Node* b = node->inputs[1];
    std::string A = c_name(a);
    std::string B = c_name(b);
    std::string C = c_name(node);

    if (a->shape == b->shape) {
        int size = num_elements(node->shape);
        os << "    for (int i = 0; i < " << size << "; i++) {\n";
        os << "        " << C << "[i] = " << A << "[i] " << symbol << " " << B << "[i];\n";
        os << "    }\n";
        return;
    }

    int M = a->shape[0];
    int N = a->shape[1];
    os << "    for (int i = 0; i < " << M << "; i++) {\n";
    os << "        for (int j = 0; j < " << N << "; j++) {\n";
    os << "            " << C << "[i * " << N << " + j] = " << A << "[i * " << N << " + j] "
       << symbol << " " << B << "[j];\n";
    os << "        }\n";
    os << "    }\n";
}

// relu: negative values become 0
static void emit_relu(std::ostream& os, const Node* node)
{
    std::string A = c_name(node->inputs[0]);
    std::string C = c_name(node);
    int size = num_elements(node->shape);

    os << "    for (int i = 0; i < " << size << "; i++) {\n";
    os << "        " << C << "[i] = " << A << "[i] > 0.0f ? " << A << "[i] : 0.0f;\n";
    os << "    }\n";
}

// ---------------------------------------------------------------------------
// tl_forward(...)
// ---------------------------------------------------------------------------

static void emit_forward(std::ostream& os, const Graph& graph, const std::vector<Node*>& order)
{
    // Signature: inputs/weights as const float*, then outputs as float*.
    os << "void tl_forward(";
    bool first = true;
    for (const Node* node : param_nodes(graph)) {
        if (!first) {
            os << ", ";
        }
        os << "const float* " << c_name(node);
        first = false;
    }
    for (const Node* node : graph.outputs) {
        if (!first) {
            os << ", ";
        }
        os << "float* " << c_name(node);
        first = false;
    }
    os << ")\n";
    os << "{\n";

    // Fill constants first.
    for (const Node* node : order) {
        if (node->op != OpKind::Const) {
            continue;
        }
        os << "    /* %" << node->id << " " << node->name << " = const "
           << shape_comment(node->shape) << " */\n";
        os << "    for (int i = 0; i < " << num_elements(node->shape) << "; i++) {\n";
        os << "        " << c_name(node) << "[i] = " << float_literal(node->const_value) << ";\n";
        os << "    }\n";
    }

    // Then every computed node, in topo order, so inputs are ready before use.
    for (const Node* node : order) {
        if (!is_computed(node)) {
            continue;
        }

        os << "    /* %" << node->id << " " << node->name << " = " << op_name(node->op);
        for (std::size_t i = 0; i < node->inputs.size(); i++) {
            if (i == 0) {
                os << " ";
            } else {
                os << ", ";
            }
            os << "%" << node->inputs[i]->id;
        }
        os << " " << shape_comment(node->shape) << " */\n";

        switch (node->op) {
            case OpKind::MatMul:
                emit_matmul(os, node);
                break;
            case OpKind::Add:
                emit_elementwise(os, node, "+");
                break;
            case OpKind::Mul:
                emit_elementwise(os, node, "*");
                break;
            case OpKind::Relu:
                emit_relu(os, node);
                break;
            case OpKind::FusedMatMulAddRelu:
                emit_fused(os, node);
                break;
            case OpKind::Input:
            case OpKind::Weight:
            case OpKind::Const:
                break;
        }
    }

    os << "}\n";
}

// ---------------------------------------------------------------------------
// main(): load inputs, time tl_forward, save outputs
// ---------------------------------------------------------------------------

// Fixed helper functions, the same in every generated file.
// R"C( ... )C" is a raw string: the text is copied as-is, no escaping needed.
static const char* RUNTIME_HELPERS = R"C(
/* Reads <dir>\<name>.bin (count raw float32 values). Exits on error. */
float* read_tensor(const char* dir, const char* name, int count)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s\\%s.bin", dir, name);
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "error: cannot open %s\n", path);
        exit(1);
    }
    float* data = (float*)malloc((size_t)count * sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "error: out of memory\n");
        exit(1);
    }
    size_t got = fread(data, sizeof(float), (size_t)count, f);
    fclose(f);
    if (got != (size_t)count) {
        fprintf(stderr, "error: %s: expected %d floats, read %d\n", path, count, (int)got);
        exit(1);
    }
    return data;
}

/* Allocates count floats for an output. Exits on error. */
float* alloc_tensor(int count)
{
    float* data = (float*)malloc((size_t)count * sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "error: out of memory\n");
        exit(1);
    }
    return data;
}

/* Writes <dir>\<name>.out.bin (count raw float32 values). Exits on error. */
void write_tensor(const char* dir, const char* name, const float* data, int count)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s\\%s.out.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "error: cannot write %s\n", path);
        exit(1);
    }
    fwrite(data, sizeof(float), (size_t)count, f);
    fclose(f);
}

/* For qsort: sorts doubles from small to large. */
int compare_doubles(const void* pa, const void* pb)
{
    double a = *(const double*)pa;
    double b = *(const double*)pb;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}
)C";

static void emit_main(std::ostream& os, const Graph& graph)
{
    std::vector<const Node*> params = param_nodes(graph);

    // "x, w1, b1, w2, y" for calling tl_forward
    std::string call_args;
    for (const Node* node : params) {
        if (!call_args.empty()) {
            call_args += ", ";
        }
        call_args += c_name(node);
    }
    for (const Node* node : graph.outputs) {
        if (!call_args.empty()) {
            call_args += ", ";
        }
        call_args += c_name(node);
    }

    os << RUNTIME_HELPERS;
    os << "\n";
    os << "int main(int argc, char** argv)\n";
    os << "{\n";
    os << "    if (argc != 3) {\n";
    os << "        fprintf(stderr, \"usage: %s <data_dir> <iterations>\\n\", argv[0]);\n";
    os << "        return 1;\n";
    os << "    }\n";
    os << "    const char* dir = argv[1];\n";
    os << "    int iters = atoi(argv[2]);\n";
    os << "    if (iters < 1) {\n";
    os << "        fprintf(stderr, \"error: iterations must be at least 1\\n\");\n";
    os << "        return 1;\n";
    os << "    }\n";
    os << "\n";

    for (const Node* node : params) {
        os << "    float* " << c_name(node) << " = read_tensor(dir, \"" << node->name << "\", "
           << num_elements(node->shape) << ");\n";
    }
    for (const Node* node : graph.outputs) {
        os << "    float* " << c_name(node) << " = alloc_tensor(" << num_elements(node->shape) << ");\n";
    }
    os << "\n";

    os << "    LARGE_INTEGER freq;\n";
    os << "    QueryPerformanceFrequency(&freq);\n";
    os << "\n";
    os << "    /* warm-up run (not timed) */\n";
    os << "    tl_forward(" << call_args << ");\n";
    os << "\n";
    os << "    double* times = (double*)malloc((size_t)iters * sizeof(double));\n";
    os << "    if (times == NULL) {\n";
    os << "        fprintf(stderr, \"error: out of memory\\n\");\n";
    os << "        return 1;\n";
    os << "    }\n";
    os << "    for (int it = 0; it < iters; it++) {\n";
    os << "        LARGE_INTEGER start;\n";
    os << "        LARGE_INTEGER end;\n";
    os << "        QueryPerformanceCounter(&start);\n";
    os << "        tl_forward(" << call_args << ");\n";
    os << "        QueryPerformanceCounter(&end);\n";
    os << "        times[it] = (double)(end.QuadPart - start.QuadPart) * 1000.0 / (double)freq.QuadPart;\n";
    os << "    }\n";
    os << "\n";
    os << "    qsort(times, (size_t)iters, sizeof(double), compare_doubles);\n";
    os << "    double median;\n";
    os << "    if (iters % 2 == 1) {\n";
    os << "        median = times[iters / 2];\n";
    os << "    } else {\n";
    os << "        median = (times[iters / 2 - 1] + times[iters / 2]) / 2.0;\n";
    os << "    }\n";
    os << "    printf(\"median_ms=%.6f\\n\", median);\n";
    os << "\n";

    for (const Node* node : graph.outputs) {
        os << "    write_tensor(dir, \"" << node->name << "\", " << c_name(node) << ", "
           << num_elements(node->shape) << ");\n";
    }
    os << "\n";

    for (const Node* node : params) {
        os << "    free(" << c_name(node) << ");\n";
    }
    for (const Node* node : graph.outputs) {
        os << "    free(" << c_name(node) << ");\n";
    }
    os << "    free(times);\n";
    os << "    return 0;\n";
    os << "}\n";
}

// ---------------------------------------------------------------------------
// Whole file
// ---------------------------------------------------------------------------

void emit_c(const Graph& graph, const CodegenOptions& options, std::ostream& os)
{
    std::vector<Node*> order = graph.topo_order();

    // Intermediate buffers = computed nodes that are not outputs.
    int buffer_count = 0;
    for (const Node* node : order) {
        if (is_computed(node) && !node->is_output) {
            buffer_count++;
        }
    }

    int fuse_flag = 0;
    if (options.fused) {
        fuse_flag = 1;
    }

    os << "/* generated by tlc from " << options.source_name << " */\n";
    os << "/* intermediate buffers: " << buffer_count << " */\n";
    os << "/* options: fuse=" << fuse_flag << " tile=" << options.tile
       << " unroll=" << options.unroll << " */\n";
    os << "#include <stdio.h>\n";
    os << "#include <stdlib.h>\n";
    os << "#include <string.h>\n";
    os << "#include <windows.h>\n";
    os << "\n";

    os << "/* static buffers for intermediates */\n";
    for (const Node* node : order) {
        if (is_computed(node) && !node->is_output) {
            os << "static float " << c_name(node) << "[" << num_elements(node->shape) << "];   /* "
               << node->name << " " << shape_comment(node->shape) << " */\n";
        }
    }
    os << "\n";

    os << "/* static arrays for constants */\n";
    for (const Node* node : order) {
        if (node->op == OpKind::Const && !node->is_output) {
            os << "static float " << c_name(node) << "[" << num_elements(node->shape) << "];   /* "
               << node->name << " " << shape_comment(node->shape) << " */\n";
        }
    }
    os << "\n";

    emit_forward(os, graph, order);
    emit_main(os, graph);
}
