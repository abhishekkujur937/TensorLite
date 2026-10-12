#pragma once

#include <ostream>
#include <string>

#include "ir.h"

// Sets the shape of every node and reports shape errors. Defined in shape.cpp.
void infer_shapes(Graph& graph, const std::string& filename);

// Dead code elimination. Returns the number of removed nodes. Defined in dce.cpp.
int run_dce(Graph& graph);

// Common subexpression elimination (runs DCE at the end). Returns the number
// of merged nodes. Defined in cse.cpp.
int run_cse(Graph& graph);

// Constant folding. Returns the number of folded nodes. Defined in constfold.cpp.
int run_constfold(Graph& graph);

struct FusionStats {
    int fused;      // chains turned into one fused node
    int refused;    // chains that matched but were not legal to fuse
};

// matmul+add+relu fusion (runs DCE at the end). Defined in fusion.cpp.
FusionStats run_fusion(Graph& graph);

struct CodegenOptions {
    bool fused;                 // was fusion run (shown in the header comment)
    int tile;                   // 0 = no tiling
    int unroll;                 // 0 = no unrolling
    std::string source_name;    // e.g. "mlp.tl", for the header comment
};

// Writes the whole generated C file. Defined in codegen.cpp.
void emit_c(const Graph& graph, const CodegenOptions& options, std::ostream& os);
