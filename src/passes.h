#pragma once

#include <ostream>
#include <string>

#include "ir.h"

// Sets the shape of every node and reports shape errors. Defined in shape.cpp.
void infer_shapes(Graph& graph, const std::string& filename);

struct CodegenOptions {
    bool fused;                 // was fusion run (shown in the header comment)
    int tile;                   // 0 = no tiling
    int unroll;                 // 0 = no unrolling
    std::string source_name;    // e.g. "mlp.tl", for the header comment
};

// Writes the whole generated C file. Defined in codegen.cpp.
void emit_c(const Graph& graph, const CodegenOptions& options, std::ostream& os);
