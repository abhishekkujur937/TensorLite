#pragma once

#include <string>

#include "ir.h"

// Sets the shape of every node and reports shape errors. Defined in shape.cpp.
void infer_shapes(Graph& graph, const std::string& filename);
