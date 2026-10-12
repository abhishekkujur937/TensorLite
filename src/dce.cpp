#include "passes.h"

// Dead code elimination: deletes every node the outputs do not depend on.
// The mark (walk back from outputs) and sweep (erase unmarked nodes) are in
// Graph::erase_unreachable(). Returns how many nodes were removed.
int run_dce(Graph& graph)
{
    int before = static_cast<int>(graph.nodes.size());
    graph.erase_unreachable();
    int after = static_cast<int>(graph.nodes.size());
    return before - after;
}
