#include "ast.h"

#include <cstddef>
#include <string>

// Prints one expression and, below it, its arguments indented by 2 more spaces.
static void dump_expr(const Expr& expr, int depth, std::ostream& os)
{
    os << std::string(depth * 2, ' ');
    if (expr.args.empty()) {
        os << "Var " << expr.name << "\n";
        return;
    }

    os << "Call " << expr.name << "\n";
    for (const std::unique_ptr<Expr>& arg : expr.args) {
        dump_expr(*arg, depth + 1, os);
    }
}

// Prints dims as [64,128]
static void dump_dims(const std::vector<int>& dims, std::ostream& os)
{
    os << "[";
    for (std::size_t i = 0; i < dims.size(); i++) {
        if (i > 0) {
            os << ",";
        }
        os << dims[i];
    }
    os << "]";
}

void dump_ast(const Program& program, std::ostream& os)
{
    for (const Stmt& stmt : program.stmts) {
        if (stmt.kind == StmtKind::Decl) {
            os << "Decl " << stmt.decl_kind << " " << stmt.name << " ";
            dump_dims(stmt.dims, os);
            if (stmt.decl_kind == "const") {
                os << " = " << stmt.const_value;
            }
            os << "\n";
        } else if (stmt.kind == StmtKind::Assign) {
            os << "Assign " << stmt.name << "\n";
            dump_expr(*stmt.expr, 1, os);
        } else {
            os << "Output " << stmt.name << "\n";
        }
    }
}
