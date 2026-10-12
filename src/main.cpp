#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ast.h"
#include "ir.h"
#include "lexer.h"
#include "parser.h"
#include "passes.h"
#include "token.h"

// Prints how to use tlc.exe. Used when the command line is wrong.
static void print_usage()
{
    std::cerr << "usage: tlc.exe <input.tl> [-o <output.c>] [flags]\n";
    std::cerr << "\n";
    std::cerr << "  -o <file>        write generated C to <file>\n";
    std::cerr << "  --dump-tokens    print the token stream and exit\n";
    std::cerr << "  --dump-ast       print the AST\n";
    std::cerr << "  --dump-ir        print the IR after lowering and after every pass\n";
    std::cerr << "  --no-passes      skip DCE, CSE, constant folding\n";
    std::cerr << "  --no-fuse        skip fusion\n";
    std::cerr << "  --tile=N         0 (default) | 16 | 32 | 64\n";
    std::cerr << "  --unroll=N       0 (default) | 4\n";
}

// Reads the whole file into a string. Binary mode, so Windows does not
// change any bytes; the lexer handles \r itself.
static std::string read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "error: " << path << ": cannot open file\n";
        std::exit(1);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// "examples\mlp.tl" → "mlp.tl" (drops the folder part).
static std::string file_name_only(const std::string& path)
{
    std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return path;
    }
    return path.substr(slash + 1);
}

// Prints one token per line: line:col  KIND  'text'
static void dump_tokens(const std::vector<Token>& tokens)
{
    for (const Token& tok : tokens) {
        std::cout << tok.line << ":" << tok.col << "  "
                  << token_kind_name(tok.kind) << "  '" << tok.text << "'\n";
    }
}

int main(int argc, char** argv)
{
    std::string input_path;
    std::string output_path;
    bool flag_dump_tokens = false;
    bool flag_dump_ast = false;
    bool flag_dump_ir = false;
    bool flag_no_passes = false;
    bool flag_no_fuse = false;
    int tile = 0;
    int unroll = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dump-tokens") {
            flag_dump_tokens = true;
        } else if (arg == "--dump-ast") {
            flag_dump_ast = true;
        } else if (arg == "--dump-ir") {
            flag_dump_ir = true;
        } else if (arg == "--no-passes") {
            flag_no_passes = true;
        } else if (arg == "--no-fuse") {
            flag_no_fuse = true;
        } else if (arg.rfind("--tile=", 0) == 0) {
            // Only the exact values 0, 16, 32, 64 are allowed.
            std::string value = arg.substr(7);
            if (value == "0" || value == "16" || value == "32" || value == "64") {
                tile = std::stoi(value);
            } else {
                print_usage();
                return 1;
            }
        } else if (arg.rfind("--unroll=", 0) == 0) {
            // Only the exact values 0 and 4 are allowed.
            std::string value = arg.substr(9);
            if (value == "0" || value == "4") {
                unroll = std::stoi(value);
            } else {
                print_usage();
                return 1;
            }
        } else if (arg == "-o") {
            // The next argument is the output file.
            if (i + 1 >= argc) {
                print_usage();
                return 1;
            }
            i++;
            output_path = argv[i];
        } else if (!arg.empty() && arg[0] == '-') {
            // Unknown flag. Other flags are added in later stages.
            print_usage();
            return 1;
        } else if (input_path.empty()) {
            input_path = arg;
        } else {
            // A second input file is not allowed.
            print_usage();
            return 1;
        }
    }

    if (input_path.empty()) {
        print_usage();
        return 1;
    }

    std::string source = read_file(input_path);

    Lexer lexer(source, input_path);
    std::vector<Token> tokens = lexer.tokenize();

    if (flag_dump_tokens) {
        dump_tokens(tokens);
        return 0;
    }

    Parser parser(tokens, input_path);
    Program program = parser.parse_program();

    if (flag_dump_ast) {
        dump_ast(program, std::cout);
    }

    Graph graph = lower(program, input_path);

    if (flag_dump_ir) {
        graph.print(std::cout, "after lowering");
    }

    infer_shapes(graph, input_path);

    if (flag_dump_ir) {
        graph.print(std::cout, "after shape inference");
    }

    if (!flag_no_passes) {
        int removed = run_dce(graph);
        if (flag_dump_ir) {
            std::cerr << "DCE: removed " << removed << " nodes\n";
            graph.print(std::cout, "after DCE");
        }

        int merged = run_cse(graph);
        if (flag_dump_ir) {
            std::cerr << "CSE: merged " << merged << " nodes\n";
            graph.print(std::cout, "after CSE");
        }

        int folded = run_constfold(graph);
        if (flag_dump_ir) {
            std::cerr << "ConstFold: folded " << folded << " nodes\n";
            graph.print(std::cout, "after constant folding");
        }

        int removed_final = run_dce(graph);
        if (flag_dump_ir) {
            std::cerr << "DCE: removed " << removed_final << " nodes\n";
            graph.print(std::cout, "after final DCE");
        }
    }

    if (!flag_no_fuse) {
        FusionStats stats = run_fusion(graph);
        if (flag_dump_ir) {
            std::cerr << "Fusion: fused " << stats.fused << " chains, refused " << stats.refused << "\n";
            graph.print(std::cout, "after fusion");
        }
    }

    if (!output_path.empty()) {
        CodegenOptions options;
        options.fused = !flag_no_fuse;
        options.tile = tile;
        options.unroll = unroll;
        options.source_name = file_name_only(input_path);

        // Binary mode: lines end in \n only, exactly as written.
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            std::cerr << "error: " << output_path << ": cannot write file\n";
            return 1;
        }
        emit_c(graph, options, out);
    }

    return 0;
}
