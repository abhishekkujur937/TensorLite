#include <iostream>

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

int main(int argc, char** argv)
{
    (void)argv;

    if (argc < 2) {
        print_usage();
        return 1;
    }

    // The compiler pipeline is added stage by stage, starting in S1.
    print_usage();
    return 1;
}
