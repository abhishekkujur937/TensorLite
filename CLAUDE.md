# CLAUDE.md — TensorLite: Mini ML Graph Compiler

Build guide for this repository. Read it fully before writing any code.
Follow it exactly. Do not add anything that is not in this file.

**Platform: Windows only (native). No Linux, no WSL, no Docker.**

---

## 1. What we are building

A compiler written in **C++17** that:

1. Reads a tiny neural-network language (`.tl` files)
2. Lexes and parses it into an AST
3. Lowers the AST into a graph IR (a DAG of operations)
4. Checks shapes
5. Optimizes the graph: dead code elimination, common subexpression elimination, constant folding
6. Fuses `matmul → add → relu` into one kernel
7. Emits **C code** (naive loops, or tiled + unrolled loops)

The generated C is compiled with `gcc`, run, timed, and checked against a **NumPy reference**
by a Python script.

### Example input (`examples/mlp.tl`)
```
input  x[64,128]
weight w1[128,256]
weight b1[256]
weight w2[256,10]
h = relu(add(matmul(x, w1), b1))
y = matmul(h, w2)
output y
```

### Full pipeline
```
.tl file
  → Lexer          (tokens)
  → Parser         (AST)
  → Lowering       (graph IR)
  → Shape check    (shapes on every node, errors)
  → Passes         (DCE → CSE → constant folding → DCE)
  → Fusion         (matmul+add+relu → one node)
  → Codegen        (.c file)
  → gcc            (.exe)
  → bench.py       (timing + NumPy check)
```

---

## 2. Rules for the AI assistant (MANDATORY)

1. **One stage at a time.** Only build the current stage (Section 12). Never generate the
   whole project at once.
2. **After each stage, explain it:** what every file and function does, why it is designed
   that way, and what concept it shows (e.g. recursive descent, topological sort, cache
   tiling). Then **STOP** and wait for the user to read, build, run, and say "next".
3. **The user must understand every line.** Write simple, readable code. If something is
   clever but hard to explain, rewrite it simpler.
4. **Formatting:** 4-space indentation, one statement per line, normal line breaks, braces on
   every `if`/`for`. Never compress code onto one line.
5. **Changing existing code:** tell the user the **file, the function, and the exact change**
   (show a before/after snippet). **Never give commands that edit files** — no `sed`, `awk`,
   PowerShell `Set-Content`/`-replace`, or similar. Commands are only for building and running.
6. **All commands are for Windows PowerShell.** Never give Linux/bash commands
   (`./`, `rm -rf`, `/dev/null`, `python3`, `lscpu`, `make`).
7. **Be direct and short.** Plain language. No padding, no repeating what was already agreed.
8. **No scope creep.** No extra ops, flags, files, or libraries beyond this document. If
   something seems missing, ask first.
9. **Never invent numbers.** Benchmark results only come from running `bench\bench.py`.
10. **Show the run command and expected output** at the end of every stage.
11. **Remind to commit** at the end of every stage with the given commit message.
12. **Ask first.** Before starting any stage, and before changing any existing file, state in
    2–3 lines what you are about to do and which files it touches. Then wait for the user's
    "yes". Never start the next stage on your own.

---

## 3. Windows environment

### Installed on this machine (verified)
| Tool | Version | Notes |
|---|---|---|
| g++ / gcc | 14.2.0 (MSYS2 build, Rev3) | Already on PATH. Full C++17 support. Nothing to install. |
| Python | 3.14.8 | System Python, used only to create the virtual environment |
| Git | 2.47.1.windows.1 | — |
| NumPy | installed inside `.venv` from `requirements.txt` | see below |

Repo location: `C:\Users\SAMSUNG\Desktop\TensorLite`

### Python virtual environment (one-time, done in S0)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
- If PowerShell blocks `Activate.ps1` with an execution-policy error, run once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and activate again.
- **Activate `.venv` at the start of every PowerShell session** before running any Python
  script. The prompt shows `(.venv)` when active.
- `.venv\` is in `.gitignore` — never commit it.

`requirements.txt`:
```
numpy>=2.3
```
(Only dependency. NumPy 2.3+ supports Python 3.14.)

**All building and running happens in normal Windows PowerShell.**

### Windows-specific rules (apply everywhere)
- Executables end in `.exe`: `tlc.exe`, `build\mlp.exe`.
- Run programs with `.\` : `.\tlc.exe`, `.\build.bat`.
- Paths in docs/commands use backslashes: `examples\mlp.tl`. Inside C++ and Python code,
  forward slashes or `pathlib` are fine.
- Python is invoked as `python`, not `python3`.
- **Binary files must be opened in binary mode**: `fopen(path, "rb")` / `"wb"` in C,
  `std::ios::binary` in C++, `np.fromfile` / `.tofile` in Python. Text mode on Windows
  corrupts binary data.
- **Source files may have `\r\n` line endings.** The lexer must treat `\r` as whitespace.
- Windows Defender may scan a new `.exe` on first run — the warm-up run in the generated
  `main()` absorbs this, so it never affects timing.

---

## 4. Locked technical decisions

| Item | Decision |
|---|---|
| OS | Windows 10/11, native |
| Toolchain | MinGW-w64 GCC 14.2 from MSYS2 **mingw64** (`C:\msys64\mingw64\bin`) — `g++` for the compiler, `gcc` for generated C |
| Language | C++17 for the compiler, C for generated output |
| Compiler flags | `g++ -std=c++17 -Wall -Wextra -O2 -g -static-libstdc++ -static-libgcc` |
| Static runtime | `tlc.exe` links the C++ runtime statically. Reason: an older `libstdc++-6.dll` (from `C:\iverilog\bin`) comes earlier on PATH and crashed `tlc.exe` with an access violation. Static linking removes any dependency on PATH order. Do not remove these flags. |
| Build system | `build.bat` (one script, rebuilds everything — project is small) |
| Binary | `tlc.exe` |
| Shell | PowerShell |
| User-visible ops | `matmul`, `add`, `relu`, `mul` |
| Internal op | `fused_matmul_add_relu` (created only by fusion) |
| Tensor rank | 1D or 2D only |
| Data type | `float` only |
| Memory layout | Row-major: element `[i][j]` of an `[M,N]` tensor is at `i*N + j` |
| Constants | `const c[N] = 0.5` → tensor filled with one value |
| AST memory | `std::unique_ptr` for child expressions |
| IR memory | `Graph` owns nodes in `std::vector<std::unique_ptr<Node>>`; edges are non-owning `Node*` |
| Topological sort | Kahn's algorithm |
| Generated C | One `.c` file with `tl_forward(...)` and `main()` |
| Timing | `QueryPerformanceCounter` (`<windows.h>`) inside generated C, median of N runs |
| Correctness | `np.allclose(out, ref, rtol=1e-4, atol=1e-4)` |
| Dependencies | C++: standard library only. Python: `numpy` only |
| Version control | One git commit per stage |

---

## 5. Repository layout

```
tensorlite\
├── CLAUDE.md
├── build.bat               # builds tlc.exe
├── clean.bat               # deletes build\ and tlc.exe
├── README.md
├── requirements.txt        # numpy>=2.3
├── .venv\                  # Python virtual environment (not committed)
├── .gitignore              # .venv/, build/, tlc.exe, *.exe, *.o, bench/out/, *.bin, __pycache__/
├── .gitattributes          # *.bat text eol=crlf
├── src\
│   ├── main.cpp            # CLI flags, runs the pipeline in order
│   ├── token.h             # TokenKind enum, Token struct
│   ├── lexer.h
│   ├── lexer.cpp
│   ├── ast.h               # AST structs
│   ├── parser.h
│   ├── parser.cpp
│   ├── ast_dump.cpp        # --dump-ast printer
│   ├── ir.h                # OpKind, Node, Graph
│   ├── ir.cpp              # Graph methods, topo sort, IR printer
│   ├── lower.cpp           # AST → IR
│   ├── shape.cpp           # shape inference + semantic errors
│   ├── error.h             # error reporting helper (prints and exits)
│   ├── passes.h            # declarations of all passes
│   ├── dce.cpp
│   ├── cse.cpp
│   ├── constfold.cpp
│   ├── fusion.cpp
│   └── codegen.cpp         # naive / tiled / unrolled C emission
├── examples\
│   ├── mlp.tl
│   ├── dead_code.tl
│   ├── duplicate.tl
│   ├── constants.tl
│   ├── multi_user.tl
│   ├── odd_dims.tl
│   └── bench_large.tl
├── tests\
│   ├── errors\             # one .tl file per error case
│   └── run_tests.py
└── bench\
    ├── reference.py        # independent NumPy interpreter of .tl files
    ├── runner.py           # shared helpers: compile, run, compare
    └── bench.py            # full benchmark, writes bench\results.md
```

---

## 6. The language

### Grammar (recursive descent — one parser function per rule)
```
program  := stmt* EOF
stmt     := decl | assign | output
decl     := ('input' | 'weight') IDENT '[' dims ']'
          | 'const' IDENT '[' dims ']' '=' NUMBER
dims     := NUMBER (',' NUMBER)?
assign   := IDENT '=' expr
expr     := IDENT
          | IDENT '(' expr (',' expr)* ')'
output   := 'output' IDENT
```

### Lexical rules
- Whitespace is skipped: space, tab, `\n`, **and `\r`** (Windows line endings).
- `#` starts a comment until end of line.
- Identifier: `[A-Za-z_][A-Za-z0-9_]*`
- Number: `[0-9]+` or `[0-9]+.[0-9]+` (optionally leading `-` for const values only)
- Keywords: `input`, `weight`, `const`, `output`
- Symbols: `( ) [ ] , =`
- Op names (`matmul`, `add`, `relu`, `mul`) are normal identifiers; lowering checks them.

### Op rules
| Op | Inputs | Output shape | Check |
|---|---|---|---|
| `matmul(A, B)` | A[M,K], B[K,N] | [M,N] | both 2D, inner dims equal |
| `add(A, B)` | A and B same shape, OR A[M,N] and B[N] | shape of A | broadcast B across rows |
| `mul(A, B)` | same as `add` | shape of A | elementwise |
| `relu(A)` | any | shape of A | exactly 1 argument |

### Semantic rules
- Every name is defined once (decl or assign).
- A name must be defined before it is used.
- Dimensions must be > 0.
- At least one `output`.
- Wrong number of arguments to an op is an error.

---

## 7. Data structures

### Tokens (`token.h`)
```cpp
enum class TokenKind {
    Ident,
    Number,
    KwInput,
    KwWeight,
    KwConst,
    KwOutput,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Equals,
    End
};

struct Token {
    TokenKind kind;
    std::string text;
    int line;
    int col;
};
```

### AST (`ast.h`)
```cpp
struct Expr {
    std::string name;                          // variable name, or op name if it is a call
    std::vector<std::unique_ptr<Expr>> args;   // empty → plain variable reference
    int line;
    int col;
};

enum class StmtKind { Decl, Assign, Output };

struct Stmt {
    StmtKind kind;
    std::string decl_kind;        // "input" | "weight" | "const"   (Decl only)
    std::string name;
    std::vector<int> dims;        // Decl only
    float const_value = 0.0f;     // const Decl only
    std::unique_ptr<Expr> expr;   // Assign only
    int line;
    int col;
};

struct Program {
    std::vector<Stmt> stmts;      // keeps source order
};
```

### IR (`ir.h`)
```cpp
enum class OpKind {
    Input,
    Weight,
    Const,
    MatMul,
    Add,
    Mul,
    Relu,
    FusedMatMulAddRelu    // inputs: A, B, bias
};

struct Node {
    int id;
    OpKind op;
    std::string name;             // user name if given, else "t<id>"
    std::vector<int> shape;
    std::vector<Node*> inputs;    // non-owning
    std::vector<Node*> users;     // non-owning, rebuilt by Graph::rebuild_users()
    float const_value = 0.0f;     // Const only
    bool is_output = false;
    int line = 0;                 // for error messages
    int col = 0;
};

struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;   // owns all nodes
    std::vector<Node*> outputs;
    int next_id = 0;

    Node* add_node(OpKind op, std::vector<Node*> inputs, std::string name);
    void rebuild_users();
    std::vector<Node*> topo_order() const;      // Kahn's algorithm
    void replace_all_uses(Node* old_node, Node* new_node);
    void erase_unreachable();                   // used by DCE
    void print(std::ostream& os, const std::string& title) const;
};
```

**Memory rules**
- Only `Graph::nodes` owns nodes. Nothing else ever `delete`s or owns a `Node`.
- Before erasing a node, no other node may still have it in `inputs`.
- After any pass that changes edges, call `rebuild_users()`.

---

## 8. CLI (`main.cpp`)

```
.\tlc.exe <input.tl> [-o <output.c>] [flags]

-o <file>        write generated C to <file>. If omitted, no C is generated
                 (useful with the dump flags).
--dump-tokens    print the token stream and exit
--dump-ast       print the AST
--dump-ir        print the IR after lowering and after every pass
--no-passes      skip DCE, CSE, constant folding
--no-fuse        skip fusion
--tile=N         0 (default, no tiling) | 16 | 32 | 64
--unroll=N       0 (default) | 4
```

Pipeline order inside `main()`:
```
read file (binary mode, whole file into a string)
→ lex
→ parse              (--dump-ast)
→ lower to IR        (--dump-ir "after lowering")
→ infer shapes       (--dump-ir "after shape inference")
→ unless --no-passes:
     DCE             (--dump-ir "after DCE")
     CSE             (--dump-ir "after CSE")
     constant fold   (--dump-ir "after constant folding")
     DCE             (--dump-ir "after final DCE")
→ unless --no-fuse:
     fusion          (--dump-ir "after fusion")
→ if -o given: codegen to that file
```
Unknown flags or a missing input file print usage and exit with code 1.

---

## 9. Output formats

### Error messages (`error.h`)
```
error: <file>:<line>:<col>: <message>
```
Printed to `stderr`; process exits with code 1 on the first error.

Required messages (tests match on these substrings):
| Case | Message contains |
|---|---|
| Bad character | `unexpected character` |
| Parser expected something else | `expected` |
| Undefined name | `undefined name` |
| Name defined twice | `redefinition of` |
| Unknown op | `unknown op` |
| Wrong argument count | `expects` (e.g. `relu expects 1 argument`) |
| matmul mismatch | `matmul inner dims mismatch` |
| add/mul mismatch | `shape mismatch` |
| Zero dimension | `dimension must be positive` |
| No output | `no output` |

### `--dump-ir` format
```
=== IR after lowering ===
%0  x    = input                 [64,128]
%1  w1   = weight                [128,256]
%2  b1   = weight                [256]
%3  w2   = weight                [256,10]
%4  t4   = matmul %0, %1         ?
%5  t5   = add %4, %2            ?
%6  h    = relu %5               ?
%7  y    = matmul %6, %3         ?
outputs: y(%7)
=== IR after shape inference ===
%0  x    = input                 [64,128]
%1  w1   = weight                [128,256]
%2  b1   = weight                [256]
%3  w2   = weight                [256,10]
%4  t4   = matmul %0, %1         [64,256]
%5  t5   = add %4, %2            [64,256]
%6  h    = relu %5               [64,256]
%7  y    = matmul %6, %3         [64,10]
outputs: y(%7)
```
- Computed nodes show `?` until shape inference has run. Declared nodes (input / weight /
  const) show their shapes from the start.
- The name column widens to fit the longest name in the graph.
- Any valid topological order is acceptable; nodes that become ready at the same time may
  print in either order.

---

## 10. Generated C contract (`codegen.cpp`)

### File structure
```c
/* generated by tlc from <file>.tl */
/* intermediate buffers: N */
/* options: fuse=<0|1> tile=<T> unroll=<U> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

/* static buffers for intermediates */
/* static arrays for constants */

void tl_forward(const float* x, const float* w1, ..., float* y);

int main(int argc, char** argv);
```

### Rules
- Parameters of `tl_forward`: every Input and Weight (in declaration order) as `const float*`,
  then every output as `float*`.
- Every non-output computed node gets a `static float t<id>[size];` buffer.
- An output node writes directly into its output parameter.
- Constants: `static float c_<name>[size];` filled at the start of `tl_forward`.
- The header comment `intermediate buffers: N` counts the `static` intermediate buffers.
  The benchmark reads it to report "buffers eliminated".
- Ops are emitted in `topo_order()`.

### `main()` behaviour
- `argv[1]` = data directory, `argv[2]` = iteration count.
- Allocate (`malloc`) and load each Input/Weight from `<dir>\<name>.bin` using
  `fopen(path, "rb")` (raw float32, row-major).
- Allocate each output.
- 1 warm-up call, then N timed calls.
- Timing with `QueryPerformanceFrequency` once and `QueryPerformanceCounter` before/after each
  call; convert to milliseconds as `(end - start) * 1000.0 / freq`.
- Sort the N times, print `median_ms=<value>` on stdout.
- Write each output to `<dir>\<name>.out.bin` using `fopen(path, "wb")`.
- Missing file → print error, exit 1.

### Loop variants

**Naive matmul** — `C[M,N] = A[M,K] × B[K,N]`, loop order `i, j, k`:
```c
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = acc;
    }
}
```

**Elementwise add/mul/relu** — one loop over all elements; broadcast uses `B[j]`.

**Fused matmul+add+relu** — same as naive matmul, but before storing:
`acc += bias[j]; C[i * N + j] = acc > 0.0f ? acc : 0.0f;` — no intermediate buffers.

**Tiled (`--tile=T`)** — six loops:
```
for ii in 0..M step T
  for jj in 0..N step T
    zero the output tile
    for kk in 0..K step T
      for i in ii..min(ii+T, M)
        for k in kk..min(kk+T, K)
          a = A[i*K + k]
          for j in jj..min(jj+T, N)
            C[i*N + j] += a * B[k*N + j]
    (fused only) apply bias + relu to the output tile
```
Inner order `i, k, j` makes the innermost loop walk `B` and `C` contiguously.
Dimensions do NOT need to be divisible by `T` — always use `min()` bounds.

**Unrolled (`--unroll=4`)** — the innermost loop is unrolled by 4, followed by a remainder
loop for leftover iterations. Applies to the innermost loop of both naive and tiled matmul.

---

## 11. Example files (create exactly these)

`examples\mlp.tl` — see Section 1.

`examples\dead_code.tl`
```
input  x[64,128]
weight w1[128,256]
weight w2[128,64]
h = matmul(x, w1)
unused = matmul(x, w2)
output h
```

`examples\duplicate.tl`
```
input  x[64,128]
weight w1[128,256]
a = matmul(x, w1)
b = matmul(x, w1)
y = add(a, b)
output y
```

`examples\constants.tl`
```
input  x[64,128]
weight w1[128,256]
const  s1[256] = 0.5
const  s2[256] = 2.0
scale = mul(s1, s2)
h = matmul(x, w1)
y = mul(h, scale)
output y
```

`examples\multi_user.tl`
```
input  x[64,128]
weight w1[128,256]
weight b1[256]
m = matmul(x, w1)
h = relu(add(m, b1))
y = add(h, m)
output y
```
(`m` has two users → fusion must be refused.)

`examples\odd_dims.tl`
```
input  x[63,129]
weight w1[129,70]
weight b1[70]
y = relu(add(matmul(x, w1), b1))
output y
```
(Dimensions not divisible by any tile size → tests `min()` bounds.)

`examples\bench_large.tl`
```
input  x[512,1024]
weight w1[1024,1024]
weight b1[1024]
weight w2[1024,512]
weight b2[512]
h = relu(add(matmul(x, w1), b1))
y = relu(add(matmul(h, w2), b2))
output y
```
(Large enough that the working set does not fit in cache, so tiling shows a real effect.)

### Error test files (`tests\errors\`)
Each file's first line is `# expect: <substring>`. `run_tests.py` checks that `tlc.exe` exits
with code 1 and stderr contains that substring.

| File | Content idea |
|---|---|
| `bad_char.tl` | contains `@` |
| `missing_bracket.tl` | `input x[64,128` |
| `undefined_name.tl` | uses `w3` never declared |
| `redefined.tl` | declares `x` twice |
| `unknown_op.tl` | `y = conv(x, w)` |
| `wrong_arity.tl` | `y = relu(x, w)` |
| `matmul_mismatch.tl` | `matmul(x[64,128], w[256,10])` |
| `add_mismatch.tl` | `add(a[64,256], b[128])` |
| `zero_dim.tl` | `input x[0,128]` |
| `no_output.tl` | no `output` statement |

---

## 12. Build stages

Every stage has: **Files → Build → Run → Done when → Commit.**
Do not start a stage until the previous one is done and committed.
All commands below are run in **PowerShell** from the repo root.

---

### S0 — Project setup (30 min)

**Files:** `build.bat`, `clean.bat`, `requirements.txt`, `.gitignore`, `.gitattributes`,
`src\main.cpp`, empty folders with `.gitkeep`.

**Environment:** `git init`, create and activate `.venv`, install `requirements.txt`
(Section 3).

**Build:**
- `build.bat`:
  ```bat
  @echo off
  setlocal enabledelayedexpansion

  if not exist build mkdir build

  set SRCS=
  for %%f in (src\*.cpp) do (
      set SRCS=!SRCS! %%f
  )

  g++ -std=c++17 -Wall -Wextra -O2 -g -static-libstdc++ -static-libgcc -o tlc.exe !SRCS!
  if errorlevel 1 (
      echo BUILD FAILED
      exit /b 1
  )
  echo BUILD OK
  ```
- `clean.bat`:
  ```bat
  @echo off
  if exist build rmdir /s /q build
  if exist tlc.exe del tlc.exe
  echo CLEAN OK
  ```
- `main.cpp` prints usage and exits.

**Run:**
```
.\build.bat
.\tlc.exe
```

**Done when:** `BUILD OK` with zero warnings; usage prints;
`python -c "import numpy; print(numpy.__version__)"` works inside the active `.venv`.

**Commit:** `S0: project skeleton and build scripts`

---

### S1 — Lexer (1 h)

**Files:** `token.h`, `lexer.h`, `lexer.cpp`, `error.h`, update `main.cpp`.

**Build:**
- `class Lexer` with `std::vector<Token> tokenize()`.
- Track `line` and `col` for every token.
- Skip whitespace (including `\r`) and `#` comments.
- Map keyword text to keyword kinds.
- Unknown character → `error: file:line:col: unexpected character '@'`.
- `--dump-tokens` prints one token per line: `line:col  KIND  'text'`.

**Run:** `.\tlc.exe examples\mlp.tl --dump-tokens`

**Done when:** token dump of `mlp.tl` is correct; `tests\errors\bad_char.tl` gives the right
error.

**Commit:** `S1: lexer with line/col tracking`

---

### S2 — Parser + AST (1.5 h)

**Files:** `ast.h`, `parser.h`, `parser.cpp`, `ast_dump.cpp`, update `main.cpp`.

**Build:**
- `class Parser` holding the token vector and a position index.
- Helper functions: `peek()`, `advance()`, `expect(kind, what)`.
- One function per grammar rule: `parse_program`, `parse_stmt`, `parse_decl`, `parse_assign`,
  `parse_expr`, `parse_output`, `parse_dims`.
- `expect` failure → `expected ']' but found ','` with line/col.
- `--dump-ast` prints an indented tree:
  ```
  Decl input x [64,128]
  ...
  Assign h
    Call relu
      Call add
        Call matmul
          Var x
          Var w1
        Var b1
  Output y
  ```

**Run:** `.\tlc.exe examples\mlp.tl --dump-ast`

**Done when:** AST is correct for all examples; `missing_bracket.tl` fails correctly.

**Commit:** `S2: recursive descent parser and AST`

---

### S3 — Graph IR + lowering + topological sort (1 h)

**Files:** `ir.h`, `ir.cpp`, `lower.cpp`, update `main.cpp`.

**Build:**
- `Graph` methods from Section 7.
- `lower(const Program&, const std::string& filename) → Graph` (filename is used in error
  messages):
  - Symbol table `std::unordered_map<std::string, Node*>`.
  - Decl → Input / Weight / Const node.
  - Assign → recursively lower the expression; each call becomes one node; the top node gets
    the user's name; inner nodes get `t<id>`.
  - Output → mark node `is_output`, add to `graph.outputs`.
  - Errors: undefined name, redefinition, unknown op, wrong argument count.
- `topo_order()` with Kahn's algorithm:
  1. count in-degree (number of inputs) of each node
  2. queue all nodes with in-degree 0
  3. pop a node, append it, decrease in-degree of its users, push any that reach 0
  4. if result size ≠ node count → cycle (internal error)
- `print()` in the format of Section 9, using topo order.

**Run:** `.\tlc.exe examples\mlp.tl --dump-ir`

**Done when:** IR dump correct; undefined/redefined/unknown/arity error files fail correctly.

**Commit:** `S3: graph IR, lowering, topological sort`

---

### S4 — Shape inference (1 h)

**Files:** `shape.cpp`, `passes.h`, update `main.cpp`, `tests\errors\*.tl`, `tests\run_tests.py`.

**Build:**
- `void infer_shapes(Graph&, const std::string& filename)`.
- Walk topo order; set `shape` per Section 6 op rules; report errors with node line/col.
- Check zero dims and the no-output rule.
- Create the remaining 4 error files from Section 11 (`matmul_mismatch`, `add_mismatch`,
  `zero_dim`, `no_output`). The other 6 already exist from S1–S3.
- Add the `--dump-ir` "after shape inference" dump (Section 8).
- First version of `tests\run_tests.py`: runs `tlc.exe` on every `tests\errors\*.tl` via
  `subprocess.run`, checks exit code 1 and expected substring; prints PASS/FAIL per file and
  a summary. Uses `pathlib` for all paths.

**Run:** `python tests\run_tests.py`

**Done when:** all 10 error tests PASS; IR dump shows shapes on every node.

**Commit:** `S4: shape inference and error tests`

---

### S5 — Naive C codegen (2 h)

**Files:** `codegen.cpp`, update `passes.h`, `main.cpp`.

**Build:**
- `void emit_c(const Graph&, const CodegenOptions&, std::ostream&)`.
- `struct CodegenOptions { bool fused; int tile; int unroll; std::string source_name; };`
- Everything from Section 10 except tiling, unrolling, and fused kernels.

**Run:**
```
.\tlc.exe examples\mlp.tl -o build\mlp.c
gcc -O2 -Wall -Wextra build\mlp.c -o build\mlp.exe
```

**Done when:** generated C is readable and compiles with zero warnings for every example.

**Commit:** `S5: naive C code generation`

---

### S6 — NumPy reference + correctness check (1 h)

**Files:** `bench\reference.py`, `bench\runner.py`, update `tests\run_tests.py`.

**Build:**
- `reference.py`: a small independent Python interpreter of `.tl` files using regex parsing
  and NumPy (`matmul` → `a @ b`, `add` → `a + b`, `mul` → `a * b`, `relu` →
  `np.maximum(a, 0)`). Does NOT share code with the C++ compiler. All arrays `float32`.
- `runner.py` functions (all paths via `pathlib`, all processes via `subprocess.run`):
  - `make_inputs(tl_path, data_dir, seed=0)` → random float32 inputs/weights in
    `[-1, 1]`, written with `.tofile()` as `<name>.bin`
  - `compile_tl(tl_path, c_path, flags)` → runs `tlc.exe`
  - `compile_c(c_path, exe_path, opt)` → runs `gcc -<opt> -o exe_path c_path`
  - `run_exe(exe_path, data_dir, iters)` → parses and returns `median_ms`
  - `check(tl_path, data_dir)` → loads `.out.bin` with `np.fromfile(..., dtype=np.float32)`,
    compares with `reference.py`, returns PASS/FAIL and max abs error
- `run_tests.py` now also runs every example (except `bench_large.tl`) end-to-end with
  default flags and checks correctness. Temporary files go in `build\test_out\`.

**Run:** `python tests\run_tests.py`

**Done when:** every example PASSES against NumPy.

**Commit:** `S6: end-to-end pipeline verified against NumPy`

**⚑ CHECKPOINT 1: compiler works end to end. If not reached on time → Section 13.**

---

### S7 — Dead code elimination + `--dump-ir` per pass (45 min)

**Files:** `dce.cpp`, update `passes.h`, `main.cpp`. (`Graph::erase_unreachable()` in
`ir.cpp` already exists from S3.)

**Build:**
- Mark phase: start from `graph.outputs`, walk `inputs` backwards (DFS/stack), mark all
  reachable nodes.
- Sweep phase: erase every unmarked node from `graph.nodes`.
- Print `DCE: removed N nodes` to stderr when `--dump-ir` is on.
- Wire `--dump-ir` dumps after each pass (Section 8 order).

**Run:** `.\tlc.exe examples\dead_code.tl --dump-ir`

**Done when:** `unused` disappears after DCE; all tests still PASS.

**Commit:** `S7: dead code elimination and per-pass IR dumps`

---

### S8 — Common subexpression elimination (1 h)

**Files:** `cse.cpp`.

**Build:**
- Walk nodes in topo order (so every node's inputs are already deduplicated).
- Skip Input / Weight / Const.
- Key = op + list of input node ids. For `add` and `mul` sort the two input ids
  (commutative). For `matmul` do NOT sort.
- Key format: `"<op>:<id>,<id>"`, e.g. `"matmul:0,1"`.
- `std::map<std::string, Node*>` keyed by that string: if key seen →
  `replace_all_uses(duplicate, first)`.
- **Output rules:**
  - If the duplicate is an output and `first` is not: move the output to `first` **and
    rename `first` to the output's name** (the output name is the C parameter name and the
    `<name>.out.bin` file, so it must survive).
  - If **both** are outputs: do **not** merge (each output needs its own parameter and file;
    one node cannot have two names).
- Run DCE afterwards (`run_dce`) to delete the now-unused duplicate.
- Returns the number of merged nodes. Print `CSE: merged N nodes`.
- One pass catches chains (duplicate matmul → duplicate relu → merged 2) because of topo order.

**Run:** `.\tlc.exe examples\duplicate.tl --dump-ir`

**Done when:** only one matmul remains; output still correct.

**Commit:** `S8: common subexpression elimination`

---

### S9 — Constant folding (1 h)

**Files:** `constfold.cpp`.

**Build:**
- Walk topo order. If every input of a node is a `Const` node, replace the node with a new
  `Const` node of the same shape:
  - `add` → `a + b`
  - `mul` → `a * b`
  - `relu` → `max(a, 0)`
  - `matmul` of uniform constants → `a * b * K`
- `replace_all_uses(old, new_const)`; keep the user's name on the new const.
- Print `ConstFold: folded N nodes`.
- Codegen already handles `Const` nodes (filled arrays).

**Run:** `.\tlc.exe examples\constants.tl --dump-ir`

**Done when:** `scale` becomes a `const` with value `1.0`; `s1`, `s2` removed by final DCE;
output still correct.

**Commit:** `S9: constant folding`

**⚑ CHECKPOINT 2: all three passes visible in `--dump-ir`, all tests PASS.**

---

### S10 — Operator fusion (1.5 h)

**Files:** `fusion.cpp`, update `codegen.cpp`.

**Build:**
- For each `Relu` node `r`:
  - `a = r->inputs[0]` must be `Add`
  - `m = a->inputs[0]` (or `inputs[1]`) must be `MatMul`; the other `Add` input is the bias
  - bias must be 1D `[N]` or same shape as `m`
  - **Legality:** `m` has exactly 1 user, `a` has exactly 1 user, and neither `m` nor `a`
    is an output. If any check fails → do not fuse.
- Create `FusedMatMulAddRelu` node with inputs `(A, B, bias)`, shape of `r`, name of `r`;
  `replace_all_uses(r, fused)`; move output flag; run DCE.
- Print `Fusion: fused N chains, refused M`.
- Codegen: emit the fused kernel from Section 10.

**Run:**
```
.\tlc.exe examples\mlp.tl --dump-ir
.\tlc.exe examples\multi_user.tl --dump-ir
python tests\run_tests.py
```

**Done when:** `mlp.tl` shows one fused node; `multi_user.tl` shows `refused 1`; buffer
count in the C header goes down; all tests PASS.

**Commit:** `S10: matmul+add+relu fusion with legality check`

---

### S11 — Tiled + unrolled codegen (1.5 h)

**Files:** update `codegen.cpp`, `main.cpp`, `tests\run_tests.py`.

**Build:**
- Parse `--tile` (0/16/32/64) and `--unroll` (0/4); reject other values.
- Tiled matmul and tiled fused kernel per Section 10.
- Unroll innermost loop by 4 with remainder loop.
- `run_tests.py`: run every example (except `bench_large.tl`) with every combination of
  `fuse {0,1} × tile {0,16,32,64} × unroll {0,4}` and check correctness.
- `odd_dims.tl` must pass every combination.

**Run:** `python tests\run_tests.py`

**Done when:** every combination PASSES, including `odd_dims.tl`.

**Commit:** `S11: tiled and unrolled code generation`

---

### S12 — Benchmark (1.5 h)

**Files:** `bench\bench.py`, output `bench\results.md`.

**Build:**
- Model: `examples\bench_large.tl`.
- Configurations:

  | Name | tlc flags |
  |---|---|
  | naive | `--no-fuse --tile=0` |
  | fused | `--tile=0` |
  | fused+tile16 | `--tile=16` |
  | fused+tile32 | `--tile=32` |
  | fused+tile64 | `--tile=64` |
  | fused+tile32+unroll4 | `--tile=32 --unroll=4` |

- Each config at `gcc -O0` and `gcc -O3`.
- Iterations: 3 at `-O0`, 10 at `-O3` (keep total runtime reasonable).
- For each row record: median ms, speedup vs `naive -O0`, speedup vs `naive -O3`,
  intermediate buffer count (from the C header comment), correctness PASS/FAIL.
- Machine info (Windows):
  - CPU name, L2 and L3 size: run PowerShell from Python —
    `powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object Name, L2CacheSize, L3CacheSize | ConvertTo-Json"`
  - `gcc --version` first line
  - L1 cache size is not reliably reported by Windows → the user reads it from
    **Task Manager → Performance → CPU** and it is written into README by hand.
- Print a Markdown table and save it plus machine info to `bench\results.md`.
- Any FAIL → mark the row and print a warning.
- Close heavy apps (browser, games) while benchmarking; plug in the laptop charger
  (Windows throttles CPU on battery).

**Run:** `python bench\bench.py`

**Done when:** `bench\results.md` exists with real numbers and every row PASS.

**Commit:** `S12: benchmark harness and results`

---

### S13 — README (1 h)

**Files:** `README.md`.

**Contents:**
1. One-paragraph summary
2. Architecture diagram (ASCII pipeline from Section 1)
3. Language example + grammar
4. IR dumps before/after: DCE (`dead_code.tl`), CSE (`duplicate.tl`), constant folding
   (`constants.tl`), fusion (`mlp.tl`), refused fusion (`multi_user.tl`)
5. Generated C snippet: naive vs fused vs tiled kernel
6. Benchmark table copied from `bench\results.md` + machine info (incl. L1 from Task Manager)
7. Build and run instructions for Windows (Section 3 setup + PowerShell commands)
8. Design decisions: memory ownership, topological sort, CSE key, fusion legality,
   tile size vs cache size, `i,k,j` loop order
9. Next steps: SIMD (AVX2) codegen, CUDA backend (thread per output → shared-memory tiles),
   INT8 quantization pass, more ops
10. Known limitations:
    - **Name collisions in generated C.** Tensor names are emitted unchanged (Section 10), so a
      `.tl` name that is a C keyword (`int`, `for`), a local in the generated `main`
      (`dir`, `iters`, `times`, `freq`, `start`, `end`, `it`, `median`), or a `windows.h` macro (`near`,
      `far`, `small`) produces C that does not compile. None of the examples hit this. The fix
      is **name mangling**: prefix every generated symbol (e.g. `x` → `tl_x`), the same idea
      real compilers use to keep symbols unique for the linker.
    - `bench\reference.py` reads one statement per line (all examples follow this).
    - 1D/2D float32 tensors and 4 ops only.

**Done when:** someone on Windows can clone, build, run tests, and run the benchmark from the
README alone.

**Commit:** `S13: README and benchmark results`

---

## 13. Fallback plan

If **Checkpoint 1** is not reached in time:
- Skip **S8 (CSE)** and **S9 (constant folding)**.
- Keep S7 (DCE), S10 (fusion), S11 (tiling), S12 (benchmark), S13 (README).
- Leave `const` in the grammar unless it is causing bugs.

Correctness checking is never skipped.

---

## 14. Schedule

| Session | Stages |
|---|---|
| 12 Oct morning | S0 – S4 |
| 12 Oct afternoon | S5 – S6 → ⚑ Checkpoint 1 |
| 12 Oct night | S7 – S9 → ⚑ Checkpoint 2 |
| 13 Oct morning | S10 – S13 |

---

## 15. Decisions made during the build (binding — keep them)

Recorded so later stages, the README, and future sessions stay consistent.

**Toolchain**
- `tlc.exe` is linked with `-static-libstdc++ -static-libgcc` (old `libstdc++-6.dll` from
  `C:\iverilog\bin` on PATH crashed it). See Section 4.

**S1 Lexer**
- `-0.5` lexes as one `Number` token anywhere; a negative dimension is rejected later by
  shape inference (`dimension must be positive`).

**S2 Parser**
- Missing-token errors point at the **next** token (e.g. missing `]` on line 2 is reported at
  `3:1 expected ']' but found 'weight'`). This is normal parser behaviour.
- `--dump-ast` prints floats with default C++ formatting, so `2.0` shows as `2`. Stored value
  is exact.
- Dimensions must be whole numbers: `x[1.5]` → `expected an integer dimension but found
  '1.5'`. More than 9 digits is rejected the same way (would overflow `int`).

**S3 IR**
- Node ids are **never reused**; dumps show gaps after a pass removes nodes (`%0, %1, %3`).
  `%3` always means the same node in every dump.
- The IR printer's name column widens to the longest name.
- The IR printer shows a const's value: `s1 = const 0.5`. (S9's done check relies on this.)
- `lower()` takes the filename as a second parameter for error messages.
- Any valid topological order is accepted.
- The 4 lowering error files (`undefined_name`, `redefined`, `unknown_op`, `wrong_arity`)
  were added in S3, not S4.

**S4 Shapes**
- `add`/`mul` broadcast is only `A[M,N] op B[N]`; `add(b, x)` with the 1D tensor first is an
  error.
- `--dump-ir` has an "after shape inference" dump; "after lowering" shows `?` for computed
  nodes.
- `no output` is reported at `1:1` (there is no statement to point at).

**S5 Codegen**
- `c_name(node)`: input / weight / output → its own name (a `tl_forward` parameter);
  const → `c_<name>`; every other computed node → `t<id>` (static buffer).
- `intermediate buffers: N` = number of computed nodes that are not outputs.
- Float literals are printed with 9 significant digits and always end in `f` with a decimal
  point (`2.0f`, not `2f`).
- The fixed C helpers (`read_tensor`, `alloc_tensor`, `write_tensor`, `compare_doubles`) are
  stored in a C++ raw string `R"C( ... )C"`.
- Median of an even number of runs = average of the two middle values.
- The `.c` file is written with `std::ios::binary` (plain `\n` line endings).
- `-o` without a following path → usage, exit 1.

**S6 Correctness**
- Each correctness test deletes and recreates `build\test_out\<name>\` so stale `.out.bin`
  files can never cause a false PASS.
- Expected `max_err` is ~1e-5, not 0: float addition order differs between the C loops and
  NumPy's BLAS.
- The checker was verified to FAIL on a modified output (error 0.01) and a missing file (inf).

**S7 DCE**
- Mark (stack-based DFS from `graph.outputs` along `inputs`) and sweep live in
  `Graph::erase_unreachable()` (`ir.cpp`); `run_dce()` in `dce.cpp` returns the removed count.
- DCE also removes **unused weights and inputs**; they disappear from `tl_forward`'s
  parameters and their `.bin` files are not loaded. (Real frameworks often keep graph inputs
  for a stable signature — a deliberate simplification here.)
- The `--no-passes` flag was added in S7 (DCE is the first pass).

**S8 CSE**
- Output rules: see S8 in Section 12 (rename `first` to the output's name; never merge two
  outputs).

**S9 Constant folding**
- `run_constfold()` returns the folded count; it does not delete anything. The old nodes and
  the now-unused input constants are removed by the final DCE.
- The "after constant folding" dump briefly shows two nodes with the same name (the new
  const and the old node it replaced, e.g. `%7 scale = const 1` and `%4 scale = mul ...`);
  the old one disappears after final DCE.
- The new const keeps the replaced node's name, including generated names (`t5` → C array
  `c_t5`).
- If the folded node is an output, the output moves to the new const; codegen then fills
  the output parameter directly with the value.
- Const values print with default C++ formatting: S9's "value `1.0`" shows as `const 1`.
- One walk in topo order folds whole chains (e.g. `relu(add(a, b))` then `add(c, a)` →
  folded 3).

**Known limitation (not fixed on purpose)**
- Tensor names are emitted unchanged into C; names that collide with C keywords, `main`
  locals (`dir`, `iters`, `times`, `freq`, `start`, `end`, `it`, `median`), or `windows.h`
  macros break compilation. Listed in the README (S13); fix would be
  name mangling.

When a later stage makes a new decision that this file does not cover, add it here.

---

## 16. Definition of done

- [ ] `.\build.bat` builds `tlc.exe` with zero warnings
- [ ] Every example compiles; generated C compiles with zero warnings at `-O0` and `-O3`
- [ ] `python tests\run_tests.py` → all error tests and all correctness combinations PASS
- [ ] `--dump-tokens`, `--dump-ast`, `--dump-ir` all work
- [ ] DCE, CSE, constant folding, fusion each visible in IR dumps
- [ ] `bench\results.md` has real numbers, all PASS
- [ ] README complete per S13
- [ ] One commit per stage, pushed to GitHub
