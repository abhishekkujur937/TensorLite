# expect: no output
input  x[64,128]
weight w[128,256]
y = matmul(x, w)
