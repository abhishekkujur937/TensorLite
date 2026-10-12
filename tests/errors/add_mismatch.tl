# expect: shape mismatch
input  x[64,128]
weight w[128,256]
weight b[128]
y = add(matmul(x, w), b)
output y
