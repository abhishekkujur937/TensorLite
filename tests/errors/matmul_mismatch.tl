# expect: matmul inner dims mismatch
input  x[64,128]
weight w[256,10]
y = matmul(x, w)
output y
