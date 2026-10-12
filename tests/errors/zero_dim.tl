# expect: dimension must be positive
input  x[0,128]
weight w[128,256]
y = matmul(x, w)
output y
