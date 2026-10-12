# expect: redefinition of
input  x[64,128]
input  x[64,128]
weight w1[128,256]
y = matmul(x, w1)
output y
