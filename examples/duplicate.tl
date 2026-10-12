input  x[64,128]
weight w1[128,256]
a = matmul(x, w1)
b = matmul(x, w1)
y = add(a, b)
output y
