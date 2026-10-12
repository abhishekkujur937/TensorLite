input  x[64,128]
weight w1[128,256]
weight b1[256]
m = matmul(x, w1)
h = relu(add(m, b1))
y = add(h, m)
output y
