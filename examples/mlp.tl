input  x[64,128]
weight w1[128,256]
weight b1[256]
weight w2[256,10]
h = relu(add(matmul(x, w1), b1))
y = matmul(h, w2)
output y
