input  x[512,1024]
weight w1[1024,1024]
weight b1[1024]
weight w2[1024,512]
weight b2[512]
h = relu(add(matmul(x, w1), b1))
y = relu(add(matmul(h, w2), b2))
output y
