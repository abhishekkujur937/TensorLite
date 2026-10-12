# expect: undefined name
input  x[64,128]
weight w1[128,256]
weight w2[256,10]
h = matmul(x, w1)
y = matmul(h, w3)
output y
