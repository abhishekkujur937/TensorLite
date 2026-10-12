# expect: expects
input  x[64,128]
weight w[128,256]
y = relu(x, w)
output y
