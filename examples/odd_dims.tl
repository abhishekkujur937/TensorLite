input  x[63,129]
weight w1[129,70]
weight b1[70]
y = relu(add(matmul(x, w1), b1))
output y
