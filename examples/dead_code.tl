input  x[64,128]
weight w1[128,256]
weight w2[128,64]
h = matmul(x, w1)
unused = matmul(x, w2)
output h
