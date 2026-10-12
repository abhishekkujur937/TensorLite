input  x[64,128]
weight w1[128,256]
const  s1[256] = 0.5
const  s2[256] = 2.0
scale = mul(s1, s2)
h = matmul(x, w1)
y = mul(h, scale)
output y
