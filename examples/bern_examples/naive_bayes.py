import probayes as pb
x = pb.RV('x')
y = pb.RV('y')
z = pb.RV('z')
zx = z / x
zy = z / y
zxy = pb.SD(zx, zy)
print(zxy.ret_roots())
print(zxy.ret_leafs())
