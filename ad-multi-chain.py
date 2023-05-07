import taichi as ti

ti.init(arch=ti.cpu)

u = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
v = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
w = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def init():
    u[None] = 0.0
    v[None] = 0.0
    w[None] = 0.0
    x[None] = 1.0    

@ti.kernel
def compute_1():
    u[None] = x[None] * 3.0

@ti.kernel
def compute_2():
    v[None] = x[None] + 1.0

@ti.kernel
def compute_3():
    w[None] = v[None] + u[None]

init()

with ti.ad.Tape(loss=w):
    compute_1()  # u = 3 * x
    compute_2()  # v = x + 1
    compute_3()  # w = u + v = 4 * x + 1

# Expect: x = 1, x.grad = 4, u.grad = 1, v.grad = 1
# Result: x = 1, x.grad = 4, u.grad = 0, v.grad = 0
print(f'x = {x[None]:4.2e}, x.grad = {x.grad[None]}, u.grad = {u.grad[None]}, v.grad={v.grad[None]}')    
