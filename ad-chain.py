import taichi as ti

ti.init(arch=ti.cpu)

scalar = lambda : ti.field(dtype=ti.f32, shape=(), needs_grad=True)

loss = scalar()
x = scalar()

@ti.kernel
def eval_x():
    x[None] = 1.0

@ti.kernel
def compute_1():
    loss[None] = (x[None])

@ti.kernel
def compute_2():
    loss[None] = 2 * (x[None])

@ti.kernel
def compute_3():
    loss[None] = 4 * (x[None])

eval_x()
with ti.ad.Tape(loss):
    compute_1()
    compute_2()
    compute_3()
    
# Expect: loss = 4, x.grad = 4, Result: loss = 4, x.grad = 7
print(f'loss = {loss[None]}, x.grad = {x.grad[None]}')

