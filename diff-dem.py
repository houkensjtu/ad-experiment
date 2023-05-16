import taichi as ti
import math
import numpy as np

ti.init(arch=ti.cpu, debug=True)
vec = ti.math.vec2

Ngrain = 2  # Number of grains
Ntime = 10000  # Number of grains
window_size = 512  # Number of pixels of the window
density = 1000.0
stiffness = 10e4
bounce_coef = 0.3  # Velocity damping
mu_d = 1.0  # Friction coef
restitution_coef = 0.001
gravity = -9.8
dt = 1e-4  # Larger dt might lead to unstable results.
substeps = 10000
grain_r = 0.05

@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force

@ti.dataclass
class Contact:
    d: vec  # Displacement (for frictional force simulation)


'''
Note that gf is 2-D in order to record all positions at all timesteps.
This is a strategy to prevent violation of GDAR (data mutation) during the forward pass.
'''
shear_modulus = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
gf = Grain.field(shape= (Ntime, Ngrain), needs_grad=True)  # gf records all positions at all timesteps
cf = Contact.field(shape=(Ntime), needs_grad=True)  # Contact field to record displacement

@ti.kernel
def init():
    for i, j in gf:
        gf[i, j].r = grain_r
        gf[i, j].m = density * math.pi * gf[i, j].r**2
    # Initial position
    gf[0, 0].p = vec(0.20, 0.5)
    gf[0, 1].p = vec(0.18, 0.1)
    gf[0, 0].r = grain_r
    gf[0, 1].r = grain_r
    gf[0, 0].v = vec(0.0, 0.0)
    gf[0, 1].v = vec(0.0, 5.5)
    cf[0].d = vec(0.0, 0.0)

@ti.kernel
def update(t:ti.i32):
    for i in range(Ngrain):
        a = gf[t-1, i].f / gf[t-1, i].m
        # Caution: in the no-time version, += was used to update positions
        # Need to modify to = + in this version
        gf[t, i].v = gf[t-1, i].v + (gf[t-1, i].a + a) * dt / 2.0
        gf[t, i].p = gf[t-1, i].p + gf[t-1, i].v * dt + 0.5 * a * dt**2
        gf[t, i].a = a
        
@ti.kernel
def apply_bc(t:ti.i32):
    ''' Caution
    Mutating the current status cause violation of GDAR.
    However, in the simplest 2-ball system, this should not be an issue.
    Because the motion of the ball is not affected by the bouncing.
    '''
    for i in range(Ngrain):
        x = gf[t, i].p[0]
        y = gf[t, i].p[1]

        if y - gf[t, i].r < 0:
            gf[t, i].p[1] = gf[t, i].r
            gf[t, i].v[1] *= -bounce_coef

        elif y + gf[t, i].r > 1.0:
            gf[t, i].p[1] = 1.0 - gf[t, i].r
            gf[t, i].v[1] *= -bounce_coef

        if x - gf[t, i].r < 0:
            gf[t, i].p[0] = gf[t, i].r
            gf[t, i].v[0] *= -bounce_coef

        elif x + gf[t, i].r > 1.0:
            gf[t, i].p[0] = 1.0 - gf[t, i].r
            gf[t, i].v[0] *= -bounce_coef

@ti.kernel
def contact(t:ti.i32):
    '''
    Handle the collision between grains.
    '''    
    for i in range(Ngrain):
        gf[t, i].f = vec(0., gravity * gf[t, i].m)  # Apply gravity.
    for i in range(Ngrain):
        for j in range(i + 1, Ngrain):
            resolve(t, i, j)

@ti.func
def resolve(t, i, j):
    rel_pos = gf[t, j].p - gf[t, i].p
    dist = rel_pos.norm()
    delta = -dist + gf[t, i].r + gf[t, j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        # Normal force
        normal = rel_pos / dist
        tangent = vec(-normal[1], normal[0]) / vec(-normal[1], normal[0]).norm()
        f1 = normal * delta * stiffness
        # Damping force
        M = (gf[t, i].m * gf[t, j].m) / (gf[t, i].m + gf[t, j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[t, j].v - gf[t, i].v) * normal
        f2 = C * V * normal
        gf[t, i].f += f2 - f1
        gf[t, j].f -= f2 - f1
        # Frictional force
        rel_vel = gf[t, j].v - gf[t, i].v - normal * ti.math.dot(normal, (gf[t, j].v - gf[t, i].v))
        cf[t].d = cf[t-1].d + rel_vel * dt  # Integrating displacement over time
        fr_d = - mu_d * f1.norm() * rel_vel / rel_vel.norm()  # Dynamic friction
        fr = cf[t].d * shear_modulus[None]
        if fr.norm() < fr_d.norm():
            gf[t, j].f += fr       
            gf[t, i].f -= fr
        else:
            gf[t, j].f += fr_d
            gf[t, i].f -= fr_d

@ti.kernel
def compute_loss(t:ti.i32):
    # Loss defined as the target ball's x distance to the center
    loss[None] = ti.sqrt((gf[t, 0].p[0] - 0.5) ** 2)

def substep(t):
    # Advance the time 
    update(t)
    apply_bc(t)
    contact(t)

def forward():
    # Forward pass
    for istep in range(1,substeps):
        substep(istep)
        if gf[istep, 0].p[1] - grain_r < 0.005:  # Stop the simulation if the target ball hit the floor
            break
        if istep % 100 == 0:  # Plot the result to the GUI every 100 steps
            pos = np.array([[gf[istep, 0].p[0], gf[istep, 0].p[1]], [gf[istep, 1].p[0], gf[istep, 1].p[1]]])
            r = np.array([gf[istep, 0].r, gf[istep, 1].r]) * window_size
            gui.line([0.5,0], [0.5,1], radius=1, color=0x068587)  # Draw the center line
            gui.circles(pos, radius=r)
            gui.show()
    compute_loss(istep)

def optimize(learning_rate):
    shear_modulus[None] = 1e3
    for iter in range(500):
        init()
        # with ti.ad.Tape(loss, validation=True):  # Use GDAR checker
        with ti.ad.Tape(loss):
            forward()
        print(f'>>> shear_modulus={shear_modulus[None]:5.3e}, shearmodulus.grad={shear_modulus.grad[None]:5.3e},\
        target ball to center distance={loss[None]:5.3e}')
        shear_modulus[None] -= learning_rate * shear_modulus.grad[None]
        if loss[None] < 1e-4:
            print(f'>>> Optimization finished. Final loss={loss[None]}')
            break
    
gui = ti.GUI('Taichi DEM', (window_size, window_size))    
optimize(learning_rate=7e6)  # Learning rate should be reasonable to ensure a smooth learning process
