import taichi as ti
import math

ti.init(arch=ti.gpu)
vec = ti.math.vec2

window_size = 512  # Number of pixels of the window
n = 2  # Number of grains
density = 1000.0
stiffness = 10e4
shear_modulus = 1e5
mu_d = 1.0  # Friction coef
restitution_coef = 0.001
gravity = -3
dt = 0.0001  # Larger dt might lead to unstable results.
substeps = 60
grain_r = 0.03

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
    d: vec  # Displacement

gf = Grain.field(shape=(n, ))
cf = Contact.field(shape=())

@ti.kernel
def init():
    gf[0].p = vec(0.20, 0.5)
    gf[1].p = vec(0.18, 0.1)
    gf[0].r = grain_r
    gf[1].r = grain_r
    for i in gf:
        gf[i].m = density * math.pi * gf[i].r**2
    gf[0].v = vec(0.0, 0.0)
    gf[1].v = vec(0.0, 3.5)
    cf[None].d = vec(0.0, 0.0)

@ti.kernel
def update():
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a


@ti.kernel
def apply_bc():
    bounce_coef = 0.3  # Velocity damping
    for i in gf:
        x = gf[i].p[0]
        y = gf[i].p[1]

        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r
            gf[i].v[1] *= -bounce_coef

        elif y + gf[i].r > 1.0:
            gf[i].p[1] = 1.0 - gf[i].r
            gf[i].v[1] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].p[0] = gf[i].r
            gf[i].v[0] *= -bounce_coef

        elif x + gf[i].r > 1.0:
            gf[i].p[0] = 1.0 - gf[i].r
            gf[i].v[0] *= -bounce_coef


@ti.func
def resolve(i, j):
    rel_pos = gf[j].p - gf[i].p
    dist = rel_pos.norm()  # dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        # Normal force
        normal = rel_pos / dist
        tangent = vec(-normal[1], normal[0]) / vec(-normal[1], normal[0]).norm()
        f1 = normal * delta * stiffness
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        f2 = C * V * normal
        gf[i].f += f2 - f1
        gf[j].f -= f2 - f1
        # Frictional force
        rel_vel = gf[j].v - gf[i].v - normal * ti.math.dot(normal, (gf[j].v - gf[i].v))
        cf[None].d += rel_vel * dt
        fr_d = - mu_d * f1.norm() * rel_vel / rel_vel.norm()  # Dynamic friction
        fr = cf[None].d * shear_modulus
        if fr.norm() < fr_d.norm():
            gf[j].f += fr
            gf[i].f -= fr
        else:
            gf[j].f += fr_d
            gf[i].f -= fr_d            
            


@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.    
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)

init()
gui = ti.GUI('Taichi DEM', (window_size, window_size))
step = 0

while gui.running:
    gui.line([0.5,0], [0.5,1], radius=1, color=0x068587)
    pos = gf.p.to_numpy()
    r = gf.r.to_numpy() * window_size
    for s in range(substeps):
        if pos[0][1] - grain_r > 0.1 * grain_r:        
            update()
            apply_bc()
            contact(gf)
    if pos[0][1] - grain_r <= 0.1 * grain_r:
        print(f'>>> Simulation finished, final position {pos[0]}')
        gui.running = False
    gui.circles(pos, radius=r)
    gui.show()
    step += 1
