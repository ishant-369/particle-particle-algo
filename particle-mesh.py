import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
Nx, Ny = 50, 50
Lx, Ly=10.0, 10.0
dx = Lx / Nx
dy = Ly / Ny
N = 200
dt = 0.005
steps = 500
epsilon0 = 1.0
Bz = 100
positions = np.random.rand(N, 2)
positions[:,0] *= Lx
positions[:,1] *= Ly
velocities = 0.2*np.random.randn(N,2)
charges = np.random.choice([-1e-4,1e-4], N)
mass = np.ones(N)

rho = np.zeros((Nx,Ny))
phi = np.zeros((Nx,Ny))
Ex = np.zeros((Nx,Ny))
Ey = np.zeros((Nx,Ny))
fig, ax = plt.subplots()
scat = ax.scatter(
    positions[:,0],
    positions[:,1],
    c=charges,
    cmap='bwr'
)
ax.set_xlim(0,Lx)
ax.set_ylim(0,Ly)
# now here we are defining an function to solve our poisson equation
def poisson_solver(rho):
    phi = np.zeros_like(rho)
    for _ in range(100):
        phi[1:-1,1:-1] = 0.25*(
            phi[2:,1:-1]
            + phi[:-2,1:-1]
            + phi[1:-1,2:]
            + phi[1:-1,:-2]
            + dx*dy*rho[1:-1,1:-1]
        )
    return phi
def update(frame):
    global positions
    global velocities
    rho.fill(0)
    for p in range(N):
        i = int(positions[p,0]/dx)
        j = int(positions[p,1]/dy)
        i = np.clip(i,0,Nx-1)
        j = np.clip(j,0,Ny-1)
        rho[i,j] += charges[p]
    phi[:] = poisson_solver(rho)

    Ex[1:-1,:] = -(phi[2:,:]-phi[:-2,:])/(2*dx)
    Ey[:,1:-1] = -(phi[:,2:]-phi[:,:-2])/(2*dy)
    for p in range(N):
        i = int(positions[p,0]/dx)
        j = int(positions[p,1]/dy)
        i = np.clip(i,0,Nx-1)
        j = np.clip(j,0,Ny-1)
        ex = Ex[i,j]
        ey = Ey[i,j]
        vx = velocities[p,0]
        vy = velocities[p,1]
        q = charges[p]
        Fx_e = q*ex
        Fy_e = q*ey
        Fx_b = q*vy*Bz
        Fy_b = -q*vx*Bz
        Fx = Fx_e + Fx_b
        Fy = Fy_e + Fy_b
        velocities[p,0] += Fx*dt
        velocities[p,1] += Fy*dt
        positions[p] += velocities[p]
        if positions[p,0] < 0:
            positions[p,0] = 0
            velocities[p,0] *= -1
        if positions[p,0] > Lx:
            positions[p,0] = Lx
            velocities[p,0] *= -1
        if positions[p,1] < 0:
            positions[p,1] = 0
            velocities[p,1] *= -1
        if positions[p,1] > Ly:
            positions[p,1] = Ly
            velocities[p,1] *= -1
    scat.set_offsets(positions)
    return scat,
ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=20,
    blit=True
)
plt.show()