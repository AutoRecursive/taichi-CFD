from matplotlib import pyplot, cm
import numpy as np
import taichi as ti
from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots

nx = 81
ny = 81
nt = 100
c = 1

dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)                            


u = ti.field(dtype=float, shape=(ny, nx))
un = ti.field(dtype=float, shape=(ny, nx))


@ti.kernel
def init():
    for i, j in ti.grouped(u):
        if i >= int(.5 / dx) and i < int(1 / dx + 1) and j >= int(.5 / dy) and j < int(1 / dy + 1):
            u[i, j] = 2
        else:
            u[i, j] = 1
        un [i, j] = u[i, j]

@ti.kernel
def propagate():
    for i, j in ti.grouped(u):
        un[i, j] = u[i, j]

    for j, i in ti.ndrange((1, ny), (1, nx)): 
        if j == 0 or j == ny - 1 or i == 0 or i == nx - 1:
            u[j, i] = 1
        else:
            u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) - (c * dt / dy * (un[j, i] - un[j - 1, i])))

def plot(ax, pause_time=0.005):
    ax.cla()
    surf = ax.plot_surface(X, Y, u.to_numpy()[:], cmap=cm.viridis)
    pyplot.pause(pause_time)

# gui = ti.GUI("2D Convection", res=(ny, nx))
def main():
    t = 0
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    fig.show()
    pyplot.ion()

    while True:
        if t == 0 or t > nt:
            init()
            t = 0
        plot(ax)
        # gui.set_image(u)
        # gui.show()
        propagate()
        t += 1

if __name__ == '__main__':
    main()
