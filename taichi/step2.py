from matplotlib import pyplot as plt
import numpy as np
import taichi as ti

nx = 101
dx = 2 / (nx - 1)
nt = 20
sigma = .5
dt = sigma * dx


u = ti.field(dtype=float, shape=(nx))
un = ti.field(dtype=float, shape=(nx))

@ti.pyfunc
def init():
    for i in range(nx):
        if i >= int(.5 / dx) and i < int(1 / dx + 1):
            u[i] = 2
        else:
            u[i] = 1
        un [i] = u[i]

@ti.kernel
def propagate():
    for i in range(1, nx):
        un[i] = u[i]

    for i in range(1, nx):
        u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1])

    #    print(u[i])

def main():
    t = 0
    plt.show()
    plt.ion()

    while True:
        if t == 0 or t > nt:
            init()
            t = 0
        plt.plot(np.linspace(0, 2, nx), u.to_numpy())
        plt.pause(0.1)
        plt.clf()
        propagate()
        t += 1

if __name__ == '__main__':
    main()
