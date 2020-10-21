import numpy as np
from matplotlib import pyplot      #here we load matplotlib
import time, sys
import taichi as ti
from time import sleep
ti.init(arch=ti.cpu)

nx = 41
dx = 2 / (nx-1)
nt = 25
dt = .025
c = 1

u = ti.field(dtype=float, shape=(nx))
un = ti.field(dtype=float, shape=(nx))

@ti.pyfunc
def init():
    for i in range(nx):
        if i >= int(.5 / dx) and i < int(1 / dx + 1):
            u[i] = 2
        else:
            u[i] = 1
        un[i] = u[i]

@ti.kernel
def propagate():
    for i in range(1, nx):
        un[i] = u[i]
    for i in range(1, nx):
        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

# gui = ti.GUI("CFD", res=(nx))
def main():
    t = 0
    pyplot.ion()
    pyplot.figure()
    while True:
        pyplot.plot(np.linspace(0, 2, nx), u.to_numpy())
        pyplot.show()
        pyplot.clf()
        if t == 0 or t > nt:
            init()
            t = 0
        propagate()

        print(u)
        t += 1
        pyplot.plot(np.linspace(0, 2, nx), u.to_numpy())
        pyplot.pause(0.1)

if __name__ == '__main__':
    main()

