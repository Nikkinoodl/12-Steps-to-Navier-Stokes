import numpy
from matplotlib import pyplot
from timeit import timeit
import timeit


nx = 41
dx = 2 / (nx - 1)
nt = 25
dt = 0.025
c = 1

u = numpy.ones(nx)

u[int(.5 / dx):int(1 / dx + 1)] = 2

pyplot.plot(numpy.linspace(0, 2, nx), u);

un = numpy.ones(nx) #initialize a temporary array

for n in range(nt):  #loop for values of n from 0 to nt, so it will run nt times
    un = u.copy() ##copy the existing values of u into un
    #for i in range(1, nx): ## you can try commenting this line and...
    #for i in range(nx): ## ... uncommenting this line and see what happens!
        #u[i] = un[i] - c * (un[i] - un[i-1]) * dt / dx

    #the above but written in to use Numpy
    u[1:-1] = un[1:-1] - c * (un[1:-1] - un[0:-2]) * dt /dx


pyplot.plot(numpy.linspace(0, 2, nx), u);
pyplot.show()