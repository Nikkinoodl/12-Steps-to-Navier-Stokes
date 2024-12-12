import numpy                     #loading our favorite library
from matplotlib import pyplot    #and the useful plotting library
from timeit import timeit
import timeit

nx = 100
dx = 2 / (nx - 1)
nt = 100    #the number of timesteps we want to calculate
nu = 0.3   #the value of viscosity
sigma = .2 #sigma is a parameter, we'll learn more about it later
dt = sigma * dx**2 / nu #dt is defined using sigma ... more later!

#Set up u array and ICs
u = numpy.ones(nx)      #a numpy array with nx elements all equal to 1.
u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s


start_time = timeit.default_timer()

for n in range(nt):  #iterate through time
    un = u.copy() ##copy the existing values of u into un
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])
     

time_1 = timeit.default_timer() - start_time
print('Python took', time_1)

pyplot.plot(numpy.linspace(0, 2, nx), u)

#Reset u
u = numpy.ones(nx)      #a numpy array with nx elements all equal to 1.
u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s

start_time = timeit.default_timer()

for n in range(nt):  #iterate through time
    un = u.copy() ##copy the existing values of u into un

    #This is my first attempt at slicing through an array with Numpy
    u[1:-1] = un[1:-1] + nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[0:-2])

time_2 = timeit.default_timer() - start_time
print('Numpy took', time_2)


pyplot.plot(numpy.linspace(0, 2, nx), u)
pyplot.show()