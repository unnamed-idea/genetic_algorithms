import ga
import numpy as np
from numpy import abs as abs
from numpy import sin as sin


def custom_fit(i):
    x = i.chromosome_[0]
    y = i.chromosome_[1]
    result = abs(80 - (x * y)) #+ np.sum(i.geneit) * -0.0

    #result = abs(838 + x * sin(abs(x) ** 0.5) + y * sin(abs(y) ** 0.5))
    return result

dimension = 2
pop = 80
low_lim = 1
high_lim = 45
iteration = 400

mypool = ga.pool(dimension, pop, low_lim, high_lim, custom_fit)
mypool.createpool()

best, fit = mypool.iterate(iteration)

print(best, fit)

mypool.show_hist()


