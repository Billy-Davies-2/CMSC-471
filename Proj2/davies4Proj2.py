# Coded By: William Davies
# Date: 03/19/2016
# Class CMSC 471
# Project: AI Project 2
# imports
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import random
from random import randint

# define objective function
def f(x, y):
    obj = np.sin(x**2 + 3*y**2)/(0.1 * np.sqrt(x**2 + y**2)**2)+(x**2 + 5*y**2)*np.exp(1-np.sqrt(x**2 + y**2)**2)/2
    return obj

def hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax):
    # start at random spot
    pathX = []
    pathY = []
    stepX = round(random.uniform(xmin,xmax),3)
    stepY = round(random.uniform(ymin,ymax),3)
    pathX.append(stepX)
    pathY.append(stepY)
    z = function_to_optimize
    # take any step that goes downhill
    currentPoint = z(stepX,stepY)
    while True:
        # can move x to left
        nextStep = currentPoint
        if (stepX - step_size) >= xmin:
            x1 = stepX - step_size
            if (z(x1,stepY) < nextStep):
                stepX = x1
                nextStep = z(x1,stepY)
        # can move x to right
        if (stepX + step_size) <= xmax:
            x2 = stepX + step_size
            if(z(x2,stepY) < nextStep):
                stepX = x2
                nextStep = z(x2,stepY)
        # can move y to left
        if (stepY - step_size) >= ymin:
            y1 = stepX - step_size
            if(z(stepX,y1) < nextStep):
                stepY = y1
                nextStep = z(stepX,y1)
        # can move y to right
        if (stepY + step_size) <= ymax:
            y2 = stepY + step_size
            if(z(stepX,y2) < nextStep):
                stepY = y2
                nextStep = z(stepX,y2)
        # x and y to left
        if ((stepX - step_size) >= xmin) and ((stepY - step_size) >= ymin):
            x3 = stepX - step_size
            y3 = stepY - step_size
            if (z(x3,y3) < nextStep):
                stepX = x3
                stepY = y3
                nextStep = z(x3,y3)
        # x and y to right
        if ((stepX + step_size) <= xmax) and ((stepY + step_size) <= ymin):
            x4 = stepX + step_size
            y4 = stepY + step_size
            if (z(x4,y4) < nextStep):
                stepX = x4
                stepY = y4
                nextStep = z(x4,y4)
        # x left y right
        if ((stepX - step_size) >= xmin) and ((stepY + step_size) <= ymax):
            x5 = stepX - step_size
            y5 = stepY + step_size
            if (z(x5,y5) < nextStep):
                stepX = x5
                stepY = y5
                nextStep = z(x5,y5)
        # x right y left
        if ((stepX + step_size) <= xmin) and ((stepY - step_size) >= ymin):
            x6 = stepX + step_size
            y6 = stepY - step_size
            if (z(x6,y6) < nextStep):
                stepX = x6
                stepY = y6
                nextStep = z(x6,y6)
        # get the smallest of all neighbor moves
        if (currentPoint > nextStep):
            # moving downhill
            currentPoint = nextStep
            pathX.append(stepX)
            pathY.append(stepY)
        else:
            break
    return (pathX, pathY)

# hill climb with random restart is just hill climb repeated
# however many times needed. If it finds the best min, it updates
# otherwise it just keeps checking.
def hill_climb_random_restart(function_to_optimize, step_size, num_restarts, xmin, xmax, ymin, ymax):
    currentMin = hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax)
    pathX = currentMin[0]
    pathY = currentMin[1]
    for starts in range(0,num_restarts-1):
        nextMin = hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax)
        if (nextMin < currentMin):
            currentMin = nextMin
            pathX = nextMin[0]
            pathY = nextMin[1]
    return (pathX, pathY)

# simulated annealing will pick a random neighbor and if the probability
# is greater than the random number between 0 and 1 it accepts it even
# if it's a bad move. It always accepts better moves, so
# as the temperature goes down, it becomes a hillclimbing algorithm.
def simulated_annealing(function_to_optimize, step_size, max_temp, xmin, xmax, ymin, ymax):
    z = function_to_optimize
    # intialize x and y
    pathX = []
    pathY = []
    stepX = round(random.uniform(xmin,xmax),3)
    stepY = round(random.uniform(ymin,ymax),3)
    pathX.append(stepX)
    pathY.append(stepY)
    # initial state
    s0 = z(stepX, stepY)
    eAvg = 0.0
    counter = 0.0
    for temp in range(max_temp, 0, -1):
        accept = False
        # pick a random neighbor
        xnew, ynew = random_neighbor(stepX, stepY, step_size, xmin, xmax, ymin, ymax)
        snew = z(xnew, ynew)
        if (temp == max_temp): eAvg = snew
        if (snew > s0):
            # if P(E(scurr), E(snew),T) >= random(0,1)
            prob = Boltzman_factor(s0, snew, eAvg, temp)
            if (random.random() < prob):
                # accept the worse solution
                accept = True
            else:
                accept = False
        else:
            # accept a better solution anyways
            accept = True
        if (accept == True):
            stepX = xnew
            stepY = ynew
            pathX.append(xnew)
            pathY.append(ynew)
            s0 = snew
            counter = counter + 1.0
            eAvg = (eAvg * counter + snew) / counter
    # return scurr
    return(pathX, pathY)

# picks a random neighbor state and returns the location.
def random_neighbor(x, y, step_size, xmin, xmax, ymin, ymax):
    movement = randint(1,4)
    if movement == 1:
        nextX = ( x + step_size ) % xmax
        nextY = ( y + step_size ) % ymax
    elif movement == 2:
        nextX = (x + step_size) % xmax
        nextY = (y - step_size) % ymin
    elif movement == 3:
        nextX = (x - step_size) % xmin
        nextY = (y + step_size) % ymax
    elif movement == 4:
        nextX = (x - step_size) % xmin
        nextY = (y - step_size) % ymin
    else:
        nextX = x
        nextY = y
    return(nextX, nextY)

# determines the likelihood of accepting a bad state.
def Boltzman_factor(expectedOld, expectedNew, eAvg, temperature):
    # should be the average delta E in the denominator
    probability = np.exp(-(expectedOld - expectedNew)/((eAvg) * temperature))
    return probability

# main function that generates the graphs, calls the search algorithms and plots the
# results.
def main():
    # algorithm section
    hillMin = hill_climb(f, 0.01, -2.5, 2.5, -2.5, 2.5)
    hillStart = hill_climb_random_restart(f, 0.01, 5, -2.5, 2.5, -2.5, 2.5)
    simAnneal = simulated_annealing(f, 0.01, 2500, -2.5, 2.5, -2.5, 2.5)
    # plot the results of the algorithms
    # plot the hillclimb
    hillMinX = hillMin[0]
    hillMinY = hillMin[1]
    hillMinZ = []
    for i in range(0, len(hillMinX)):
        hillMinZ.append(f(hillMinX[i], hillMinY[i]))
    # plot hill climb random restart
    hillStartX = hillStart[0]
    hillStartY = hillStart[1]
    hillStartZ = []
    for i in range(0, len(hillStartX)):
        hillStartZ.append(f(hillStartX[i], hillStartY[i]))
    # plot the simulated annealing
    simAnnealX = simAnneal[0]
    simAnnealY = simAnneal[1]
    simAnnealZ = []
    for i in range(0, len(simAnnealX)):
        simAnnealZ.append(f(simAnnealX[i], simAnnealY[i]))
    # graphing section
    # establish the range between x and y
    x = np.arange(-2.5, 2.6, .1)
    y = np.arange(-2.5, 2.6, .1)
    X, Y = np.meshgrid(x,y)
    xi = np.linspace(-2.5, 2.5, 20)
    yi = np.linspace(-2.5, 2.5, 20)
    z = f(X, Y)
    # hill climb graph
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot(hillMinX, hillMinY, hillMinZ, 'm')
    # hill climb random restart graph
    fig1 = plt.figure(2)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.plot(hillStartX, hillStartY, hillStartZ, 'm')
    # simulated annealing graph.
    fig2 = plt.figure(3)
    ax2 = fig2.add_subplot(111, projection='3d')
    # didn't plot the surface since it was hard to see all the data points
    #ax2.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.scatter(simAnnealX, simAnnealY, simAnnealZ, c='m')
    # display the three graphs
    plt.show()
main()