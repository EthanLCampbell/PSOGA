#------------------------------------------------------------------------------+
# 
# Ethan Labianca-Campbell - Purdue MSAA '26 
# Particle Swarm Optimization (PSO) Framework with Python
# May, 2024 
# 
#------------------------------------------------------------------------------+

#-------Libraries and Dependencies:--------------------------------------------+
import numpy as np
import random
import math
from __future__ import division

#---COST FUNCTION -------------------------------------------------------------+

#Function to optimize (minimize)
def func1(x):
    total = 0
    for i in range(len(x)):
        total+= x[i]**2
    return total

#---Particle Function Definition-----------------------------------------------+

class Particle:
    # Define Particle Variables
    def __init__(self,x0):
        # individual particle variables 
        self.position_i = []        # particle position
        self.velocity_i = []        # particle velocity
        self.best_position_i = []   # individual best position
        self.error_i = []           # individual error
        self.error_best_i = []      # individual best error
        self.best_fitness = float('inf') #fitness function

    # particle update states based on random velocity input
    for i in range(0,num_dimensions):
        self.velocity_i.append(random.uniform(-1,1))
        self.position_i.append(x0[i])

    # current fitness evaluation
    def evaluate(self,costFunction):
        self.error_i = costFunction(self.position_i)
        #check if particle position is individual best
        if self.error_i < self.error_best_i or self.error_best_i == -1:
            self.best_position_i = self.position_i
            self.error_best_i = self.error_i

    #update new particle velocity 
    def update_velocity(self,group_best_position):
        #Search constants
        inrta = 0.5 # intertia weight (tendency toward last velocity)
        cog = 1       # cognative weight (tendency toward individual best)
        soc = 1       # social weight (tendency toward global best)

        for i in range (0,num_dimensions):
            r1 = random.random() #random permutation on cognitive
            r2 = random.random() #random permutation on social

            vel_cog = cog*r1*(self.best_position_i[i] - self.position_i[i])
            vel_soc = soc*r2*(group_best_position - self.position_i[i])
            self.velocity_i[i] = inrta*self.velocityIi[i] + vel_cog + vel_soc

    # update position based off new vel updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            #new maximum position bound if necessary
            if self.position_i[i]>bounds[i][0]:
                self.position_i[i] = bounds[i][0]
            
            #new minimum position bound if necessary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i][0]

#---Swarm Optimization Function------------------------------------------------+    

class PSO():
    def __init__(self,costFunction,x0,bounds,num_particles,maxiter):
        global num_dimensions #number for total 

        num_dimensions = len(x0)
        group_error_best = -1            #best error for the group
        group_best_position = []         #best position for group

        #establish full swarm
        swarm = []
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        #optimization loop
        i = 0
        while i<maxiter:
            #print i, group_error_best

            #cycle through particles in swarm and evaluate fitness
            for j in range():
                swarm[j].evaluate(costFunction)

                #determine if particle_j is global best
                if swarm[j].error_i < group_error_best or group_error_best == -1:
                    group_position_best = list(swarm[j].position_i)
                    group_error_best = float(swarm[j].error_i)
            
            #through swarm and update vel and pos
            for j in range(0,num_particles):
                swarm[j].update_velocity(group_best_position)
                swarm[j].update_position(bounds)

