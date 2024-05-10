#------------------------------------------------------------------------------+
# 
# Ethan Labianca-Campbell - Purdue MSAA '26 
# Particle Swarm Optimization (PSO) Framework with Python
# May, 2024 
# 
#------------------------------------------------------------------------------+

#-------Libraries and Dependencies:--------------------------------------------+
from __future__ import division
import numpy as np
import random
import math
from pso.cost_functions import sphere #another function to check with

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
        self.error_i = -1           # individual error
        self.error_best_i = -1      # individual best error
        #self.best_fitness = float('inf') #fitness function

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
        inertia = 0.5   # intertia weight (tendency toward last velocity)
        cog = 1       # cognitive weight (tendency toward individual best)
        soc = 1       # social weight (tendency toward global best)

        for i in range (0,num_dimensions):
            r1 = random.random() #random permutation on cognitive
            r2 = random.random() #random permutation on social

            vel_cog = cog*r1*(self.best_position_i[i] - self.position_i[i])
            vel_soc = soc*r2*(group_best_position - self.position_i[i])
            self.velocity_i[i] = inertia*self.velocity_i[i] + vel_cog + vel_soc

    # update position based off new vel updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            #new maximum position bound if necessary
            if self.position_i[i]>bounds[i][0]:
                self.position_i[i] = bounds[i][1]
            
            #new minimum position bound if necessary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i][0] = bounds[i][0]

#---Swarm Optimization Function------------------------------------------------+    

class PSO():
    def particle_swarm(costFunction,x0,bounds,num_particles,maxiter):
        global num_dimensions #number for total 

        num_dimensions = len(x0)
        group_error_best = -1            #best error for the group
        group_best_position = -1         #best position for group

        #establish full swarm
        swarm = []
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        #optimization loop
        i = 0
        while i<maxiter:
            #print i, group_error_best
            print(f'iter: {i:>4d}, best solution: {group_error_best:10.6f}')
            #cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunction)

                #determine if particle_j is global best
                if swarm[j].error_i < group_error_best or group_error_best == -1:
                    group_position_best = list(swarm[j].position_i)
                    group_error_best = float(swarm[j].error_i)
            
            #through swarm and update vel and pos
            for j in range(0,num_particles):
                swarm[j].update_velocity(group_best_position)
                swarm[j].update_position(bounds)
            
            #loop again
            i+=1
        #print final results
        print('Final:')
        print(f'   > {group_position_best}')
        print(f'   > {group_error_best}\n')

        return group_error_best, group_position_best
#---RUN------------------------------------------------------------------------+    

x0 = [5,5]                   #starting location [x1,x2,x3,....]
bounds = [(-10,10),(-10,10)] #bounds for search [(x1min,x1max),(x2min,x2max),..]
PSO.particle_swarm(sphere,x0,bounds,num_particles=20,maxiter=11)

#---END------------------------------------------------------------------------+    
