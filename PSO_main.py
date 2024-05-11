#------------------------------------------------------------------------------+
# 
# Ethan Labianca-Campbell - Purdue MSAA '26 
# Particle Swarm Optimization (PSO) Framework with Python
# Last Update: May 10, 2024 
# 
#------------------------------------------------------------------------------+

#-------Libraries and Dependencies:--------------------------------------------+
from __future__ import division
import numpy as np
import random
import math
from pso.cost_functions import sphere #another function to check with
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

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
            self.best_position_i = self.position_i.copy()
            self.error_best_i = self.error_i

    #update new particle velocity 
    def update_velocity(self,group_best_position):
        #Search constants
        inertia = 0.5   # intertia weight (tendency toward last velocity)
        cog = 1      # cognitive weight (tendency toward individual best)
        soc = 0.8       # social weight (tendency toward global best)

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
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i] = bounds[i][1]
            
            #new minimum position bound if necessary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i][0] = bounds[i][0]

#---Swarm Optimization Function------------------------------------------------+    

class PSO():
    def particle_swarm(costFunction,x0,bounds,num_particles,maxiter):
        global num_dimensions #number for total 
        global xdata
        global ydata

        num_dimensions = len(x0)
        group_error_best = -1            #best error for the group
        group_best_position = -1         #best position for group

        #establish full swarm
        swarm = []
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        #optimization loop
        i = 0
        xdata = []
        ydata = []
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
            
            #make globals of positions to animate:
            for j in range(0,num_particles):
                xdata.append(swarm[j].position_i[0])
                ydata.append(swarm[j].position_i[1])

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
num_particles = 20           #number of swarm particles
maxiter = 30                 #number of steps
PSO.particle_swarm(func1,x0,bounds,num_particles,maxiter)

#---ANIMATIONS-(WIP)------------------------------------------------------------+    

#generate figure 
fig, ax = plt.subplots()
ax = plt.axes(xlim = bounds[0],ylim = bounds[1]) #change to be defined by bounds
n = num_particles
ln, = plt.plot([],[],'ro',lw=2)

#def init():
#    ax.set_xlim(-1,1)
#    ax.set_ylim(-1,1)
#    return ln,

def update(frame):
    xdata_ani = xdata[frame]
    ydata_ani = ydata[frame]
    #ln.set_data(xdata_ani,ydata_ani)
    plt.plot(xdata[frame],ydata[frame],'ro')
    return ln,

ani = FuncAnimation(fig,update,frames=len(xdata),interval=1000)
plt.show()
#ani.save('swarm_ani.gif',writer='pillow')

#---END------------------------------------------------------------------------+    

