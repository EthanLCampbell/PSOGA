#------------------------------------------------------------------------------+
# 
# Ethan Labianca-Campbell - Purdue MSAA '26 
# Particle Swarm Optimization (PSO) Framework with Python
# Last Update: May 10, 2024 
# 
#------------------------------------------------------------------------------+

#-------Libraries and Dependencies:--------------------------------------------+
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

#particle swarm optimization function
def pso(func, bounds, swarm_size=10, inertia=0.5, pv=1, gv=1, 
        max_vnorm=10, num_iters=100, verbose=False, func_name=None):
    # """Particle Swarm Optimization (PSO)
    # Input:
    #     func - cost function
    #     bounds - list, xy dimension bounds
    #     swarm_size - int, the number of fish
    #     inertia: float, tendency to not change velocity
    #     pv: float, tendency to personal best 
    #     gv: float, tendency to global best
    #     max_vnorm: max velocity norm
    #     num_iters: int, the number of iterations
    #     verbose: boolean, whether to print results or not
    #     func_name: the name of object function to optimize

    # Returns
    #     history: history of particles and global bests
    # """
    bounds = np.array(bounds)
    assert np.all(bounds[:,0] < bounds[:,1]) # each boundaries have to satisfy this condition
    dim = len(bounds)
    X = np.random.rand(swarm_size, dim) # range:0~1, domain:(swarm_size,dim)
    print('## Optimize:',func_name)

    def clip_by_norm(x, max_norm):
        norm = np.linalg.norm(x)
        return x if norm <=max_norm else x * max_norm / norm

    # Initialize all particle randomly in the search-space
    particles = X * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
    velocities = X * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
    personal_bests = np.copy(particles)
    personal_best_fitness = [np.inf for p in particles] 
    global_best_idx = np.argmin(personal_best_fitness)
    global_best = personal_bests[global_best_idx]
    global_best_fitness = func(global_best)

    # define history object to have all info to be passed to animation function
    history = {'particles':[], 
               'global_best_fitness':[], 
               'global_best':[[np.inf, np.inf] for i in range(num_iters)],
               'obj_func': func_name,}

    # Iteration starts
    for i in range(num_iters):
        history['particles'].append(particles)
        history['global_best_fitness'].append(global_best_fitness)
        # history['global_best'].append(global_best) # seems not working
        history['global_best'][i][0] = global_best[0]
        history['global_best'][i][1] = global_best[1]

        if verbose: print('iter# {}:'.format(i), end='')
        # Evaluate current swarm's fitness:
        # personal best
        for p_i in range(swarm_size):
            fitness = func(particles[p_i])
            if fitness < personal_best_fitness[p_i]:
                personal_bests[p_i] = particles[p_i] # particle
                personal_best_fitness[p_i] = fitness # its fitness
        
        # global best
        if np.min(personal_best_fitness) < global_best_fitness:
            global_best_idx = np.argmin(personal_best_fitness)
            global_best = personal_bests[global_best_idx]
            global_best_fitness = func(global_best)

        # Compute new velocities based on fish brain weightings
        vel_old = inertia * velocities #inertial
        vel_personal = pv * np.random.rand() * (personal_bests - particles) #cognitive
        vel_global = gv * np.random.rand() * (global_best - particles) #social 
        velocities = vel_old + vel_personal + vel_global #update vel
        velocities = clip_by_norm(velocities, max_vnorm) 

        # Update position
        particles = particles + velocities

        # if verbose, print intermediate steps
        if verbose:
            print(' Fitness:{:.5f}, Position:{}, Velocity:{}'.format(global_best_fitness, global_best, np.linalg.norm(velocities)))

    return history

    
#---RUN------------------------------------------------------------------------+    

x0 = [5,5]                   #starting location [x1,x2,x3,....]
bounds = [(-10,10),(-10,10)] #bounds for search [(x1min,x1max),(x2min,x2max),..]
num_particles = 20           #number of swarm particles
maxiter = 30                 #number of steps
PSO.particle_swarm(func1,x0,bounds,num_particles,maxiter)

#---ANIMATIONS-(WIP)------------------------------------------------------------+    

#generate figure 
fig, ax = plt.subplots()
ax = plt.axes(xlim = (-10,10),ylim = (-10,10)) #change to be defined by bounds
n = maxiter
xdata_ani, ydata_ani = [], []
ln, = plt.plot([],[],'ro')
xdata_fields = np.array(xdata_fields)
ydata_fields = np.array(ydata_fields)

#print(ydata_fields[4,4]) #check that its actually a 2D matrix; wont be O.O.B.

def init():
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    return ln,

def update(frame):
    xdata_ani = xdata_fields[frame].tolist()
    ydata_ani = ydata_fields[frame].tolist()
    ln.set_data(xdata_ani,ydata_ani)
    #plt.plot(xdata[frame],ydata[frame],'ro')
    return ln,

ani = FuncAnimation(fig,update,frames=len(xdata_fields),interval=1000)
plt.show()
#ani.save('swarm_ani.gif',writer='pillow')

#---END------------------------------------------------------------------------+    