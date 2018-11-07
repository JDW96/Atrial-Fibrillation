#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:39:08 2018

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import time


greys = np.linspace(210, 30, 50, endpoint = True)/255   ### Various grey colours
greys = [(x, x, x) for x in greys]

cmap = ListedColormap(['k', 'w'] + list(greys))

np.random.seed(172359643)

class Cell:
    
    def __init__(self, tauRefrac, delta, epsilon):
        self.state = 0   ### State = 0, 1, 2, 3, 4, ... for resting, excited, 
                         ### refractory 1, refractory 2 ....
        self.tauRefrac = tauRefrac
        self.delta = delta
        self.epsilon = epsilon
        self.dysfunctional = self.isDysfunctional()
        self.connections = [False, False]   ### [down, up] 
        self.wasUpdated = False         
        """ Four boolean vector maybe??? When nu,x < 1 """

    def isDysfunctional(self):
        rand = np.random.random()
        if rand < delta:
            return True
        else:
            return False


    def dysfuncExcites(self):
        rand = np.random.random()
        if rand < epsilon:
            return False   ### Doesn't excite
        else:
            return True    ### Excites


    def changeState(self):
        self.state = (self.state + 1) % (self.tauRefrac + 2) # Need to make sure state is correct
        

L = 200 ### Size of lattice (currently square)  
T = 220  # Excitement time
tauRefrac = 50  # Refractory time
nu = 0.1
delta = 0.05   # Fraction of dysfunctional cells
epsilon = 0.05  # Probability of not firing
j = 0

""" Make lattice 1D to save memory """

lattice = [Cell(tauRefrac, delta, epsilon) for i in range(L**2)]

"""
This defines the lattice connections

0 means no connections
1 means connected down
2 means connected up
3 means connected both down and up

MAKE SURE PERIODIC FROM TOP AND BOTTOM
"""

for i in range(L**2):   ### Don't do last row due to periodic conditions
    rand = np.random.random()
    
    if rand < nu:
        #### Connection below
        cellStart = lattice[i]
        cellStart.connections[0] = True     ### Connected down
        
        cellEnd = lattice[(i + L) % L**2]
        cellEnd.connections[1] = True       ### Connected up
        
""" No pacemaker cells implemented yet 
    No dysfunctional cells implemented yet
    Seem to have gotten boundary conditions wrong way around
    Connections not yet implemented
"""

states = np.array([cell.state for cell in lattice])
grid = states.reshape([L, L]) 

###for i in range(10**3):   # Change this to run indefinitely


#### Maybe trying each cell twice??????


start = time.time()
def update(data):     # Probably needs other arguments    
    global grid
    global states
    global j   
    
    if j == 10**4:
        end = time.time()
        print end - start
    
    newGrid = grid.copy()
    
    excited = np.where(states == 1)[0]     #### Array of excited state positions
    refractory = np.where(states > 1)[0]
    
    if j % T == 0:       ### Pacemaker           
        pacemakerExcite = np.where(states[::L] == 0)[0]   ### Don't excite refractory/excited cells, only checks firstrow
        
        for rowIndex in pacemakerExcite:
            cell = lattice[rowIndex * L]
            
            if cell.dysfunctional:
                if cell.dysfuncExcites():
                    lattice[rowIndex * L].changeState()    ### Need to multiply by L as above only gives rows to be excited    
            else:
                lattice[rowIndex * L].changeState()
    
    j += 1      ### Used for pacemaker
    
    toBeUpdated = set()    ### Set of cell index positions to be updated at end of turn
    
    for excitedCellIndex in excited:
        if (excitedCellIndex + 1) % L != 0 and states[excitedCellIndex + 1] == 0:    ### Cell is to the right (closed boundary)
            cell = lattice[excitedCellIndex + 1]
            if cell.dysfunctional:
                if cell.dysfuncExcites():
                    toBeUpdated.add(excitedCellIndex + 1)
            else:
                toBeUpdated.add(excitedCellIndex + 1)
            
        if (excitedCellIndex - 1) % L != L - 1  and states[excitedCellIndex - 1] == 0:    ### Cell is to the left (closed boundary)
            cell = lattice[excitedCellIndex - 1]
            if cell.dysfunctional:
                if cell.dysfuncExcites():
                    toBeUpdated.add(excitedCellIndex - 1)
            else:
                toBeUpdated.add(excitedCellIndex - 1)
            
        if lattice[excitedCellIndex].connections[0] == True and states[(excitedCellIndex + L) % L**2] == 0:    ### Cell is below (open boundary)
            cell = lattice[(excitedCellIndex + L) % L**2]
            if cell.dysfunctional:
                if cell.dysfuncExcites():
                    toBeUpdated.add((excitedCellIndex + L) % L**2)
            else:
                toBeUpdated.add((excitedCellIndex + L) % L**2)
            
        if lattice[excitedCellIndex].connections[1] == True and states[(excitedCellIndex - L) % L**2] == 0:    ### Cell is above (open boundary)
            cell = lattice[(excitedCellIndex - L) % L**2]
            if cell.dysfunctional:
                if cell.dysfuncExcites():
                    toBeUpdated.add((excitedCellIndex - L) % L**2)
            else:
                toBeUpdated.add((excitedCellIndex - L) % L**2)
        
    toBeUpdated.update(excited, refractory)   
    
    for cell in toBeUpdated:
        lattice[cell].changeState()      ### Excites all resting cells
        
    states = np.array([cell.state for cell in lattice])
    newGrid = states.reshape([L, L])

    mat.set_data(newGrid)
    grid = newGrid
    
    return mat

fig, ax = plt.subplots()
ax.set_axis_off()

mat = ax.matshow(grid, cmap = cmap)
mat.set_clim(0, tauRefrac)
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
plt.show()

            
        
        
        
        
        
        
        
        
        
        

