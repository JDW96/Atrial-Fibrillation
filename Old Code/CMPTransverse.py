# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:21:09 2018

@author: jdw115
"""

"""
LOCATING SLOW CONDUCTION WITH MACHINE LEARNING???
INTEGRATED FIRE MODEL
TRY COUPLING CELLS IN A DIFFERENT LATTICE
READ 'MODELS OF CARDIAC TISSUE' - ALTHOUGH TRYING TO AVOID CONTINUOUS TISSUE MODELS, HEART CELLS NOT CONTINUOUS
HEXAGONAL LATTICE???

REDUCE NU PARALLEL
INVESTIGATE CONNECTIONS
INVESTIGATE CONDUCTION BLOCK

READ REPORTS FROM PAST STUDENTS
"""

import numpy as np
#from line_profiler import LineProfiler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import cPickle as pickle
import os

class Atrium():
    def __init__(self, L = 200, nu = 0.17, delta = 0.05, epsilon = 0.05, tauRefrac = 50,
                 tPace = 220, tTot = 10**6, seed_dysfunc = 3, seed_connections = 2, seed_prop = 4, nuParallel = 0.8, seed_parallel = 5):
        
        ### Properties that are arguments
        self.L = L
        self.nu = nu
        self.delta = 0.05
        self.epsilon = 0.05
        self.tauRefrac = tauRefrac
        self.tPace = tPace
        self.tTot = tTot
        self.nuParallel = nuParallel
        
        ### Things to measure
        self.tNow = 0    ### index to track current position
        self.totalTimeAF = 0
        self.currentTimeAF = 0
        self.currentTimeSinus = 0
        self.numExcited = 0
        self.arrayNumExcited = np.full(self.tTot, 0)
        self.belowThreshAFTime = 0
        
        ### Things to define
        self.pacemakerCells = np.array([i * 200 for i in range(L)])   ### [0, 200, 400, ...]
        self.states = [[]] * tauRefrac    ### 50 arrays, each has cell numbers that are in that refractory period
        self.toBeExcited = np.full(L**2, False)   ### Says whether a cell should be excited or not (boolean)
        self.latticeNums = np.arange(L**2)
        self.resting = np.full(L**2, True)
        
        ### Used for animation
        self.phases = np.full((L*L), fill_value = self.tauRefrac)   
        self.pacemakerTimes = np.arange(0,self.tTot, self.tPace)
        
        self.seed_dysfunc = seed_dysfunc
        np.random.seed(self.seed_dysfunc)
        rand = np.random.random(L**2)
        self.dysfunctionalCells = rand < delta  ### True if cell dysfunctional
        
        self.connectionsDown = np.full(L**2, None, dtype = float)   ### [Down, Up]
        self.connectionsUp = np.full(L**2, None, dtype = float)
        self.connectionsRight = np.full(L**2, None, dtype = float)
        self.connectionsLeft = np.full(L**2, None, dtype = float)

        self.seed_connections = seed_connections
        np.random.seed(self.seed_connections)
        rand = np.random.random(L**2)
        
        connectDown = np.where(rand <= nu)[0]
        
        for i in connectDown:         ### Defines 2D array of connections
            self.connectionsDown[i] = (i + L) % L**2
            self.connectionsUp[(i + L) % L**2] = i
            
        self.seed_parallel = seed_parallel
        np.random.seed(self.seed_parallel)
        rand = np.random.random(L**2)
        
        connectRight = np.where(rand <= self.nuParallel)[0]
        
        for i in connectRight:
            if i + 1 % L != 0:
                self.connectionsRight[i] = (i + 1)
                self.connectionsLeft[i + 1] = i
        
        self.pacemakerDysfunc = np.array(self.pacemakerCells[self.dysfunctionalCells[self.pacemakerCells]]) ### Array of positions of all dysfunctional pacemaker cells
        self.pacemakerNormal = np.array(self.pacemakerCells[~self.dysfunctionalCells[self.pacemakerCells]])  ### ~ negates all elements in boolean array

        self.seed_prop = seed_prop  ### Seed used when model is run
        
        
    def pacemakerExcite(self):            
        randDysfuncFire = np.random.rand(len(self.pacemakerDysfunc))
        self.toBeExcited[self.pacemakerDysfunc[randDysfuncFire > self.epsilon]] = True
        self.toBeExcited[self.pacemakerNormal] = True
     
    def restCells(self):
        self.resting[self.toBeExcited] = False     ### Cells about to be excited will be resting next turn
        self.resting[self.states[-1]] = True     ### Cells in the last state will be resting again next turn
        del self.states[-1]      ### Cells in last refractory state are not refractory next turn
        self.states.insert(0, self.latticeNums[self.toBeExcited])    ### Add in all new excited cells as will be refractory next turn
        
    def rest_animation(self):   ### Only for animation
        """All cells move to the next phase"""
        self.phases[self.toBeExcited] = 0
        self.phases[~self.resting] += 1
        
        self.resting[self.toBeExcited] = False
        self.resting[self.states[-1]] = True
        del self.states[-1]
        self.states.insert(0, self.latticeNums[self.toBeExcited])   
        
    def cellExcite(self):
        neighboursUp = self.connectionsUp[self.states[0][~np.isnan(self.connectionsUp[self.states[0]])]]   ### Lists of all neighbours that can be excited
        neighboursDown = self.connectionsDown[self.states[0][~np.isnan(self.connectionsDown[self.states[0]])]]
        neighboursRight = self.connectionsRight[self.states[0][~np.isnan(self.connectionsRight[self.states[0]])]]
        neighboursLeft = self.connectionsLeft[self.states[0][~np.isnan(self.connectionsLeft[self.states[0]])]]
        
        neighbours = [neighboursUp, neighboursDown, neighboursLeft, neighboursRight]
        neighbours = np.array(np.concatenate(neighbours),dtype = int)
        
        neighbours = neighbours[self.resting[neighbours]]
        
        neighbours_dys = neighbours[self.dysfunctionalCells[neighbours]]
        neighbours_func = neighbours[~self.dysfunctionalCells[neighbours]]
        
        randDysfuncFire = np.random.rand(len(neighbours_dys))
        neighbours_dys = neighbours_dys[randDysfuncFire > self.epsilon]
        
        self.toBeExcited[neighbours_func] = True
        self.toBeExcited[neighbours_dys] = True
        self.toBeExcited[self.states[0]] = False   ### Excited at start of turn so now becoming refractory
        
        
    def cmpTimestep(self):
        if self.tNow % self.tPace == 0:
            self.pacemakerExcite()
            
        self.restCells()
        self.cellExcite()
                
    def cmpFull(self):
        np.random.seed(self.seed_prop)
        
        while self.tNow < self.tTot:
            self.numExcited = np.sum(self.toBeExcited)
            
            if self.numExcited > self.tPace:
                self.totalTimeAF += 1
                self.currentTimeAF += 1
                self.currentTimeSinus = 0
                
            elif self.numExcited <= self.tPace and self.belowThreshAFTime < self.tPace and self.currentTimeAF > 0:
                self.totalTimeAF += 1
                self.currentTimeAF += 1
                self.belowThreshAFTime += 1
            
            elif self.belowThreshAFTime >= self.tPace and self.numExcited < self.tPace:
                self.currentTimeAF = 0
                self.currentTimeSinus += 1
                
            elif self.numExcited <= self.tPace and self.currentTimeAF == 0:
                self.currentTimeSinus += 1
                
            self.arrayNumExcited[self.tNow] = self.numExcited
            
            self.cmpTimestep()
            self.tNow += 1

'''
def lineProfiling():
    A = Atrium()
    
    lp = LineProfiler()
    lp_wrapper = lp(A.cmpFull)
    lp.add_function(A.cmpTimestep)
    lp.add_function(A.pacemakerExcite)
    lp.add_function(A.restCells)
    lp.add_function(A.cellExcite)
    lp_wrapper()
    lp.print_stats()
    
    return A
'''

def plotting():     ### Plots theoretical distribution
    A = Atrium(tTot = 10**4)
    
    A.cmpFull()
    
    plt.figure(1)
    plt.plot([x for x in range(A.tTot)], A.arrayNumExcited)   ### Number of Excited Cells
    
    
def riskCurveCollection(nuListNum, nRuns = 50):
    nuList = np.array([[0.02, 0.04, 0.06, 0.08, 0.11], [0.13, 0.15, 0.17, 0.19, 0.21], 
                       [0.23, 0.25, 0.27, 0.29, 0.12], [0.14, 0.16, 0.18, 0.2, 0.22], 
                       [0.24, 0.26, 0.28, 0.3, 0.1]])

    np.random.seed(10)
    seeds = np.random.randint(0, 1000000000, (50,3))

    percAFList = np.zeros((len(nuList), nRuns))
        
    for i in range(len(nuList[nuListNum])):
        #print nuList[nuListNum][i]
        for j in range(nRuns):
            A = Atrium(nu = nuList[nuListNum][i], seed_dysfunc = seeds[j][0], seed_connections = seeds[j][1], seed_prop = seeds[j][2])
            A.cmpFull()
            
            percAFList[i][j] = float(A.totalTimeAF)/A.tTot
            
        with open(os.getcwd() + '/PickleData/atriumDataNu%s.pkl' %nuList[nuListNum][i], 'wb') as writeData:
            pickle.dump(percAFList[i], writeData)
            


def riskCurvePlotting():
    
    """np.array([0.02, 0.04, 0.06, 0.08, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29,
                0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.1])"""
    
    kishanNuList = np.array([0.02, 0.04, 0.06, 0.08, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29,
                0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.1])   
    
    nuList = np.array([0.02, 0.04, 0.06, 0.08, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29,
                0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.1]) 
        
    nRuns = 10
    
    A = Atrium()
    
    percAFList = np.zeros((len(nuList), nRuns))
    
    for i in range(len(nuList)):
        with open(os.getcwd() + '/PickleData/atriumDataNu%s.pkl' %nuList[i], 'rb') as readData:
            percAFList[i] = pickle.load(readData)
    
    kishanTimeAF = [0.99981, 0.99983, 0.9998, 0.99968, 0.99772, 0.96099, 0.60984, 
                     0.16381, 0.017807, 0.020737, 4.922e-05, 0.0001084, 0, 0, 0.99152,
                     0.86184, 0.29714, 0.039206, 0.0056277, 4.834e-05, 0.00082172, 0, 0,
                     9.406e-05, 0.99919]
    
    kishanError = [4.3015e-06, 3.8088e-06, 1.0454e-05, 3.0663e-05, 0.00044859, 0.018246, 0.054379,
                   0.041092, 0.0080603, 0.016513, 4.8685e-05, 8.4968e-05, 0, 0, 0.0027053, 0.028043, 
                   0.055185, 0.013863, 0.0028284, 3.6005e-05, 0.00081342, 0, 0, 9.3115e-05, 0.00010423]
    
    avgAFList = np.average(percAFList, axis = 1)      ### Actual data
    
    continuousNu = np.linspace(0, 1, 100)
    predFracTimeAF = 1 - (1 - (1 - continuousNu)**A.tauRefrac)**(A.delta*A.L**2)    ### Predicted curve
    
    plt.plot(continuousNu, predFracTimeAF, label = 'Theoretical Data')
    plt.errorbar(kishanNuList, kishanTimeAF, kishanError, fmt='x', label = 'Kishan Data')            
    plt.plot(nuList, avgAFList, 'x', label = 'Compuational Data')
    
    plt.legend()
    
    
    
def update(frame_number, mat, A):
    
    if frame_number in A.pacemakerTimes:
        A.pacemakerExcite()
        
    A.rest_animation()
    A.cellExcite()

    data = A.phases.reshape([A.L,A.L])
    mat.set_data(data)
    
    return mat,


A = Atrium(tTot = 10**4)

fig = plt.figure(figsize = [15,15])
ax = plt.subplot()
ax.set_axis_off()
mat = ax.matshow(A.phases.reshape([A.L,A.L]),cmap=plt.cm.gray_r)
mat.set_clim(0,A.tauRefrac)
ani = animation.FuncAnimation(fig, update, frames = A.tTot,fargs = (mat,A), interval=10, repeat = None)
plt.axis([0,A.L,0,A.L])
plt.show()

"""
for i in range(10):
    A = Atrium(seed_dysfunc = i, seed_prop = i + 1, seed_connections = i + 2)
    A.cmpFull()
    
    print A.numExcited
"""
