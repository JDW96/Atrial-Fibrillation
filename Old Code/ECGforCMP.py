#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:45:25 2018

@author: Jack
"""

import numpy as np
from line_profiler import LineProfiler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cPickle as pickle

class Atrium():
    def __init__(self, L = 200, nu = 0.17, delta = 0.05, epsilon = 0.05, tauRefrac = 50,
                 tPace = 220, tTot = 10**6, seed_dysfunc = 3, seed_connections = 2, seed_prop = 4):
        
        ### Properties that are arguments
        self.L = L
        self.nu = nu
        self.delta = 0.05
        self.epsilon = 0.05
        self.tauRefrac = tauRefrac
        self.tPace = tPace
        self.tTot = tTot
        
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
        self.ecgProbe = 100.5   ### Lattice num of ECG probe
        self.ecgVoltages = np.full(L**2, -90)
        self.indices = np.arange(L)

        ### Used for animation
        self.phases = np.full((L*L), fill_value = self.tauRefrac)   
        self.pacemakerTimes = np.arange(0,self.tTot, self.tPace)
        
        self.seed_dysfunc = seed_dysfunc
        np.random.seed(self.seed_dysfunc)
        rand = np.random.random(L**2)
        self.dysfunctionalCells = rand < delta  ### True if cell dysfunctional
        

        self.seed_connections = seed_connections
        np.random.seed(self.seed_connections)
        rand = np.random.random(L**2)
        self.connectionsDown = np.full(L**2, None, dtype = float)   ### [Down, Up]
        self.connectionsUp = np.full(L**2, None, dtype = float)
        self.connectionsRight = np.array([i + 1 if i % L != L - 1 else None for i in range(L**2)], dtype = float)
        self.connectionsLeft = np.array([i - 1 if i % L != 0 else None for i in range(L**2)], dtype = float)
        
        connectDown = np.where(rand <= nu)[0]
        
        for i in connectDown:         ### Defines 2D array of connections
            self.connectionsDown[i] = (i + L) % L**2
            self.connectionsUp[(i + L) % L**2] = i
            
        
        self.pacemakerDysfunc = np.array(self.pacemakerCells[self.dysfunctionalCells[self.pacemakerCells]]) ### Array of positions of all dysfunctional pacemaker cells
        self.pacemakerNormal = np.array(self.pacemakerCells[~self.dysfunctionalCells[self.pacemakerCells]])  ### ~ negates all elements in boolean array

        self.seed_prop = seed_prop  ### Seed used when model is run
      
        
    def ECG(self):      
        voltageSum = 0
        
        for x in range(self.L**2):
            if x % self.L == 0:
                continue
            
            i = x % self.L
            j = x / self.L
            
            term1 = (i  - self.ecgProbe) * (self.ecgVoltages[x] - self.ecgVoltages[x - 1])
            term2 = (j - self.ecgProbe) * (self.ecgVoltages[x] - self.ecgVoltages[(x - self.L) % self.L**2])
            divisor = ((i - self.ecgProbe)**2 + (j - self.ecgProbe)**2)**(3./2)
            
            ijVoltage = (term1 + term2) / divisor
            
            voltageSum += ijVoltage
        
        
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
        self.toBeExcited[self.states[0]] = False   ### Excited at start of turn so no longer "to be excited"
        
        
        """
        Is this in the right place?
        Is this fast enough?
        """
        self.ecgVoltages[self.toBeExcited] = 20
        
        for i in range(len(self.states) - 1, 0, -1):
            self.ecgVoltages[self.states[i]] = 20 - ((110./self.tauRefrac) * (self.tauRefrac - i))
            
        self.ecgVoltages[self.resting] = -90
        
        
    def cmpTimestep(self):
        self.ECG()
        
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


def plotting():     ### Plots theoretical distribution
    A = Atrium(tTot = 10**4)
    
    A.cmpFull()
    
    plt.figure(1)
    plt.plot([x for x in range(A.tTot)], A.arrayNumExcited)   ### Number of Excited Cells
    
    
def riskCurveCollection(nRuns = 10):
    nuList = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.3, 0.5, 1])
    np.random.seed(10)
    seeds = np.random.randint(0, 1000000000, (50,3))

    percAFList = np.zeros((len(nuList), nRuns))
    
    for i in range(len(nuList)):
        print nuList[i]
        for j in range(nRuns):
            A = Atrium(nu = nuList[i], seed_dysfunc = seeds[j][0], seed_connections = seeds[j][1], seed_prop = seeds[j][2])
            A.cmpFull()
            
            percAFList[i][j] = float(A.totalTimeAF)/A.tTot
            
    with open('atriumData.pkl', 'wb') as writeData:
        pickle.dump([nuList, percAFList], writeData)



def riskCurvePlotting():
    A = Atrium()
    
    with open('atriumData.pkl', 'rb') as readData:
       nuList, percAFList = pickle.load(readData)
    
    kishanNu = [0.02, 0.04, 0.06, 0.08, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29,
                0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.1]
    
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
    plt.errorbar(kishanNu, kishanTimeAF, kishanError, fmt='x', label = 'Kishan Data')            
    plt.plot(nuList, avgAFList, 'x', 'Compuational Data')
        
    
    
def cmpUpdate(frame_number, mat1, A):
    
    if frame_number in A.pacemakerTimes:
        A.pacemakerExcite()
        
    A.rest_animation()
    A.cellExcite()

    data = A.phases.reshape([A.L,A.L])
    mat1.set_data(data)
    
    return mat1,


"""

Animation here
Doesn't work when in a function???

"""

doAnimation = True

if doAnimation:

    A = Atrium()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = [15,15])
    
    ax1.set_axis_off()
    mat1 = ax1.matshow(A.phases.reshape([A.L,A.L]),cmap=plt.cm.gray_r)
    mat1.set_clim(0,A.tauRefrac)
    ani = animation.FuncAnimation(fig, cmpUpdate, frames = A.tTot,fargs = (mat1,A), interval=50, repeat = None)
    plt.axis([0,A.L,0,A.L])
    
    
    plt.show()















