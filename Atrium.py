#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:31:46 2018

@author: Jack
"""

import numpy as np
from line_profiler import LineProfiler
import matplotlib.pyplot as plt
import cPickle as pickle
import os
import scipy.stats as stats


class Atrium():
    def __init__(self, L = 200, nu_downLeft = 0.3, nu_downRight = 0.3, nu_parallel = 0.3, nu_trans = 0.14, delta = 0.05, epsilon = 0.05, tauRefrac = 50,
                 tPace = 220, tTot = 10**6, seed_dysfunc = 3, seed_connections = 2, seed_prop = 4, hexagonal = False):
        
        ### Properties that are arguments
        self.L = L
        self.nu_downLeft = nu_downLeft
        self.nu_downRight = nu_downRight
        self.nu_parallel = nu_parallel
        self.nu_trans = nu_trans
        self.delta = delta
        self.epsilon = epsilon
        self.tauRefrac = tauRefrac
        self.tPace = tPace
        self.tTot = tTot
        self.hexagonal = hexagonal
        
        ### Things to measure
        self.tNow = 0    ### index to track current position
        self.totalTimeAF = 0
        self.currentTimeAF = 0
        self.currentTimeSinus = 0
        self.numExcited = 0
        self.arrayNumExcited = np.full(self.tTot, 0)
        self.belowThreshAFTime = 0
        
        ### Things to define
        self.excitations = np.zeros(L**2, dtype = int)      ### Used in new definition of AF time
        self.pacemakerCells = np.arange(0, L**2, L)  ### [0, 200, 400, ...]
        self.states = [[]] * tauRefrac    ### 50 arrays, each has cell numbers that are in that refractory period
        self.AFLocation = 0
        
        self.toBeExcited = np.full(L**2, False)   ### Says whether a cell should be excited or not (boolean)
        self.latticeNums = np.arange(L**2)
        self.resting = np.full(L**2, True)
        self.y = np.indices((self.L, self.L))[0] # y coordinate for cells
        self.x = np.indices((self.L, self.L))[1] # x coordinate for cells
        self.modeList = np.zeros(tTot)
        
        
        ### Used for animation
        self.phases = np.full((L*L), fill_value = self.tauRefrac)   
        self.pacemakerTimes = np.arange(0,self.tTot, self.tPace)
        
        self.seed_dysfunc = seed_dysfunc
        np.random.seed(self.seed_dysfunc)
        rand = np.random.random(L**2)
        self.dysfunctionalCells = rand < delta  ### True if cell dysfunctional
        
        if self.hexagonal:       #### Hexagonal lattice
            self.seed_connections = seed_connections
            np.random.seed(self.seed_connections)
            rand1 = np.random.random(L**2)
            rand2 = np.random.random(L**2)
            rand3 = np.random.random(L**2)
            self.connectionsDownLeft = np.full(L**2, None, dtype = float)
            self.connectionsDownRight = np.full(L**2, None, dtype = float)
            self.connectionsUpLeft = np.full(L**2, None, dtype = float)
            self.connectionsUpRight = np.full(L**2, None, dtype = float)
            self.connectionsRight = np.full(L**2, None, dtype = float)
            self.connectionsLeft = np.full(L**2, None, dtype = float)
            
            #connectDownLeft = np.where(rand1 <= nu)[0]    ### WRONG NU!!!
            #connectDownRight = np.where(rand2 <= nu)[0]      #### Try to make it this way to be more efficient
            
            for i in range(self.L**2):
                if (i / L) % 2 == 0:     #### Use even row rules from Elizabeth Hallowell report
                    if rand1[i] < nu_downLeft:
                        if i % self.L != 0:
                            self.connectionsDownLeft[i] = (i + L - 1) % L**2
                            self.connectionsUpRight[(i + L - 1) % L**2] = i
                        
                    if rand2[i] < nu_downRight:
                        self.connectionsDownRight[i] = (i + L) % L**2
                        self.connectionsUpLeft[(i + L) % L**2] = i
                        
                    if rand3[i] < nu_parallel and i % L != L - 1:
                        self.connectionsRight[i] = (i + 1)
                        self.connectionsLeft[i + 1] = i

                else:        ### Use odd row rules
                    if rand1[i] < nu_downLeft:
                        self.connectionsDownLeft[i] = (i + L) % L**2
                        self.connectionsUpRight[(i + L) % L**2] = i
                    
                    if rand2[i] < nu_downRight:
                        if i % L != L - 1:
                            self.connectionsDownRight[i] = (i + L + 1) % L**2
                            self.connectionsUpLeft[(i + L + 1) % L**2] = i
                            
                    if rand3[i] < nu_parallel and i % L != L - 1:
                        self.connectionsRight[i] = (i + 1)
                        self.connectionsLeft[i + 1] = i
                    
                    
        else:         #### Square lattice
            self.seed_connections = seed_connections
            np.random.seed(self.seed_connections)
            rand1 = np.random.random(L**2)
            rand2 = np.random.random(L**2)
            self.connectionsDown = np.full(L**2, None, dtype = float)   ### [Down, Up]
            self.connectionsUp = np.full(L**2, None, dtype = float)
            self.connectionsRight = np.full(L**2, None, dtype = float)
            self.connectionsLeft = np.full(L**2, None, dtype = float)
            
            ##########################
            ### Having an isolated fibre on the pacemaker boundary fucks everything up
            ### Means we get a reentry circuit closed on one end which excites more times
            ### Makes it a phase singularity when we don't want it to be
            ### Experimented with fully connected pacemaker cells but this messes up the dynamics
            
            """
            
            for i in self.pacemakerCells:
                self.connectionsDown[i] = (i + L) % L**2
                self.connectionsUp[(i + L) % L**2] = i
            
            """
            ##########################
            
            
            
            connectDown = np.where(rand1 <= nu_trans)[0]
            connectRight = np.where(rand2 <= nu_parallel)[0]
            
            
            
            
            for i in connectDown:         ### Defines 2D array of connections
                self.connectionsDown[i] = (i + L) % L**2
                self.connectionsUp[(i + L) % L**2] = i
                
            for i in connectRight:       ### Completely connected for nu_parallel = 1
                if i % L != L - 1:
                    self.connectionsRight[i] = i + 1
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
        
        if self.hexagonal:
            neighboursUpLeft = self.connectionsUpLeft[self.states[0][~np.isnan(self.connectionsUpLeft[self.states[0]])]]   ### Lists of all neighbours that can be excited
            neighboursUpRight = self.connectionsUpRight[self.states[0][~np.isnan(self.connectionsUpRight[self.states[0]])]] 
            neighboursDownLeft = self.connectionsDownLeft[self.states[0][~np.isnan(self.connectionsDownLeft[self.states[0]])]]
            neighboursDownRight = self.connectionsDownRight[self.states[0][~np.isnan(self.connectionsDownRight[self.states[0]])]]
            
            neighboursRight = self.connectionsRight[self.states[0][~np.isnan(self.connectionsRight[self.states[0]])]]
            neighboursLeft = self.connectionsLeft[self.states[0][~np.isnan(self.connectionsLeft[self.states[0]])]]
            
            neighbours = [neighboursLeft, neighboursRight, neighboursUpLeft, neighboursUpRight, neighboursDownLeft, neighboursDownRight]
            
        else:
        
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
        
        
        self.excitations[self.states[0]] += 1
        
        #print self.states[0]
        
        #print "Max excitations value: ", self.excitations.max()
        #print "Excitation at AF: ", self.excitations[self.AFLocation]
        #print "First location: ", np.argmax(self.excitations)
        #print "AF Locations: ", self.AFLocation
        
        #if self.excitations.max() > self.excitations[self.AFLocation]:
        #    self.AFLocation = np.argmax(self.excitations)
        
        #print self.excitations[4200:4205], self.excitations[4240:4245]
        #print self.excitations[4400:4405], self.excitations[4440:4445]
        #print self.excitations[4600:4605], self.excitations[4640:4645]
        
        
        #mode = np.bincount(self.excitations).argmax()
        #self.modeList[self.tNow] = mode
        
        self.tNow += 1
        

        
    def cmpFull(self):
        np.random.seed(self.seed_prop)
        
        previousMax = 0
        locPacemaker = True  ### Location of max excitement
        
        times = 0
        elseTimes = 0
        
        while self.tNow < self.tTot:
            
            #self.numExcited = np.sum(self.toBeExcited)
            #self.arrayNumExcited[self.tNow] = self.numExcited    ### Used for plotting, an array of the number of excited cells at each time
            
            self.cmpTimestep()     ### Do one step of CMP model
    
            #print "excitations: ", self.excitations.max()
            #print "pacemaker: ", (self.excitations[self.pacemakerCells]).max()
            
            ### Old AF Definition
              
            """
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
            """
              
            ### Phase singularity AF definition
            
            #print "Pacemaker: ", (self.excitations[self.pacemakerCells]).max()
            #print "Excitations: ", (self.excitations).max()
            #print "num excited: ", self.numExcited
            
            
            #newMax = (self.excitations).max()
            #paceMax = (self.excitations[self.pacemakerCells]).max()
            
            """
            if newMax > previousMax:  
                times += 1
                if paceMax == newMax:
                    self.excitations[:] = 0
                    self.currentTimeAF = 0
                    locPacemaker = True
                    
                else:     #### AF starts here
                    self.currentTimeAF += 1
                    self.totalTimeAF += 1
                    locPacemaker = False
                    
            else:      ### No change in state
                elseTimes += 1
                if locPacemaker:    ### Longest standing excitation is at pacemaker
                    self.currentTimeAF = 0
                    
                else:     ### AF continues here
                    self.currentTimeAF += 1
                    self.totalTimeAF += 1
            """      
            
            """
            ### Sinus Rhythm
            if self.currentTimeAF != 0 and (self.excitations[self.pacemakerCells]).max() >= (self.excitations).max():
                self.currentTimeAF = 0
                
            ######### need to empty excitations??????
                
            ### Atrial Fibrillation       
            elif (self.excitations).max() > (self.excitations[self.pacemakerCells]).max():
                self.currentTimeAF += 1   
                self.totalTimeAF += 1
            """


            #maxLocs = np.where(self.excitations == self.excitations.max())[0]
            
            #if self.excitations.max() > :
            #    self.maxExciteLocs = self.excitations[maxLocs]
        
            #print self.maxExciteLocs
    
            #self.excitations = self.excitations % 4
            #self.excitations[self.maxExciteLocs] = 10



            #previousMax = newMax


            
        #print "times: ", times
        #print "elseTimes: ", elseTimes



def lineProfiling():
    A = Atrium(hexagonal = False, tTot = 10**3, nu_parallel = 1, nu_trans = 0.13)
    
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
    
    
def riskCurveCollection(nuNum, nRuns = 50, hexagonal = True):
    if hexagonal:
        string = "Hex"
    else:
        string = "Square"
    
    nuList = np.array([[0.01, 0.02, 0.03, 0.035],
                       [0.04, 0.045, 0.05, 0.055, 0.06],
                       [0.065, 0.07, 0.075, 0.08, 0.09],
                       [0.095, 0.1, 0.11, 0.12],
                       [0.13, 0.15, 0.2, 0.3]])     ### Just need to run for this value....

    np.random.seed(10)
    seeds = np.random.randint(0, 1000000000, (50,3))

    percAFList = np.zeros((len(nuList), nRuns))
        
    for i in range(len(nuList[nuNum])):
        print nuList[nuNum][i]
        for j in range(nRuns):
            A = Atrium(tTot = 10**5, hexagonal = hexagonal, nu_downLeft = nuList[nuNum][i], nu_downRight = nuList[nuNum][i], nu_parallel = 1, seed_dysfunc = seeds[j][0], seed_connections = seeds[j][1], seed_prop = seeds[j][2])
            A.cmpFull()
            
            percAFList[i][j] = float(A.totalTimeAF)/A.tTot
            
        with open(os.getcwd() + '/PickleData/atriumDataNuNewAF%s%s.pkl' %(string, nuList[nuNum][i]), 'wb') as writeData:
            pickle.dump(percAFList[i], writeData)
            


def riskCurvePlotting():
    
    """np.array([0.02, 0.04, 0.06, 0.08, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29,
                0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.1])"""
    
    hexNuList = np.array([0.01, 0.02, 0.03, 0.035,
                       0.04, 0.045, 0.05, 0.055, 0.06,
                       0.065, 0.07, 0.075, 0.08, 0.09,
                       0.095, 0.1, 0.11, 0.12,
                       0.13, 0.15, 0.2, 0.3])
    
    kishanNuList = np.array([0.02, 0.04, 0.06, 0.08, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29,
                0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.1])   
    
    #nuList = np.array([0.02, 0.04, 0.06, 0.08, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29,
    #            0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.1]) 
     
    nuList = np.array([0.01, 0.02, 0.03, 0.035,
                       0.04, 0.045, 0.05, 0.055, 0.06,
                       0.065, 0.07, 0.075, 0.08, 0.09,
                       0.095, 0.1, 0.11, 0.12,
                       0.13, 0.15, 0.2, 0.3])
    
    nRuns = 1
    
    A = Atrium()
    
    percAFList = np.zeros((len(nuList), nRuns))
    percHexList = np.zeros((len(hexNuList), nRuns))
    
    for i in range(len(nuList)):
        with open(os.getcwd() + '/PickleData/atriumDataNuNewAFSquare%s.pkl' %nuList[i], 'rb') as readData:
            percAFList[i] = pickle.load(readData)
            
    #for i in range(len(hexNuList)):
    #    with open(os.getcwd() + '/PickleData/atriumDataNuHex%s.pkl' %hexNuList[i], 'rb') as readData:
    #        percHexList[i] = pickle.load(readData)
    
    kishanTimeAF = [0.99981, 0.99983, 0.9998, 0.99968, 0.99772, 0.96099, 0.60984, 
                     0.16381, 0.017807, 0.020737, 4.922e-05, 0.0001084, 0, 0, 0.99152,
                     0.86184, 0.29714, 0.039206, 0.0056277, 4.834e-05, 0.00082172, 0, 0,
                     9.406e-05, 0.99919]
    
    kishanError = [4.3015e-06, 3.8088e-06, 1.0454e-05, 3.0663e-05, 0.00044859, 0.018246, 0.054379,
                   0.041092, 0.0080603, 0.016513, 4.8685e-05, 8.4968e-05, 0, 0, 0.0027053, 0.028043, 
                   0.055185, 0.013863, 0.0028284, 3.6005e-05, 0.00081342, 0, 0, 9.3115e-05, 0.00010423]
    
    avgAFList = np.average(percAFList, axis = 1)      ### Actual data
    avgHexList = np.average(percHexList, axis = 1)
    
    continuousNu = np.linspace(0, 1, 100)
    predFracTimeAF = 1 - (1 - (1 - continuousNu)**A.tauRefrac)**(A.delta*A.L**2)    ### Predicted curve
    predHexTimeAF = 1 - (1 - (1 - continuousNu)**(2*A.tauRefrac))**(A.delta*A.L**2)
    
    plt.plot(continuousNu, predFracTimeAF, label = 'Theoretical Data')
    #plt.plot(continuousNu, predHexTimeAF, label = 'Theoretical Hex Data')
    plt.errorbar(kishanNuList, kishanTimeAF, kishanError, fmt='x', label = 'Kishan Data')            
    plt.plot(nuList, avgAFList, 'x', label = 'Compuational Data')
    #plt.plot(hexNuList, avgHexList, 'x', label = 'Hex Data')
    
    plt.legend()
    
    
    
def update(frame_number, mat, A):
    
    if frame_number in A.pacemakerTimes:
        A.pacemakerExcite()
        
    A.rest_animation()
    A.cellExcite()

    data = A.phases.reshape([A.L,A.L])
    mat.set_data(data)
    
    return mat,



def avgHexNeighbours(nuList, nRuns):
    avgList = np.zeros((len(nuList), len(nuList)))
    seed = 137532
    
    for i in range(len(nuList)):
        print nuList[i]
        
        for j in range(len(nuList)):
            numConnections = 0
            
            for k in range(nRuns):
                A = Atrium(nu_parallel = nuList[i], nu_downLeft = nuList[j], nu_downRight = nuList[j], hexagonal = True, seed_dysfunc = seed, seed_connections = seed * 2, seed_prop = seed * 4)
                seed += 12441
                
                for l in range(A.L**2):
                    if not np.isnan(A.connectionsRight[l]):
                        numConnections += 1
                        
                    if not np.isnan(A.connectionsDownRight[l]):
                        numConnections += 1
                        
                    if not np.isnan(A.connectionsUpRight[l]):
                        numConnections += 1
                        
                    if not np.isnan(A.connectionsLeft[l]):
                        numConnections += 1
                        
                    if not np.isnan(A.connectionsDownLeft[l]):
                        numConnections += 1
                        
                    if not np.isnan(A.connectionsUpLeft[l]):
                        numConnections += 1
                
            avgList[i][j] += numConnections / float(A.L**2 * nRuns)

    return avgList



def hexPhaseSpace(nuPara, nuTrans, numRepeats = 50):      ### nuTrans sets both non parallel axes
    results = np.zeros((len(nuPara), len(nuTrans)))  ### List of all results
    subRes = np.zeros(numRepeats)      ### 50 repeats of same nu values (different atria)
    
    np.random.seed(1259884)
    seeds = np.random.randint(0, 1000000, (len(nuPara), len(nuTrans), numRepeats))     #### Generates a bunch of random seeds for use
    
    for i in range(len(nuPara)):
        print nuPara[i]
        for j in range(len(nuTrans)):
            for k in range(numRepeats):
            ### Need some way of defining seeds
                A = Atrium(tTot = 10**5, nu_downLeft = nuTrans[j], nu_downRight = nuTrans[j], nu_parallel = nuPara[i], hexagonal = True, seed_prop = seeds[i][j][k], seed_dysfunc = 3 * seeds[i][j][k], seed_connections = 7 * seeds[i][j][k])    ### Do we modify the times and stuff for scaling???
                A.cmpFull()
                
                subRes[k] = float(A.totalTimeAF) / A.tTot
                
            results[i][j] = np.average(subRes)    ### Each nu pair has a len(numRepeats) array of different        
                
    with open(os.getcwd() + '/PickleData/hexPhaseSpace%s.pkl' %nuPara[0], 'wb') as writeData:
        pickle.dump(results, writeData)
            


def hexSpacePlotting(nuPara, nuTrans):
    
    with open(os.getcwd() + '/PickleData/hexPhaseSpace.pkl', 'rb') as readData:
        results = np.asarray(pickle.load(readData))
        
    plt.contourf(nuPara, nuTrans, results, cmap = plt.cm.viridis_r)
    plt.colorbar()

nuList = np.linspace(0, 1, 25, endpoint = False)

nuPara = np.linspace(0, 1, 10)


def modePlotting():
    A = Atrium(nu_parallel = 1, nu_trans = 0.1, tTot = 10**5)
    
    A.cmpFull()
    
    plt.plot(A.modeList)
    
    


