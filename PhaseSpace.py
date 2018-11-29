#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 00:36:35 2018

@author: Jack
"""

from Atrium import Atrium
import numpy as np
import os
import cPickle as pickle
import matplotlib.pyplot as plt


def hexPhaseSpace1(numRepeats = 1):      ### nuTrans sets both non parallel axes
    x = np.linspace(0, 0.2, 5, endpoint = False)
    y = np.linspace(0, 1, 20)
    
    results = np.zeros((len(x), len(y)))  ### List of all results
    subRes = np.zeros(numRepeats)      ### 50 repeats of same nu values (different atria)
    
    np.random.seed(1259884)
    seeds = np.random.randint(0, 1000000, (len(x), len(y), numRepeats))     #### Generates a bunch of random seeds for use
    
    for i in range(len(x)):
        print x[i]
        for j in range(len(y)):
            for k in range(numRepeats):
            ### Need some way of defining seeds
                A = Atrium(tTot = 10**5, nu_downLeft = y[j], nu_downRight = y[j], nu_parallel = x[i], hexagonal = True, seed_prop = seeds[i][j][k], seed_dysfunc = 3 * seeds[i][j][k], seed_connections = 7 * seeds[i][j][k])    ### Do we modify the times and stuff for scaling???
                A.cmpFull()
                
                subRes[k] = float(A.totalTimeAF) / A.tTot
                
            results[i][j] = np.average(subRes)    ### Each nu pair has a len(numRepeats) array of different        
                
    with open(os.getcwd() + '/PickleData/hexPhaseSpace%s.pkl' %x[0], 'wb') as writeData:
        pickle.dump(results, writeData)
        
def hexPhaseSpace2(numRepeats = 1):      ### nuTrans sets both non parallel axes
    x = np.linspace(0.2, 0.4, 5, endpoint = False)
    y = np.linspace(0, 1, 20)
    
    results = np.zeros((len(x), len(y)))  ### List of all results
    subRes = np.zeros(numRepeats)      ### 50 repeats of same nu values (different atria)
    
    np.random.seed(1259884)
    seeds = np.random.randint(0, 1000000, (len(x), len(y), numRepeats))     #### Generates a bunch of random seeds for use
    
    for i in range(len(x)):
        print x[i]
        for j in range(len(y)):
            for k in range(numRepeats):
            ### Need some way of defining seeds
                A = Atrium(tTot = 10**5, nu_downLeft = y[j], nu_downRight = y[j], nu_parallel = x[i], hexagonal = True, seed_prop = seeds[i][j][k], seed_dysfunc = 3 * seeds[i][j][k], seed_connections = 7 * seeds[i][j][k])    ### Do we modify the times and stuff for scaling???
                A.cmpFull()
                
                subRes[k] = float(A.totalTimeAF) / A.tTot
                
            results[i][j] = np.average(subRes)    ### Each nu pair has a len(numRepeats) array of different        
                
    with open(os.getcwd() + '/PickleData/hexPhaseSpace%s.pkl' %x[0], 'wb') as writeData:
        pickle.dump(results, writeData)
        
def hexPhaseSpace3(numRepeats = 1):      ### nuTrans sets both non parallel axes
    x = np.linspace(0.4, 0.6, 5, endpoint = False)
    y = np.linspace(0, 1, 20)
    
    results = np.zeros((len(x), len(y)))  ### List of all results
    subRes = np.zeros(numRepeats)      ### 50 repeats of same nu values (different atria)
    
    np.random.seed(1259884)
    seeds = np.random.randint(0, 1000000, (len(x), len(y), numRepeats))     #### Generates a bunch of random seeds for use
    
    for i in range(len(x)):
        print x[i]
        for j in range(len(y)):
            for k in range(numRepeats):
            ### Need some way of defining seeds
                A = Atrium(tTot = 10**5, nu_downLeft = y[j], nu_downRight = y[j], nu_parallel = x[i], hexagonal = True, seed_prop = seeds[i][j][k], seed_dysfunc = 3 * seeds[i][j][k], seed_connections = 7 * seeds[i][j][k])    ### Do we modify the times and stuff for scaling???
                A.cmpFull()
                
                subRes[k] = float(A.totalTimeAF) / A.tTot
                
            results[i][j] = np.average(subRes)    ### Each nu pair has a len(numRepeats) array of different        
                
    with open(os.getcwd() + '/PickleData/hexPhaseSpace%s.pkl' %x[0], 'wb') as writeData:
        pickle.dump(results, writeData)
        
def hexPhaseSpace4(numRepeats = 1):      ### nuTrans sets both non parallel axes
    x = np.linspace(0.6, 0.8, 5, endpoint = False)
    y = np.linspace(0, 1, 20)
    
    results = np.zeros((len(x), len(y)))  ### List of all results
    subRes = np.zeros(numRepeats)      ### 50 repeats of same nu values (different atria)
    
    np.random.seed(1259884)
    seeds = np.random.randint(0, 1000000, (len(x), len(y), numRepeats))     #### Generates a bunch of random seeds for use
    
    for i in range(len(x)):
        print x[i]
        for j in range(len(y)):
            for k in range(numRepeats):
            ### Need some way of defining seeds
                A = Atrium(tTot = 10**5, nu_downLeft = y[j], nu_downRight = y[j], nu_parallel = x[i], hexagonal = True, seed_prop = seeds[i][j][k], seed_dysfunc = 3 * seeds[i][j][k], seed_connections = 7 * seeds[i][j][k])    ### Do we modify the times and stuff for scaling???
                A.cmpFull()
                
                subRes[k] = float(A.totalTimeAF) / A.tTot
                
            results[i][j] = np.average(subRes)    ### Each nu pair has a len(numRepeats) array of different        
                
    with open(os.getcwd() + '/PickleData/hexPhaseSpace%s.pkl' %x[0], 'wb') as writeData:
        pickle.dump(results, writeData)
        
def hexPhaseSpace5(numRepeats = 1):      ### nuTrans sets both non parallel axes
    x = np.linspace(0.8, 1, 5)
    y = np.linspace(0, 1, 20)
    
    results = np.zeros((len(x), len(y)))  ### List of all results
    subRes = np.zeros(numRepeats)      ### 50 repeats of same nu values (different atria)
    
    np.random.seed(1259884)
    seeds = np.random.randint(0, 1000000, (len(x), len(y), numRepeats))     #### Generates a bunch of random seeds for use
    
    for i in range(len(x)):
        print x[i]
        for j in range(len(y)):
            for k in range(numRepeats):
            ### Need some way of defining seeds
                A = Atrium(tTot = 10**5, nu_downLeft = y[j], nu_downRight = y[j], nu_parallel = x[i], hexagonal = True, seed_prop = seeds[i][j][k], seed_dysfunc = 3 * seeds[i][j][k], seed_connections = 7 * seeds[i][j][k])    ### Do we modify the times and stuff for scaling???
                A.cmpFull()
                
                subRes[k] = float(A.totalTimeAF) / A.tTot
            
            results[i][j] = np.average(subRes)    ### Each nu pair has a len(numRepeats) array of different        
                
    with open(os.getcwd() + '/PickleData/hexPhaseSpace%s.pkl' %x[0], 'wb') as writeData:
        pickle.dump(results, writeData)
        
        
        
        
        
        
def hexSpacePlotting():
    x = np.linspace(0, 1, 26)
    y = np.linspace(0, 1, 20)
    
    for x in [0.0, 0.2, 0.4, 0.6, 0.8]:
        with open(os.getcwd() + '/PickleData/hexPhaseSpace%s.pkl' %x, 'rb') as readData:
            results = pickle.load(readData)
            print results
        
    plt.contourf(x, y, results, cmap = plt.cm.viridis_r)
    plt.colorbar()
    

        