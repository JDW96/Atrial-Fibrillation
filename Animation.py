#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:06:29 2018

@author: Jack
"""

from Atrium import Atrium
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as mpat
from matplotlib import collections

def updateSquare(frame_number, mat, A):
    
    if frame_number in A.pacemakerTimes:
        A.pacemakerExcite()
        
    A.rest_animation()
    A.cellExcite()

    data = A.phases.reshape([A.L,A.L])
    mat.set_data(data)
    
    return mat,

def updateHex(frame_number, collection, A):
    if frame_number in A.pacemakerTimes:
        A.pacemakerExcite()
        
    A.rest_animation()
    A.cellExcite()
    
    collection.set_array(np.array(A.phases))
    #ax.add_collection(collection, autolim=True)
    #ax.set_xlim(0,A.size) 
    #ax.set_ylim(0,A.size) 
    
    return ax,

A = Atrium(tTot = 10**4, hexagonal = True, nu_downLeft = 0.3, nu_downRight = 0.3, nu_parallel = 0.5)

if A.hexagonal == True:    
    fig1 = plt.figure(figsize = [15,15])
    ax = fig1.subplots(1,1)
    patches = []
    offsets = []
    
    for i in A.x[0]:
        for j in A.y[:,0]:
            if i % 2 ==0:
                offsets.extend([(j + 0.5, i)]) 
            else:
                offsets.extend([(j + 1, i)]) 
    for k in offsets:
        l = mpat.RegularPolygon(k, 6, radius = 0.55)
        patches.extend([l])
    collection = collections.PatchCollection(patches, cmap=plt.cm.gray_r)
    ax.add_collection(collection, autolim = True)
    
    ax.set_axis_off()
    ani1 = animation.FuncAnimation(fig1, updateHex, frames = A.tTot, 
                                   fargs = (collection, A), interval=100, repeat = None)
    
    plt.axis([0, A.L + 1, 0, A.L + 1])

else:
    fig = plt.figure(figsize = [15, 15])
    ax = plt.subplot()
    ax.set_axis_off()
    mat = ax.matshow(A.phases.reshape([A.L, A.L]),cmap=plt.cm.gray_r)
    mat.set_clim(0,A.tauRefrac)
    ani = animation.FuncAnimation(fig, updateSquare, frames = A.tTot,fargs = (mat, A), interval=10, repeat = None)
    plt.axis([0, A.L,0, A.L])
    plt.show()