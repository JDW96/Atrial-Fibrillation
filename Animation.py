#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:06:29 2018

@author: Jack
"""

from Atrium import Atrium
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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