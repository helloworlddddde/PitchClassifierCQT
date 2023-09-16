# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:49:53 2023

@author: rayne
"""
import matplotlib.pyplot as plt
import numpy as np
def spectogram(spec, f, t, minf, maxf):
    fig, ax = plt.subplots()
    T, F = np.meshgrid(t, f)
    cp = ax.contourf(T, F, spec)
    ax.set_ylim([minf, maxf])
    plt.yscale('log')
    fig.colorbar(cp)
    return fig, ax

def chroma(chroma, f, t):
    fig, ax = plt.subplots()
    c = ax.imshow(chroma, origin='lower')
    fig.colorbar(c)
    yticks = np.arange(0, 12)
    ax.set_yticks(yticks)
    return fig, ax
    

