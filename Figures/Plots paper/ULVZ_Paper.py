#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:14:36 2018

@author: irenebonati
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import*
import os
from decimal import Decimal
from matplotlib import colors
import csv
import pandas as pd


font = {'family' : 'normal',
        'size'   : 14}

font2 = {'family' : 'normal',
         
        'size'   : 12,
        'weight': 'bold'}


pattern = '00000','00010','050','100','399'
color = ['gold','darkorange','crimson','mediumorchid','rebeccapurple']
label = ['0','0.05','0.25','0.5','2']

#lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,+0.1))

# ------------------------------ AXISYMMETRIC ---------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../../Axisymmetric/Du1_Dp0_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../../Axisymmetric/Du1_Dp0_plot')
for i in range(len(matching_files)):
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(figsize=(14,6.5))
plt.figure(figsize=(12,5))
plt.subplots_adjust(hspace=0.7)
ax1 = plt.subplot(245)
ax = plt.gca()
for j in range(len(pattern)):
    ax1.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
plt.ylabel('$h/h_{\mathrm{0}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,2])
ax.text(.5,.87,'$D_{\mathrm{u}}=1$',
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du0.1_Dp0_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du0.1_Dp0_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(246)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,3])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-1}$',
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.text(.78,1.1,'Axisymmetric',
        horizontalalignment='left',
        transform=ax.transAxes,
        **font2)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-2_Dp0_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-2_Dp0_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
plt.subplot(247)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlabel('$s/L_{\mathrm{r}}$')
plt.xlim([0,1])
plt.ylim([0,6])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-2}$',
        horizontalalignment='center',
        transform=ax.transAxes)
# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-3_Dp0_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-3_Dp0_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.subplot(248)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,10])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-3}$',
        horizontalalignment='center',
        transform=ax.transAxes)


# CARTESIAN
# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../../Cartesian/Du1_Dp0_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../../Cartesian/Du1_Dp0_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.subplot(241)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j],label=label[j])
plt.ylabel('$h/h_{\mathrm{0}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,2])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=1$',
        horizontalalignment='center',
        transform=ax.transAxes)
#plt.legend(title='Non-dimensional time',bbox_to_anchor=(4, 1.6),
#          ncol=5, fancybox=True)
#plt.figlegend(title='Time (non-dimensional)',loc = (0.33,0.895), ncol=5)#, labelspacing=0.5)
#(0.33,0.925)
L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-1_Dp0_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-1_Dp0_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(242)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,3])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-1}$',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.text(.88,1.1,'Cartesian',
        horizontalalignment='left',
        transform=ax.transAxes,
        **font2)

dir = '../Du1e-2_Dp0_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-2_Dp0_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(243)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,5])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-2}$',
        horizontalalignment='center',
        transform=ax.transAxes)

dir = '../Du1e-3_Dp0_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-3_Dp0_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(244)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,8])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-3}$',
        horizontalalignment='center',
        transform=ax.transAxes)
#os.chdir('../../ULVZ_Paper')
#plt.savefig('Time_evolution_Dp0.pdf', bbox_inches='tight',format='pdf')
plt.savefig('ALLORA.pdf', bbox_inches='tight',format='pdf')
plt.show()


# DP = 1

pattern = '00000','00010','050','099'
color = ['gold','darkorange','crimson','rebeccapurple']

# ------------------------------ AXISYMMETRIC ---------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1_Dp1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1_Dp1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.figure(figsize=(12,5))
plt.subplots_adjust(hspace=0.7)
#plt.subplots_adjust(wspace=0.3)
ax1 = plt.subplot(245)
ax = plt.gca()
for j in range(len(pattern)):
    ax1.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
plt.ylabel('$h/h_{\mathrm{0}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,7])
ax.text(.5,.87,'$D_{\mathrm{u}}=1$',
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('$s/L_{\mathrm{r}}$')


pattern = '00000','00010','050','100','300'
color = ['gold','darkorange','crimson','mediumorchid','rebeccapurple']

# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du0.1_Dp1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du0.1_Dp1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(246)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,7])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-1}$',
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.text(.78,1.1,'Axisymmetric',
        horizontalalignment='left',
        transform=ax.transAxes,
        **font2)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-2_Dp1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-2_Dp1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
plt.subplot(247)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlabel('$s/L_{\mathrm{r}}$')
plt.xlim([0,1])
plt.ylim([0,9])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-2}$',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-3_Dp1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-3_Dp1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.subplot(248)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,13])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-3}$',
        horizontalalignment='center',
        transform=ax.transAxes)

# CARTESIAN
# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../../Cartesian/Du1_Dp1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../../Cartesian/Du1_Dp1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.subplot(241)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j],label=label[j])
plt.ylabel('$h/h_{\mathrm{0}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,4])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
#plt.figlegend(title='Non-dimensional time [$U_{\mathrm{0}}/L_{\mathrm{r}}]$',loc = (0.31,0.925), ncol=5)#, labelspacing=0.5)
ax.text(.5,.87,'$D_{\mathrm{u}}=1$',
        horizontalalignment='center',
        transform=ax.transAxes)

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du0.1_Dp1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du0.1_Dp1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(242)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,4])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-1}$',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.text(.88,1.1,'Cartesian',
        horizontalalignment='left',
        transform=ax.transAxes,
        **font2)



dir = '../Du1e-2_Dp1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-2_Dp1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(243)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,5])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-2}$',
        horizontalalignment='center',
        transform=ax.transAxes)

dir = '../Du1e-3_Dp1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-3_Dp1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(244)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,8])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-3}$',
        horizontalalignment='center',
        transform=ax.transAxes)
#os.chdir('../../ULVZ_Paper')
plt.savefig('Time_evolution_Dp1.pdf', bbox_inches='tight',format='pdf')
plt.show()


# DP = 0.1

# ------------------------------ AXISYMMETRIC ---------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.figure(figsize=(12,5))
plt.subplots_adjust(hspace=0.7)
ax1 = plt.subplot(245)
ax = plt.gca()
for j in range(len(pattern)):
    ax1.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
plt.ylabel('$h/h_{\mathrm{0}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,4])
ax.text(.5,.87,'$D_{\mathrm{u}}=1$',
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('$s/L_{\mathrm{r}}$')

# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du0.1_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du0.1_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(246)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,4])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-1}$',
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.text(.78,1.1,'Axisymmetric',
        horizontalalignment='left',
        transform=ax.transAxes,
        **font2)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-2_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-2_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
plt.subplot(247)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlabel('$s/L_{\mathrm{r}}$')
plt.xlim([0,1])
plt.ylim([0,6])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-2}$',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-3_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-3_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.subplot(248)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,11])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-3}$',
        horizontalalignment='center',
        transform=ax.transAxes)

# CARTESIAN
# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../../Cartesian/Du1_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../../Cartesian/Du1_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.subplot(241)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
plt.ylabel('$h/h_{\mathrm{0}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,2])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=1$',
        horizontalalignment='center',
        transform=ax.transAxes)

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du0.1_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du0.1_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(242)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,3])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-1}$',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.text(.88,1.1,'Cartesian',
        horizontalalignment='left',
        transform=ax.transAxes,
        **font2)



dir = '../Du1e-2_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-2_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(243)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,5])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-2}$',
        horizontalalignment='center',
        transform=ax.transAxes)

dir = '../Du1e-3_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-3_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(244)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,7])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-3}$',
        horizontalalignment='center',
        transform=ax.transAxes)
#os.chdir('../../ULVZ_Paper')
plt.savefig('Time_evolution_Dp0.1.pdf', bbox_inches='tight',format='pdf')
plt.show()




# DP = 0.01

# ------------------------------ AXISYMMETRIC ---------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1_Dp0.01_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1_Dp0.01_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.figure(figsize=(12,5))
plt.subplots_adjust(hspace=0.7)
ax1 = plt.subplot(245)
ax = plt.gca()
for j in range(len(pattern)):
    ax1.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
plt.ylabel('$h/h_{\mathrm{0}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,4])
ax.text(.5,.87,'$D_{\mathrm{u}}=1$',
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du0.1_Dp0.01_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du0.1_Dp0.01_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(246)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,4])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-1}$',
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.text(.78,1.1,'Axisymmetric',
        horizontalalignment='left',
        transform=ax.transAxes,
        **font2)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-2_Dp0.01_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-2_Dp0.01_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
plt.subplot(247)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,6])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-2}$',
        horizontalalignment='center',
        transform=ax.transAxes)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du1e-3_Dp0.01_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-3_Dp0.01_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.subplot(248)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
#plt.xlabel('$s/L_{\mathrm{r}}$')
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlabel('$s/L_{\mathrm{r}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,11])
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-3}$',
        horizontalalignment='center',
        transform=ax.transAxes)

# CARTESIAN
# ----------------------------------------------------------------------------

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../../Cartesian/Du1_Dp0.01_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../../Cartesian/Du1_Dp0.01_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

plt.subplot(241)
ax = plt.gca()
for j in range(len(pattern)):
    plt.plot(L_out[:,j],h_out[:,j],color=color[j])
plt.ylabel('$h/h_{\mathrm{0}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlim([0,1])
plt.ylim([0,2])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=1$',
        horizontalalignment='center',
        transform=ax.transAxes)

L_out   = np.zeros((257,40),dtype=np.float32)
h_out   = np.zeros((257,40),dtype=np.float32)

dir = '../Du0.1_Dp0.01_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du0.1_Dp0.01_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(242)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,3])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-1}$',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.text(.88,1.1,'Cartesian',
        horizontalalignment='left',
        transform=ax.transAxes,
        **font2)



dir = '../Du1e-2_Dp0.01_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-2_Dp0.01_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(243)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,5])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-2}$',
        horizontalalignment='center',
        transform=ax.transAxes)

dir = '../Du1e-3_Dp0.1_plot'      

matching_files = [0]*len(pattern)
for i in range(len(pattern)):
    matching_files[i] = [f for f in os.listdir(dir) if pattern[i] in f]

files = [i[0] for i in matching_files]

os.chdir('../Du1e-3_Dp0.1_plot')
for i in range(len(matching_files)): 
    df=pd.read_csv(files[i],sep='\s+')

    df.columns = ["L", "h"]

    L = df.loc[:,'L']
    L = L.astype(np.float32)
    h = df.loc[:,'h']
    h = h.astype(np.float32)

    for j in range(len(L)):
        L_out[j,i] = L[j]
        h_out[j,i] = h[j]

#plt.figure(1)
ax2 = plt.subplot(244)
ax = plt.gca()
for j in range(len(pattern)):
    ax2.plot(L_out[:,j],h_out[:,j],color=color[j])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
#plt.ylabel('$h/h_{\mathrm{0}}$')
plt.xlim([0,1])
plt.ylim([0,7])
#ax.set_xticklabels([])
plt.xlabel('$x/L_{\mathrm{r}}$')
ax.text(.5,.87,'$D_{\mathrm{u}}=10^{-3}$',
        horizontalalignment='center',
        transform=ax.transAxes)
#os.chdir('../../ULVZ_Paper')
plt.savefig('Time_evolution_Dp0.01.pdf', bbox_inches='tight',format='pdf')
plt.show()
