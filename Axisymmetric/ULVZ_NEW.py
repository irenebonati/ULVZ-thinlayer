#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:57:27 2018

@author: irenebonati
"""

import numpy as np
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from matplotlib.pyplot import*
import os
from decimal import Decimal
from matplotlib.mlab import griddata
from matplotlib import colors
import csv
import pandas as pd



font = {'family' : 'normal',
        'size'   : 14}

font2 = {'family' : 'normal',
         
        'size'   : 12,
        'weight': 'bold'}


# ------------------------------ AXISYMMETRIC ---------------------------------


res = 0
           
os.chdir("Du1e-10_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results = np.zeros((40,3),dtype=np.float32)
h_max   = np.zeros((40),dtype=np.float32)
L_final   = np.zeros((len_data,40),dtype=np.float32)
h_final   = np.zeros((len_data,40),dtype=np.float32)

results[res,0] = 1e-10     # Du
results[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = ((results[res,2])**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-9_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-9      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-8_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-8      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-7_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-7      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-6_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-6      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]


#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-5_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-5      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-4_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-4      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-3_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-3      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-2_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-2      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du0.1_Dp0")

df=pd.read_csv('h00099.txt',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-1      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = 1#*np.pi

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]


#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-10_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-10      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]


#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-9_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-9      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]


#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-8_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-8      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-7_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-7      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-6_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-6     # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]


#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-5_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-5      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-4_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-4      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-3_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-3      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1e-2_Dp1")

df=pd.read_csv('h00091.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1e-2      # Du
results[res,1]= 0       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (results[res,2]**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]


#----------------------------------------------------- 

res = res + 1

os.chdir("../Du0.1_Dp1")

df=pd.read_csv('h00034.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 0.1      # Du
results[res,1]= 1       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (0.55**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]

#----------------------------------------------------- 

res = res + 1

os.chdir("../Du1_Dp1")

df=pd.read_csv('h00024.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

results[res,0] = 1.      # Du
results[res,1]= 1       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results[res,2] = L[i]
        break

results[res,2] = (0.98**2)

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]

h_max[res] = h[0]



plt.figure(2)
plt.plot(L_final[:,10],h_final[:,10],label='$D_{\mathrm{u}}$=$1$')
plt.plot(L_final[:,9],h_final[:,9],label='$D_{\mathrm{u}}$=$10^{-1}$')
plt.plot(L_final[:,8],h_final[:,8],label='$D_{\mathrm{u}}$=$10^{-2}$')
plt.plot(L_final[:,7],h_final[:,7],label='$D_{\mathrm{u}}$=$10^{-3}$')
plt.plot(L_final[:,21],h_final[:,21],color='#1f77b4',linestyle = ':')
plt.plot(L_final[:,20],h_final[:,20],color='#ff7f0e',linestyle = ':')
plt.plot(L_final[:,19],h_final[:,19],color='#2ca02c',linestyle = ':')
plt.plot(L_final[:,18],h_final[:,18],color='#d62728',linestyle = ':')
#
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
display = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)

simArtist2 = plt.Line2D((0,1),(0,0), color='k')
anyArtist2 = plt.Line2D((0,1),(0,0), color='k',marker='',linestyle=':')


plt.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist2,anyArtist2],
          [label for i,label in enumerate(labels) if i in display]+['$D_{\mathrm{p}}$=0','$D_{\mathrm{p}}$=1'],bbox_to_anchor=(0.98, 0.98),borderaxespad=0.,fontsize=8)
         

plt.xlim([0,0.8])
plt.ylim([0,13])
plt.xlabel('L/$L_{\mathrm{m}}$')
plt.ylabel('$h/h_{\mathrm{0}}$')
plt.title('Axisymmetric')
plt.savefig('ULVZ_CMB_cov_ax.pdf', bbox_inches='tight',format='pdf')               
plt.show()



plt.figure(2)
plt.plot(L_final[:,10],h_final[:,10],label='$D_{\mathrm{u}}$=$1$')
plt.plot(L_final[:,9],h_final[:,9],label='$D_{\mathrm{u}}$=$10^{-1}$')
plt.plot(L_final[:,8],h_final[:,8],label='$D_{\mathrm{u}}$=$10^{-2}$')
plt.plot(L_final[:,7],h_final[:,7],label='$D_{\mathrm{u}}$=$10^{-3}$')

# ------------------------------ CARTESIAN ---------------------------------
res = 0
           
os.chdir("../../Cartesian/Du1e-10_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car = np.zeros((40,3),dtype=np.float32)
h_max   = np.zeros((40),dtype=np.float32)
L_final   = np.zeros((len_data,40),dtype=np.float32)
h_final   = np.zeros((len_data,40),dtype=np.float32)

results_car[res,0] = 1e-10     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#-----------------------------------------------------

res = res+1
           
os.chdir("../Du1e-9_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-9     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 

res = res+1
           
os.chdir("../Du1e-8_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-8     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 

res = res+1
           
os.chdir("../Du1e-7_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-7     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 
res = res+1
           
os.chdir("../Du1e-6_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-6     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 

res = res+1
           
os.chdir("../Du1e-5_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-5     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 

res = res+1
           
os.chdir("../Du1e-4_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-4     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 

res = res+1
           
os.chdir("../Du1e-3_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-3     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 

res = res+1
           
os.chdir("../Du1e-2_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-2     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 

res = res+1
           
os.chdir("../Du1e-1_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

results_car[res,0] = 1e-1     # Du
results_car[res,1]= 0.       # Dp

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

#----------------------------------------------------- 
#----------------------------------------------------- 

res = res+1

os.chdir("../Du1_Dp0")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1     # Du
results_car[res,1]= 0.       # Dp

results_car[res,2] = 1


#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-10_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-10     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-9_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-9     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-8_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-8     # Du
results_car[res,1]= 1.       # Dp


#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-7_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-7     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-6_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-6     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-5_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-5     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-4_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-4     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-3_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-3     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-2_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-2     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1e-1_Dp1")

df=pd.read_csv('h00249.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = L[i]
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1e-1     # Du
results_car[res,1]= 1.       # Dp

#----------------------------------------------------- 

res = res+1

os.chdir("../Du1_Dp1")

df=pd.read_csv('h00049.jwh',sep='\s+')
df.columns = ["L", "h"]

L = df.loc[:,'L']
L = L.astype(np.float32)
h = df.loc[:,'h']
h = h.astype(np.float32)

len_data = len(L)

for i in range(len(L)):
    if h[i]<=1e-6:
        results_car[res,2] = 0.42
        break

for i in range(len_data):
    L_final[i,res] = L[i]
    h_final[i,res] = h[i]
    
h_max[res] = h[0]

results_car[res,0] = 1     # Du
results_car[res,1]= 1.     # Dp
results_car[res,2] = 0.42

os.chdir("../../")



plt.plot(L_final[:,10],h_final[:,10],color='#1f77b4',linestyle = ':')
plt.plot(L_final[:,9],h_final[:,9],color='#ff7f0e',linestyle = ':')
plt.plot(L_final[:,8],h_final[:,8],color='#2ca02c',linestyle = ':')
plt.plot(L_final[:,7],h_final[:,7],color='#d62728',linestyle = ':')
#
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
display = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)

anyArtist2 = plt.Line2D((0,1),(0,0), color='k')
simArtist2 = plt.Line2D((0,1),(0,0), color='k',marker='',linestyle=':')


plt.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist2,anyArtist2],
          [label for i,label in enumerate(labels) if i in display]+['Cartesian','Axisymmetric'],bbox_to_anchor=(0.98, 0.98),borderaxespad=0.,fontsize=8)
         

plt.xlim([0,1])
plt.ylim([0,10])
plt.xlabel('x/$L_{\mathrm{r}}$ or s/$L_{\mathrm{r}}$')
plt.ylabel('$h/h_{\mathrm{0}}$')
plt.title('ULVZ profiles')
plt.savefig('ULVZ_shape.pdf', bbox_inches='tight',format='pdf')               
plt.show()


# PLOT EVERYTHING!             

x = [1e-10,1]
y_car = [1e-10**(1./4.),1**(1./4.)]
y_ax = [1e-10**(2./7.),1**(2./7.)]

plt.figure(1)
plt.plot(results_car[0:11,0],results_car[0:11,2],'-o',label="Cartesian",color='indianred')
plt.plot(results[0:11,0],results[0:11,2],'-o',label="Axisymmetric",color='steelblue')
plt.plot(x,y_car,color='k',linestyle=':')
plt.plot(x,y_ax,color='k',linestyle=':')

plt.loglog()
plt.xlim([1e-6,1])
plt.ylim([1e-2,1.5])
plt.text(.5,.95,"$\mu_{\mathrm{m}}'=0$",
        horizontalalignment='center',
        transform=ax.transAxes)
plt.xlabel('Topography diffusivity $D_{\mathrm{u}}$')
plt.legend()
plt.ylabel('CMB coverage $\phi_{\mathrm{u}}$')
#plt.title('ULVZ coverage at CMB')
plt.savefig('ULVZ_CMB_cov_dp0.pdf', bbox_inches='tight',format='pdf') 


plt.figure(2)
plt.plot(L_final[:,10],h_final[:,10],label='$D_{\mathrm{u}}$=$1$')
plt.plot(L_final[:,9],h_final[:,9],label='$D_{\mathrm{u}}$=$10^{-1}$')
plt.plot(L_final[:,8],h_final[:,8],label='$D_{\mathrm{u}}$=$10^{-2}$')
plt.plot(L_final[:,7],h_final[:,7],label='$D_{\mathrm{u}}$=$10^{-3}$')
plt.plot(L_final[:,21],h_final[:,21],color='#1f77b4',linestyle = ':')
plt.plot(L_final[:,20],h_final[:,20],color='#ff7f0e',linestyle = ':')
plt.plot(L_final[:,19],h_final[:,19],color='#2ca02c',linestyle = ':')
plt.plot(L_final[:,18],h_final[:,18],color='#d62728',linestyle = ':')
#
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
display = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)

simArtist2 = plt.Line2D((0,1),(0,0), color='k')
anyArtist2 = plt.Line2D((0,1),(0,0), color='k',marker='',linestyle=':')


plt.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist2,anyArtist2],
          [label for i,label in enumerate(labels) if i in display]+['$D_{\mathrm{p}}$=0','$D_{\mathrm{p}}$=1'],bbox_to_anchor=(0.98, 0.98),borderaxespad=0.,fontsize=8)
         

plt.xlim([0,0.6])
plt.ylim([0,7])
plt.xlabel('x/$L_{\mathrm{r}}$ or s/$L_{\mathrm{r}}$')
plt.ylabel('$h/h_{\mathrm{0}}$')
plt.title('Cartesian')
plt.savefig('ULVZ_CMB_cov_cart.pdf', bbox_inches='tight',format='pdf')               
plt.show()


# Curve with slope 1/4
# Curve with slope 2/7

x1 = np.linspace(1e-10,1,100)
x2 = np.linspace(1e-10,1,100)

for i in range(len(x1)):
    x1[i] = 10**(1./4.*x1[i])
    x2[i] = 2./7.*x2[i]

plt.figure(1)
ax=plt.gca()
plt.plot(x,y_car,color='k',linestyle=':')
plt.plot(x,y_ax,color='k',linestyle=':')
plt.plot(results_car[11:22,0],results_car[11:22,2],'-o',color='indianred')#,label="Cartesian")
plt.plot(results[11:22,0],results[11:22,2],'-o',color='steelblue')#label="Axisymmetric"
#plt.legend(title="$\mu_{\mathrm{m}'}$=1")
plt.text(.5,.95,"$\mu_{\mathrm{m}}'=1$",
        horizontalalignment='center',
        transform=ax.transAxes)
plt.loglog()
plt.xlim([1e-6,1.5])
plt.ylim([1e-2,1.5])
plt.xlabel('$D_{\mathrm{u}}$')

#plt.ylabel('CMB coverage $\phi_{\mathrm{u}}$')
plt.savefig('ULVZ_CMB_cov.pdf', bbox_inches='tight',format='pdf') 






