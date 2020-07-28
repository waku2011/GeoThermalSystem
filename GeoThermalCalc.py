#!/usr/bin/env python3
#
#  geothermal system simulation with W-tube
#  All unit are in SI
# 

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTetra, VtkHexahedron, VtkWedge, VtkPyramid

#from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI

import numpy as np
import pandas as pd
import matplotlib as plt
import sys
import time
import math
import csv
import os

# Global params. (water input)
Tin   = 273.0+30.0 # K
Pin   = 101.3e3    # Pa
Win   = 3.5        # kg/s = 210L/min
Rhoin = 1000.0     # kg/m3

# W-tube pipe params. 
# inner 100mm(Downward), outer circular150mm(Upward), thickness both 5mm 
Ri1 = 45.0e-3
Ro1 = 50.0e-3
Ri2 = 70.0e-3
Ro2 = 75.0e-3

rhokPipe = 7800.0
tkPipe   = 20.0 
cpPipe   = 7800.0

# hydraulic diameter of W-tube cross sections 
dD   = 2.0 * Ri1
dU   = 2.0 *(Ri2 - Ro1)

# Misc params.
cg   = 9.81 # m/s2

# thickness of geo-layers, m
Lg1 = 350.0 
Lg2 =1150.0

# depth from surface
Z1  = 0.0      
Z2  = Lg1
Z3  = Lg1+Lg2

RiBC = 0.1  # soil inner boundary radius, m
RoBC = 50.0 # soil outer boundary radius, m
 
# number of cells for each direction
NR =  4   #16 # radius
NT =  8   #32 # tangential 
NZ = 30   # depth
NZ1 = 7
NZ2 =23   # NZ1+NZ2=NZ

nNode = (NR+1)*NT*(NZ+1)  # num. of node
nCell = NR*NT*NZ  # num. of cell
#print(nNode,nCell)

# global initialize ndarrays
Theta = np.zeros(NT  ,dtype=float) # @node
R     = np.zeros(NR+1,dtype=float) # @node
dZ    = np.zeros(NZ  ,dtype=float)

X     = np.zeros((NR+1,NT),dtype=float) # @node
Y     = np.zeros((NR+1,NT),dtype=float) # @node
Z     = np.zeros( NZ+1    ,dtype=float) # @node

ijk2n  = np.zeros((NR+1,NT,NZ+1),dtype=int)
n2ijk  = np.zeros((nNode,3),dtype=int)
nvNode = np.zeros((nCell,8),dtype=int)

kj2iadj = np.zeros((NZ,NT,2),dtype=int)

xyzNode  = np.zeros((nCell,8,3) ,dtype=float)
edgeNode = np.zeros((nCell,12,2),dtype=float)
vol      = np.zeros( nCell      ,dtype=float)
faceArea = np.zeros((nCell,6)   ,dtype=float)
xyzC     = np.zeros((nCell,3)   ,dtype=float)
xyzCf    = np.zeros((nCell,6,3) ,dtype=float)

bftype   = np.zeros((nCell,6)   ,dtype=int) # default 0
voltype  = np.zeros( nCell      ,dtype=int) # default 0

nCellAdj = np.zeros((nCell,6)   ,dtype=int) 
nVecf    = np.zeros((nCell,6,3) ,dtype=float)
dl       = np.zeros((nCell,6)   ,dtype=float)
dln      = np.zeros((nCell,6)   ,dtype=float)
dlf      = np.zeros((nCell,6)   ,dtype=float)
dlnf     = np.zeros((nCell,6)   ,dtype=float)

# Physical parameters
t    = np.full(nCell,300,dtype=float)
tnew = t

# grounda water velocity, m/s
u = np.full(nCell,0.0,dtype=float)
v = np.full(nCell,0.0,dtype=float)
w = np.full(nCell,0.0,dtype=float)

tk  = np.full(nCell,1.50  ,dtype=float)
cp  = np.full(nCell,1000.0 ,dtype=float) 
rho = np.full(nCell,1600.0,dtype=float)
nut = tk/(rho*cp)  # = 1.5/(1000*1600)

# Boudnary flux, W/m2
fluxExt  = np.zeros((nCell,6),dtype=float)

proveXYZ = [0,0,0]
probeTemp = 300.0

timeMax = 3600.0*24*365*20  # 10 years simulation
dtime = 6*3600.0
Tref = 50.0 # deg.C

# double-tube side parameters
# - inner tube is upward flow
# - outer tube is downward flow
AU = math.pi * Ri1**2 # inner tube / upward
AD = math.pi * (Ri2**2-Ro1**2)  # outer tube / downward
UUmean = Win/Rhoin * AU # Upward mean velocity
UDmean = Win/Rhoin * AD # Downward mean velocity

Ttube  = np.zeros(2*NZ)
Ttuben = np.zeros(2*NZ)
Ptube  = np.zeros(2*NZ)
Ztube  = np.zeros(2*NZ)
DZtube = np.zeros(2*NZ)
Vtube  = np.zeros(2*NZ)
Qtube  = np.zeros(2*NZ)
Dtube  = np.zeros(2*NZ)
htube  = np.zeros(2*NZ) 
mutube = np.zeros(2*NZ)
cptube = np.zeros(2*NZ)
Prtube = np.zeros(2*NZ)
tktube = np.zeros(2*NZ)

# work. arrays
netHeattube = np.zeros(2*NZ)
Redtube     = np.zeros(2*NZ)
htctube     = np.zeros((NZ,4))

# geometry and depth
DZtube[0:NZ1]       = Lg1/NZ1
DZtube[NZ1:NZ]      = Lg2/NZ2
DZtube[NZ:NZ+NZ2]   = Lg2/NZ2
DZtube[NZ+NZ2:2*NZ] = Lg1/NZ1
  
Ztube[0] = 0.5 * DZtube[0]
for i in range(1,NZ):
   Ztube[i] = Ztube[i-1] +  0.5 * (DZtube[i-1]+DZtube[i])
for j in range(0,NZ):
   Ztube[j+NZ] = Ztube[NZ-1-j] 

for i in range(NZ):
   Vtube[i] = AD * DZtube[i]
   Vtube[i+NZ] = AU * DZtube[i+NZ]
 
Tg = np.zeros(NZ)
for i in range(NZ):
   if Z[i] < Z2:
     Tg[i] = 273.2 + 30.0+(100.0/350.0)*Z[i]  
   else: 
     Tg[i] = 273.2 + 120.0 

# initilize in tube 
Qtube[:] = 1.0
Ttube[:]  = Tin
Ttuben[:] = Tin
 
for i in range(2*NZ):
   Ptube[i]  = PropsSI('P','T',Ttube[i],'Q',Qtube[i],'Water')
   Dtube[i]  = PropsSI('D','T',Ttube[i],'P',Ptube[i],'Water')
   htube[i]  = PropsSI('H','T',Ttube[i],'P',Ptube[i],'Water')
   mutube[i] = PropsSI('V','T',Ttube[i],'P',Ptube[i],'Water')/Dtube[i] # m2/s
   Prtube[i] = PropsSI('PRANDTL','T',Ttube[i],'P',Ptube[i],'Water')
   tktube[i] = PropsSI('CONDUCTIVITY','T',Ttube[i],'P',Ptube[i],'Water')
   cptube[i] = PropsSI('C','T',Ttube[i],'P',Ptube[i],'Water')

#-----------------  Ground temperature as reference (real data will be required)

def Tg1(z):
   Tsg1 = 20.0+273.2
   dtdz = 100.0/350.0    # 100K/350m
   return Tsg1+dtdz*z
     
def Tg2(z):
   Tsg2 = 120.0+273.2  
   dtdz = 0.0/1000
   return Tsg2+dtdz*(z-Lg1)


def Pg(z):   # static water pressure 
   return Pin + Rhoin * cg * z

# surface heat transfer coeffs. of w-tube
def htc_DB(re, pr, tk, d):
    nud = 0.023 * re**0.8 * pr**0.4 # Dittus-Boelter 
    return (nud*tk/d)

def htc_OUT(z):
    if z < Z1:
      h = 100.0
    else:
      h = 200.0
    return h

#------------------------------------ vector functions rquires np.array
#                                     A, B, C and D

def normalize(A):
    len = norm(A)
    if len < 1e-16:
      return np.array([0,0,0])
    else: 
      return np.array([A[0]/len, A[1]/len, A[2]/len])

def norm(A):
   a0,a1,a2 = A[:]
   return math.sqrt(a0**2+a1**2+a2**2)

def dot_product(A,B):
   a0,a1,a2 = A[:]
   b0,b1,b2 = B[:]
   return  a0*b0+a1*b1+a2*b2

def cross_product(A,B):
   C0 = A[1]*B[2]-A[2]*B[1]
   C1 = A[2]*B[0]-A[0]*B[2]
   C2 = A[0]*B[1]-A[1]*B[0]
   return np.array([C0,C1,C2])

def normal_vec(A,B,C):
  return cross_product(B-A,C-A)

def triArea(A,B,C):
   area = 0.5*norm(cross_product(B-A,C-A))
   return area

def rectArea(A,B,C,D):
   area = triArea(A,B,C)+triArea(A,C,D)
   return area

#------------------------------------------------------------------------------

def makeMesh():
  dRadius = (RoBC-RiBC)/NR  
  for i in range(NR+1):	  
    R[i] = RiBC + dRadius * i  
 
  dTheta = 2.0*math.pi/NT
  for j in range(NT):	  
    Theta[j] = dTheta * j  

  dZ[0:NZ] = (Lg1+Lg2)/NZ
  
  # XYZ node Info.
  for j in range(NT):
    for i in range(NR+1):
      X[i,j] = R[i]*math.cos(Theta[j])
      Y[i,j] = R[i]*math.sin(Theta[j])

  Z[0] = 0.0
  for k in range(NZ):
      Z[k+1] = Z[k] + dZ[k]

  # list to csv
  xyz = []
  for k in range(NZ+1): 
    for j in range(NT):
      for i in range(NR+1):
         xyz.append([X[i,j],Y[i,j],Z[k]])
  pd.DataFrame(data=xyz,index=None, columns=["x","y","z"]).to_csv("xyz.csv",index=None,sep=" ")

  # Mesh Information  
  #-- index: i,j,k to n
  for k in range(NZ+1): 
    for j in range(NT):
      for i in range(NR+1):
         n = NT*(NR+1)*k + (NR+1)*j + i 
         ijk2n[i,j,k] = n
         n2ijk[n,0:3] = [i,j,k]

  #-- index: iv,jv,kv to nv
  for kv in range((NZ+1)-1): 
    for jv in range(NT):
      for iv in range((NR+1)-1):
         nv = NT*NR*kv + NR*jv + iv
        
         jm = jv
         jp = jv+1
         if jv == NT-1: # cyclic
           jp = 0

         nvNode[nv,0] = ijk2n[iv  ,jm,kv]
         nvNode[nv,1] = ijk2n[iv  ,jp,kv]
         nvNode[nv,2] = ijk2n[iv+1,jp,kv]
         nvNode[nv,3] = ijk2n[iv+1,jm,kv]

         nvNode[nv,4] = ijk2n[iv  ,jm,kv+1]
         nvNode[nv,5] = ijk2n[iv  ,jp,kv+1]
         nvNode[nv,6] = ijk2n[iv+1,jp,kv+1]
         nvNode[nv,7] = ijk2n[iv+1,jm,kv+1]

         #print(nv, nvNode[nv,:])         

  #-- Node Info.  
  for kv in range((NZ+1)-1): 
    for jv in range(NT):
      for iv in range((NR+1)-1):
           nv = NT*NR*kv + NR*jv + iv
           #print(nv, nCell)
    
           jm = jv
           jp = jv+1
           if jv == NT-1: # cyclic
             jp = 0

           xyzNode[nv,0] = [X[iv  ,jm] ,Y[iv  ,jm] ,Z[kv]]
           xyzNode[nv,1] = [X[iv  ,jp] ,Y[iv  ,jp] ,Z[kv]]
           xyzNode[nv,2] = [X[iv+1,jp] ,Y[iv+1,jp] ,Z[kv]]
           xyzNode[nv,3] = [X[iv+1,jm] ,Y[iv+1,jm] ,Z[kv]]

           xyzNode[nv,4] = [X[iv  ,jm] ,Y[iv  ,jm] ,Z[kv+1]]
           xyzNode[nv,5] = [X[iv  ,jp] ,Y[iv  ,jp] ,Z[kv+1]]
           xyzNode[nv,6] = [X[iv+1,jp] ,Y[iv+1,jp] ,Z[kv+1]]
           xyzNode[nv,7] = [X[iv+1,jm] ,Y[iv+1,jm] ,Z[kv+1]]

  # list to csv 
  xyz = []
  for i in range(nCell):
     for j in range(8):
        xyz.append([xyzNode[i,j,0],xyzNode[i,j,1],xyzNode[i,j,2]])
  pd.DataFrame(data=xyz,index=None, columns=["x","y","z"]).to_csv("xyzNode.csv",index=None,sep=" ")

  #-- Edge Info.
  for kv in range((NZ+1)-1): 
    for jv in range(NT):
      for iv in range((NR+1)-1):
            nv = NT*NR*kv + NR*jv + iv
        
            jm = jv
            jp = jv+1
            if jv == NT-1: # cyclic
              jp = 0

            # openfoam like edge numbering 
            edgeNode[nv,0,0] = ijk2n[iv,jm,kv]    
            edgeNode[nv,0,1] = ijk2n[iv,jp,kv] 

            edgeNode[nv,1,0] = ijk2n[iv+1,jm,kv] 
            edgeNode[nv,1,1] = ijk2n[iv+1,jp,kv] 

            edgeNode[nv,2,0] = ijk2n[iv+1,jm,kv+1] 
            edgeNode[nv,2,1] = ijk2n[iv+1,jp,kv+1] 
            
            edgeNode[nv,3,0] = ijk2n[iv,jm,kv+1] 
            edgeNode[nv,3,1] = ijk2n[iv,jp,kv+1] 
            
            edgeNode[nv,4,0] = ijk2n[iv,jm,kv] 
            edgeNode[nv,4,1] = ijk2n[iv,jp,kv] 

            edgeNode[nv,5,0] = ijk2n[iv,jm,kv] 
            edgeNode[nv,5,1] = ijk2n[iv+1,jp,kv] 

            edgeNode[nv,6,0] = ijk2n[iv,jm,kv+1] 
            edgeNode[nv,6,1] = ijk2n[iv+1,jp,kv+1] 
            
            edgeNode[nv,7,0] = ijk2n[iv,jm,kv+1] 
            edgeNode[nv,7,1] = ijk2n[iv+1,jm,kv+1] 

            edgeNode[nv,8,0] = ijk2n[iv,jm,kv] 
            edgeNode[nv,8,1] = ijk2n[iv,jm,kv+1] 

            edgeNode[nv,9,0] = ijk2n[iv,jp,kv] 
            edgeNode[nv,9,1] = ijk2n[iv,jp,kv+1] 
            
            edgeNode[nv,10,0] = ijk2n[iv+1,jp,kv]  
            edgeNode[nv,10,1] = ijk2n[iv+1,jp,kv+1] 

            edgeNode[nv,11,0] = ijk2n[iv+1,jm,kv] 
            edgeNode[nv,11,1] = ijk2n[iv+1,jm,kv+1] 
      

  #-- Cell Info.
  for kv in range(NZ): 
    for jv in range(NT):
      for iv in range(NR):
        nv = NT*NR*kv + NR*jv + iv

        ## Volume
        P0 = xyzNode[nv,0,:]
        P1 = xyzNode[nv,1,:]
        P2 = xyzNode[nv,2,:]  
        P3 = xyzNode[nv,3,:]
        areaXY = rectArea(P0,P1,P2,P3)
        vol[nv] = areaXY*dZ[kv]
        
        ## FaceAreas
        faceArea[nv,0] = rectArea(xyzNode[nv,0,:], xyzNode[nv,3,:], xyzNode[nv,2,:], xyzNode[nv,1,:])   # 
        faceArea[nv,1] = rectArea(xyzNode[nv,0,:], xyzNode[nv,1,:], xyzNode[nv,5,:], xyzNode[nv,4,:])   #  
        faceArea[nv,2] = rectArea(xyzNode[nv,1,:], xyzNode[nv,2,:], xyzNode[nv,6,:], xyzNode[nv,5,:])   # 
        faceArea[nv,3] = rectArea(xyzNode[nv,2,:], xyzNode[nv,3,:], xyzNode[nv,7,:], xyzNode[nv,6,:])   # 
        faceArea[nv,4] = rectArea(xyzNode[nv,3,:], xyzNode[nv,0,:], xyzNode[nv,4,:], xyzNode[nv,7,:])   # 
        faceArea[nv,5] = rectArea(xyzNode[nv,4,:], xyzNode[nv,5,:], xyzNode[nv,6,:], xyzNode[nv,7,:])   #  

        ## set bftype
        if kv == 0:
          bftype[nv,0] = 1  # top 
        if kv == NZ-1: 
          bftype[nv,5] = 2  # bottom
        if iv == 0:  
          bftype[nv,1] = 3  # pipeSurface

          kj2iadj[kv,jv,0] = nv
          kj2iadj[kv,jv,1] = 1

        if iv == NR-1:
          bftype[nv,3] = 4  # farField


        ## Cell Center XYZ of Volume
        xyzC[nv,:] = 0.125*( xyzNode[nv,0,:]+xyzNode[nv,1,:]+xyzNode[nv,2,:]+xyzNode[nv,3,:] \
                            +xyzNode[nv,4,:]+xyzNode[nv,5,:]+xyzNode[nv,6,:]+xyzNode[nv,7,:])    
       
        ## Center XYZ of FaceAreas
        xyzCf[nv,0,:] = 0.25*(xyzNode[nv,0,:]+xyzNode[nv,3,:]+xyzNode[nv,2,:]+xyzNode[nv,1,:])  
        xyzCf[nv,1,:] = 0.25*(xyzNode[nv,0,:]+xyzNode[nv,1,:]+xyzNode[nv,5,:]+xyzNode[nv,4,:])  
        xyzCf[nv,2,:] = 0.25*(xyzNode[nv,1,:]+xyzNode[nv,2,:]+xyzNode[nv,6,:]+xyzNode[nv,5,:])  
        xyzCf[nv,3,:] = 0.25*(xyzNode[nv,2,:]+xyzNode[nv,3,:]+xyzNode[nv,7,:]+xyzNode[nv,6,:])  
        xyzCf[nv,4,:] = 0.25*(xyzNode[nv,3,:]+xyzNode[nv,0,:]+xyzNode[nv,4,:]+xyzNode[nv,7,:])  
        xyzCf[nv,5,:] = 0.25*(xyzNode[nv,4,:]+xyzNode[nv,5,:]+xyzNode[nv,6,:]+xyzNode[nv,7,:])  

  xyz = []
  for i in range(nCell):
     for j in range(6):
        xyz.append([xyzCf[i,j,0],xyzCf[i,j,1],xyzCf[i,j,2]])
  pd.DataFrame(data=xyz,index=None, columns=["x","y","z"]).to_csv("xyzCf.csv",index=None,sep=" ")

  xyz = []
  for i in range(nCell):
     xyz.append([xyzC[i,0],xyzC[i,1],xyzC[i,2]])
  pd.DataFrame(data=xyz,index=None, columns=["x","y","z"]).to_csv("xyzC.csv",index=None,sep=" ")

  #-- Adj cell Info. 

  for j in range(6):
     for i in range(nCell): 
        nCellAdj[i,j] = i  # initialize

  ### ToDo: file I/O 
  t1 = time.time()
  
  for j in range(6):
     for i in range(nCell): 
        for l in range(6):
           for k in range(nCell): 
              if (k != i) and (np.dot(xyzCf[i,j,:]-xyzCf[k,l,:],xyzCf[i,j,:]-xyzCf[k,l,:]) <= 1.e-16):
                nCellAdj[i,j] = k
  '''
  for i in range(nCell): 
     for k in range(nCell): 
          if (i == k) or (np.dot(xyzC[i,:]-xyzC[k,:],xyzC[i,:]-xyzC[k,:]) >= 0.1):
            continue 
          else:
            for j in range(6):
               for l in range(6):
                  if np.dot(xyzCf[i,j,:]-xyzCf[k,l,:],xyzCf[i,j,:]-xyzCf[k,l,:]) <= 1e-16 :
                    nCellAdj[i,j] = k
                    break  # exit inner loop
               else:
                  continue
               break  # exit outer loop
  '''
  t2 = time.time()
  print("nCelAdj found !   Elapsed time:", t2-t1)

   
  for i in range(nCell): 
         
        nVecf[i,0,:] = normalize(  normal_vec(xyzNode[i,0,:],xyzNode[i,3,:],xyzCf[i,0,:]) \
                                 + normal_vec(xyzNode[i,3,:],xyzNode[i,2,:],xyzCf[i,0,:]) \
                                 + normal_vec(xyzNode[i,2,:],xyzNode[i,1,:],xyzCf[i,0,:]) \
                                 + normal_vec(xyzNode[i,1,:],xyzNode[i,0,:],xyzCf[i,0,:]) ) 

        nVecf[i,1,:] = normalize(  normal_vec(xyzNode[i,0,:],xyzNode[i,1,:],xyzCf[i,1,:]) \
                                 + normal_vec(xyzNode[i,1,:],xyzNode[i,5,:],xyzCf[i,1,:]) \
                                 + normal_vec(xyzNode[i,5,:],xyzNode[i,4,:],xyzCf[i,1,:]) \
                                 + normal_vec(xyzNode[i,4,:],xyzNode[i,0,:],xyzCf[i,1,:]) ) 

        nVecf[i,2,:] = normalize(  normal_vec(xyzNode[i,1,:],xyzNode[i,2,:],xyzCf[i,2,:]) \
                                 + normal_vec(xyzNode[i,2,:],xyzNode[i,6,:],xyzCf[i,2,:]) \
                                 + normal_vec(xyzNode[i,6,:],xyzNode[i,5,:],xyzCf[i,2,:]) \
                                 + normal_vec(xyzNode[i,5,:],xyzNode[i,1,:],xyzCf[i,2,:]) ) 

        nVecf[i,3,:] = normalize(  normal_vec(xyzNode[i,2,:],xyzNode[i,3,:],xyzCf[i,3,:]) \
                                 + normal_vec(xyzNode[i,3,:],xyzNode[i,7,:],xyzCf[i,3,:]) \
                                 + normal_vec(xyzNode[i,7,:],xyzNode[i,6,:],xyzCf[i,3,:]) \
                                 + normal_vec(xyzNode[i,6,:],xyzNode[i,2,:],xyzCf[i,3,:]) ) 

        nVecf[i,4,:] = normalize(  normal_vec(xyzNode[i,3,:],xyzNode[i,0,:],xyzCf[i,4,:]) \
                                 + normal_vec(xyzNode[i,0,:],xyzNode[i,4,:],xyzCf[i,4,:]) \
                                 + normal_vec(xyzNode[i,4,:],xyzNode[i,7,:],xyzCf[i,4,:]) \
                                 + normal_vec(xyzNode[i,7,:],xyzNode[i,3,:],xyzCf[i,4,:]) ) 

        nVecf[i,5,:] = normalize(  normal_vec(xyzNode[i,4,:],xyzNode[i,5,:],xyzCf[i,5,:]) \
                                 + normal_vec(xyzNode[i,5,:],xyzNode[i,6,:],xyzCf[i,5,:]) \
                                 + normal_vec(xyzNode[i,6,:],xyzNode[i,7,:],xyzCf[i,5,:]) \
                                 + normal_vec(xyzNode[i,7,:],xyzNode[i,4,:],xyzCf[i,5,:]) ) 
 
  for j in range(6):
     for i in range(nCell): 
        
        dl   [i,j] = norm(xyzC[nCellAdj[i,j],:]-xyzC[i,:])
        dln  [i,j] = abs(dot_product(xyzC[nCellAdj[i,j],:]-xyzC[i,:], nVecf[i,j,:]))
        dlf  [i,j] = norm(xyzCf[i,j,:]-xyzC[i,:])
        dlnf [i,j] = abs(dot_product(xyzCf[i,j,:]-xyzC[i,:], nVecf[i,j,:]))
        if dlf[i,j] <= 0:
            print("dlf",nCellAdj[i,j],i,j, bftype[i,j]) 
            print(xyzC[nCellAdj[i,j],:]-xyzC[i,:])
            print(nVecf[i,j,:])
        if dlnf[i,j] <= 0:
            print("dlnf", nCellAdj[i,j],i,j, bftype[i,j]) 
            print(xyzC[nCellAdj[i,j],:]-xyzC[i,:])
            print(nVecf[i,j,:])

#------------------------------------------------------------------------------

def exportMesh():   # export VTU (unstructured VTK) mesh via PyEVTK 
  # Define vertices
  x = np.zeros(nNode)
  y = np.zeros(nNode)
  z = np.zeros(nNode)
  for n in range(nNode):
    i = n2ijk[n,0]
    j = n2ijk[n,1]
    k = n2ijk[n,2]
    x[n], y[n], z[n] = X[i,j], Y[i,j], Z[k]
  #### list to csv
  xyz = []
  for n in range(nNode):
     xyz.append([x[n],y[n],z[n]])
  pd.DataFrame(data=xyz,index=None, columns=["x","y","z"]).to_csv("xyz2.csv",index=None)
  ####

  # Define connectivity or vertices that belongs to each element
  nCellConn = 8*nCell
  conn = np.zeros(nCellConn)
  nv = 0
  for n in range(0,nCellConn,8):
     conn[n:n+8] = nvNode[nv,0:8]
     nv = nv + 1
  #print(conn)

  # Define offset of last vertex of each cell (assuming all Hexahedron)
  offset = np.zeros(nCell)
  for nv in range(nCell):
    offset[nv] = 8*(nv+1)
  #print(offset[:])
  
  # Define cell types
  ctype = np.zeros(nCell)
  ctype[:] = VtkHexahedron.tid
  
  # some works
  bftypeMax = np.zeros(nCell)
  bftypeMax[:] = np.max(bftype, axis=1)
  field_data = {"test_fd": np.array([1.0, 2.0])}

  unstructuredGridToVTK(
                       "unstructured_mesh",
                       x, y, z, 
                       connectivity = conn,
                       offsets = offset, 
                       cell_types = ctype,
                       cellData = {"temp": t[:], "bftypeMax": bftypeMax[:]},   
                       pointData = None, 
                       #fieldData = field_data,
                       )

#------------------------------------------------------------------------------

def updateTemp(probeTemp):
  
  # interface velocity, uufn 
  uuf = np.zeros((nCell,6,3))
  for j in range(6):
     for i in range(nCell):
        uuf[i,j,0] = 0.5*(u[i]+u[nCellAdj[i,j]])
        uuf[i,j,1] = 0.5*(v[i]+v[nCellAdj[i,j]])
        uuf[i,j,2] = 0.5*(w[i]+w[nCellAdj[i,j]])

  uufn = np.zeros((nCell,6))
  for j in range(6):
     for i in range(nCell):
        uufn[i,j] = nVecf[i,j,0]*uuf[i,j,0] \
                  + nVecf[i,j,1]*uuf[i,j,1] \
                  + nVecf[i,j,2]*uuf[i,j,2] 

  # interface temperature, tf
  tf = np.zeros((nCell,6))
  for j in range(6):
    for i in range(nCell):
        if bftype[i,j]   == 1:         # top surface (insulated)
          tf[i,j] = t[i]
        elif bftype[i,j] == 2:         # bottom surface (fixed temp.)
          tf[i,j] = 273.0 + 120.0                                
        elif bftype[i,j] == 3:         # solid/interface boundary (insulated) 
                                       # (Pipe<->Soil heat flux is externally defined)
          tf[i,j] = t[i]                             
        elif bftype[i,j] == 4:         # farfield boundary (depth dependent)
          if xyzC[i,2] <= Z2:
             tf[i,j] = Tg1(xyzC[i,2])
          else:
             tf[i,j] = Tg2(xyzC[i,2])
        else:                          # temperature (simple upwind interpolation)
          if uufn[i,j] >= 0.0 : 
            tf[i,j] = t[i]
          else:
            tf[i,j] = t[nCellAdj[i,j]]

  # make surface fluxes, W/m2 (advective + diffusive flux + additional boundary flux)
  fluxf = np.zeros((nCell,6))
  for j in range(6):
    for i in range(nCell):
       if 1<= bftype[i,j] <= 4:       # boundary surfaces
         tkf = 0.5*(tk[i]+tk[nCellAdj[i,j]])  
         fluxf[i,j] = - rho[i]*cp[i]*uufn[i,j]*tf[i,j] + tkf*(tf[i,j]-t[i])/dlnf[i,j] + fluxExt[i,j]
       else:                          # temperature (simple upwind interpolation)
         tkf = 0.5*(tk[i]+tk[nCellAdj[i,j]])  
         fluxf[i,j] = - rho[i]*cp[i]*uufn[i,j]*tf[i,j] + tkf*(t[nCellAdj[i,j]]-t[i])/dln[i,j] 

  # make body flux, W 
  fluxb = np.zeros(nCell)
  for i in range(nCell):
     fluxb[i] = 0.0

  # RHS of thermodynamic equation
  RHS = np.zeros(nCell,dtype=float)
  for i in range(nCell):
    RHS[i] = 0.0
    for j in range(6):
       RHS[i] = RHS[i] + fluxf[i,j]*faceArea[i,j]
    RHS[i] = RHS[i] + fluxb[i]
    RHS[i] = RHS[i]/(rho[i]*cp[i]*vol[i]) # K/s

  # Temperature update
  for i in range(nCell):
    tnew[i] = t[i] + RHS[i]*dtime
  for i in range(nCell):
    t[i] = tnew[i]

#------------------------------------------------------------------------------

def updateHeatFlux():    # double tube system (outer is downward/innner is upward)

     # Local Reynolds number and Prandtl numbers
     for k in range(0,NZ): 
        Redtube[k] = UDmean * dD / mutube[k]   
     for k in range(NZ,2*NZ): 
        Redtube[k] = UUmean * dU / mutube[k]
     print(UDmean, UUmean, dD, dU, mutube[1], np.ndarray.max(Redtube))

     # update boundary heat transfer coeffcients  
     for k in range(0,NZ): 
        htctube[k,0] = htc_DB(Redtube[k], Prtube[k], tktube[k], dD)
        htctube[k,1] = htc_DB(Redtube[2*NZ-1-k], Prtube[2*NZ-1-k], tktube[2*NZ-1-k], dU)
        htctube[k,2] = htc_DB(Redtube[2*NZ-1-k], Prtube[2*NZ-1-k], tktube[2*NZ-1-k], dU)
        htctube[k,3] = htc_OUT(Z[k])

     # update input heat flux, W
     ## inner tube heat budget, W
     for k in range(0,NZ):
        Regist = 1./(htctube[k,0]*Ri1)+math.log(Ro1/Ri1)/tkPipe + 1./(htctube[k,1]*Ro1)
        hf_r   = 2 * math.pi * DZtube[k] / Regist * (Ttube[k] - Ttube[2*NZ-1-k])
       
        if k == 0:
          hf_in  = UDmean * (Dtube[k]*cptube[k]*Tin) * AD 
        else:
          hf_in  = UDmean * (Dtube[k-1]*cptube[k-1]*Ttube[k-1]) * AD 
        hf_out =  UDmean * (Dtube[k]*cptube[k]*Ttube[k]) * AD   
        netHeattube[k] = hf_in - hf_out - hf_r 
     ## outer tube heat budget  
     for k in range(NZ,2*NZ):
     
        kk = 2*NZ-1-k

        Registi = 1./(htctube[kk,0]*Ri1)+math.log(Ro1/Ri1)/tkPipe + 1./(htctube[kk,1]*Ro1)
        Registo = 1./(htctube[kk,2]*Ri2)+math.log(Ro2/Ri2)/tkPipe + 1./(htctube[kk,3]*Ro2)
        hf_ri   = 2 * math.pi * DZtube[k] / Registi * (Ttube[k] - Ttube[2*NZ-1-k])
 
        hf_ro   = 0.0 
        for j in range(NT):
          iadj = kj2iadj[kk,j,0]
          jadj = kj2iadj[kk,j,1]
          areaWeight = faceArea[iadj,jadj]/(2*math.pi*Ro2 * DZtube[k])
          fluxExt[iadj,jadj] = 2 * math.pi * DZtube[k] / Registo * (Ttube[k] - t[iadj]) * areaWeight
          hf_ro = hf_ro + fluxExt[iadj,jadj]
        
        hf_in  =  UUmean * (Dtube[k-1]*cptube[k-1]*Ttube[k-1]) * AU
        hf_out =  UUmean * (Dtube[k]*cptube[k]*Ttube[k]) * AU   
        netHeattube[k] = hf_in - hf_out + hf_ri - hf_ro 

     # update temperature (Euler) and pressure
     for k in range(2*NZ):
        Ttuben[k] = Ttube[k] + netHeattube[k]/(Dtube[k]*cptube[k]*Vtube[k]) * dtime 

        Ptube[k]  = 101326.50 #PropsSI('P','T',Ttuben[i],'Q',Qtube[i],'Water')
        Dtube[k]  = 1000.0    #PropsSI('D','T',Ttuben[i],'P',Ptube[i],'Water')
        htube[k]  = 335.0e3   #PropsSI('H','T',Ttuben[i],'P',Ptube[i],'Water')
        mutube[k] = 1e-6      #PropsSI('V','T',Ttuben[i],'P',Ptube[i],'Water')/Dtube[i] # m2/s
        Prtube[k] = 3.0       #PropsSI('PRANDTL','T',Ttuben[i],'P',Ptube[i],'Water')
        tktube[k] = 0.56      #PropsSI('CONDUCTIVITY','T',Ttuben[i],'P',Ptube[i],'Water')
        cptube[k] = 4190.0    #PropsSI('C','T',Ttuben[i],'P',Ptube[i],'Water')

     Ttube[:] = Ttuben[:]

     # calc. thermal budget
     TotalWatt = 0.0 
     for k in range(NZ,2*NZ):
        kk = 2*NZ-1-k 
        for j in range(NT):
          iadj = kj2iadj[kk,j,0]
          jadj = kj2iadj[kk,j,1]  
          # minus means outward flux (e.g.,inward ground) "fluxExt" is positive.
          TotalWatt = TotalWatt - fluxExt[iadj,jadj] * faceArea[iadj,jadj]

     return TotalWatt

#------------------------------------------------------------------------------

   
def main():

  if os.path.exists("output.csv"): 
    os.remove("output.csv")
  
  with open("output.csv", mode='a') as f:
    writer=csv.writer(f,delimiter=" ")
        
    makeMesh()

    # init ground temperature
    for i in range(nCell):
       if xyzC[i,2] <= Z2:
           t[i] = Tg1(xyzC[i,2])
       else:
           t[i] = Tg2(xyzC[i,2])

    # total ground volume
    totalVol = 0.0
    for i in range(nCell):
      totalVol = totalVol + vol[i]


    time = 0.0
    itime = 0
    while time < timeMax:

      time = time + dtime
      itime = itime + 1

      TotalWatt = updateHeatFlux()

      updateTemp(probeTemp)

      # calc. mean ground temp.
      meanT = 0.0     
      for i in range(nCell):
        meanT = meanT + t[i]*vol[i]
      meanT = meanT/totalVol  

      print("time=",time, tnew.max(), tnew.min(), meanT, TotalWatt, Ttube[2*NZ-1], Ttube[NZ-1]) 
      writer.writerow([time,meanT, TotalWatt, Ttube[2*NZ-1], Ttube[NZ-1]])
    
      if itime % 10000 == 0 : 
         exportMesh()

if __name__ == "__main__":
    main()
    
