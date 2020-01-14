#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The user file includes the data preprocessing and function definitions

import os
import numpy as np
import pandas as pd
from math import log
from numpy import matrix
from numpy import array
from scipy.linalg import svd

## Data preprocessing
def gen_data(n):
    global C
    global V
    global Vsquare
    global E
    
    ## input the matrix C
    C = pd.read_table(os.getcwd()+'/Matrix2000.txt',
                         header=None,encoding = 'utf-8',sep=',')  
    
##    C = pd.read_table(os.getcwd()+'/data90.txt',
##                         header=None,encoding = 'utf-8',sep='\s+')
#    
#        
#    C = pd.read_table(os.getcwd()+'/Data124.dms',
#                         header=None,encoding = 'utf-8',sep='\s+')
#    C = array(C)
#    C = C.reshape(n,n)
    
    C = matrix(C)
    
    ## size of the problem
    n = C.shape[0]
    d = np.linalg.matrix_rank(C)  
    print('The size of matrix C is',n)
    print('The rank of matrix C is',d)
    
    ## Cholesky factorization of C    
    U, s, V = svd(C) # SVD decomposition
    s[s<1e-6]=0 
    
    sqrt_eigen = [0]*n
    for i in range(n):
        if s[i]>0:
            sqrt_eigen[i] = np.sqrt(s[i])
        else:
            sqrt_eigen[i] = 0
               
    V = np.dot(np.diag(sqrt_eigen),V)    
    V = V[0:d,:]
    
    V = matrix(V)
    V = V.T #size of V is n*d
    
    Vsquare = [[V[i].T * V[i] for i in range(n)]] # V[i] is the row vector
    Vsquare = Vsquare[0] # each element of S is matrix V[i].T*V[i] of size d*d
    
    E=np.eye(d, dtype=int) # identity matrix 
    
 
## The objective function of MESP
def f(x):     
    val = 0.0   
    sel = np.flatnonzero(x) 
    
    for i in sel:
        val = val + Vsquare[i]
        
    r = np.linalg.matrix_rank(val)     
    [a,b] = np.linalg.eigh(val)
    a = a.real # engivalues
    b = b.real # engivectors
        
    sorted_a = sorted(a, reverse=True) # sort eigenvalues in a decreasing order
    
    f = 1.0
    for i in range(r):
        f = f * sorted_a[i]
        
    if f <= 0:
        print("Singular matrix")
        return 0
        
    return log(f) 

# Update the inverse matrix by adding a rank-one matrix
def upd_inv_add(X, Xs, opti):    
    Y = 0.0
    Ya = 0.0
    Yb = 0.0
    Yc = 0.0
    Ys = 0.0
    
    Y = X + V[opti].T*V[opti]
    
    Ya = Xs*V[opti].T   
    Yb = (E-Xs*X)*V[opti].T
    Yc = 1/(V[opti]*Yb)[0,0]
    Ys = Xs - Yc*(Ya*Yb.T)-Yc*(Yb*Ya.T)+(Yc**2)*(1+(V[opti]*Ya)[0,0])*(np.dot(Yb,Yb.T))
    
    return Y, Ys

# Update the inverse matrix by minusing a rank-one matrix
def upd_inv_minus(X, Xs, opti):    
    Y = 0.0
    Ya = 0.0
    Yb = 0.0
    Yc = 0.0
    Ys = 0.0
    
    Y = X - V[opti].T*V[opti]
    
    Ya = Xs*V[opti].T
    Yb = Xs*Xs*V[opti].T
    Yc = 1/(Ya.T*Ya)[0,0] 
    Ys = Xs - Yc*(Ya*Yb.T)-Yc*(Yb*Ya.T)+(Yc**2)*(Yb.T*Ya)[0,0]*(Ya*Ya.T)  
    
    return Y, Ys


## The rank one update for greedy
def srankone(X, Xs, indexN, n, val):   
    opti = 0.0
    Y = 0.0
    
    temp = V*(E-Xs*X)*V.T # determinant
    xval = temp.diagonal()
    xval = xval.reshape(n,1)
    xval = list(xval)
    
    maxf=0.0
    opti=0 # add opti to set S
    
    for j in indexN:
        if(xval[j] > maxf):
            maxf = xval[j]
            opti = j
    
    Y,Ys = upd_inv_add(X,Xs,opti) # update X and Xs
    val = val*maxf # update the objective value
    
    return Y,Ys,opti,val 


## The rank one update for local search
def findopt(X, Xs, i, indexN,n,val):
    Y=0.0
    Ys=0.0
    
    Y, Ys = upd_inv_minus(X,Xs,i)
    
    temp = V*(E-Y*Ys)*V.T
    xval = temp.diagonal()
    xval = xval.reshape(n,1)
    xval = list(xval)
    
    maxf=0.0
    opti=0
    
    for j in indexN:
        if(xval[j]>maxf):
            maxf = xval[j]
            opti = j
    
    val = val+ log(maxf)-log(xval[i])
       
    return Y, Ys, opti, val



#### Compute upper bound and duality gap for Frank-Wolfe ####
   
def find_k(x,s,d):
    for i in range(s-1):
        k = i
        mid_val = sum(x[j] for j in range(k+1,d))/(s-k-1)
        if mid_val >= x[k+1]-1e-10 and mid_val < x[k]+1e-10:        
            return k, mid_val
        

def grad_fw(x,s,d):
    nx = len(x)
    
    val = 0.0
    
    sel=np.flatnonzero(x) 
    
    for i in sel:
        val = val+x[i]*Vsquare[i]
        
    [a,b] = np.linalg.eigh(val) 
    a = a.real # engivalues
    b = b.real # engivectors

    sorted_a = sorted(a, reverse=True)     
    k,nu = find_k(sorted_a,s,d)
 
    engi_val = [0]*d
    
    for i in range(d):
        if(a[i] > nu):
            engi_val[i] = 1/a[i]
        else:
            if nu > 0:
                engi_val[i] = 1/nu
            else:
                engi_val[i] = 1/sorted_a[k]
    W = b*np.diag(engi_val)*b.T # supgradient

    val = 0.0
    subg = [0.0]*nx
    
    temp = V*W*V.T
    val = temp.diagonal()
    val = val.reshape(nx,1)   
    val = list(val)
    
    for i in range(nx):
        subg[i] = val[i][0,0] # supgradient at x
        
    temp = np.array(subg)
    sindex = np.argsort(-temp)   
    y = [0]*nx
    for i in range(s):
        y[sindex[i]] = 1 # linear oracle

    
    nu = subg[sindex[s-1]]
    mu = [0]*nx
    for i in range(s):
        mu[sindex[i]] = subg[sindex[i]]- nu # duality gap
        
    f = 1    
    engi_val = sorted(engi_val)
    for i in range(s):
        f = f*engi_val[i] # the objective value of PC
        
    return -log(f), subg, y, s*nu+sum(mu)-s
        
 
 
