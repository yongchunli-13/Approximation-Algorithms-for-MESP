#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The implementation of the randomized sampling algorithm

import numpy as np
import frank_wolfe
import user
import random
import datetime

# assign local names to functions in the user and frank_wolfe file
frankwolfe  = frank_wolfe.frankwolfe
f = user.f


# Function randrounding needs input \alpha-optimal solution of PC, n and s;
# outputs the solution, objective value and running time
def randrounding(xsol, n, s):
    start = datetime.datetime.now()
    
    np.random.seed(2)

    xsel=[]
    A=0.0
    B=0.0    
    A=calconvolve(xsel,xsol,n,s)
    for c in range(n):
        if sum(xsel)>=s:
            for i in range(n-c):
                xsel.append(0.0)
            break
        
        if c-sum(xsel)>=n-s:
            for i in range(n-c):
                xsel.append(1.0)
            break
        
        num=random.uniform(0, 1)
        
        xsel.append(1)
        B=calconvolve(xsel,xsol,n,s)
        if((xsol[c]*B)/A >=num):
            A=B
        else:
            A=A-xsol[c]*B
            xsel[c]=0.0
            
    end = datetime.datetime.now()
    time = (end-start).seconds
            
    return xsel, f(xsel), time

################################
def calconvolve(xsel,xsol,n,s):
    
    l=len(xsel)
    nz=sum(k>0 for k in xsel)
    value=0.0
    acon=[1,xsol[l]]
    for i in range(n-l-1):
        acon=np.convolve(acon,[1,xsol[1+i+l]])

    value=acon[s-nz]

    return value

# run Frank-Wolfe
#[xsol, supp, primal, ftime]=frankwolfe(n, s, d)  
#[x, fval, time] = randrounding(xsol, n, s)

