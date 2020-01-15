#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Implementation of Frank-Wolfe, local search and sampling Algorithms on the instance of n=124

import pandas as pd
import numpy as np
import local_search
import frank_wolfe
import randomized

# assign local names
localsearch  = local_search.localsearch
frankwolfe  = frank_wolfe.frankwolfe
sampling = randomized.sampling

# parameters
n = 124
d = 124

# Local Search Algorithm
loc = 0
df = pd.DataFrame(columns=('n','d', 's', 'objective value', 'time'))

for s in range(20,40,10): # set the values of s
    print("This is case ", loc+1)
    fval, xsol, time  = localsearch(n ,d, s) 
    df.loc[loc] = np.array([n, d, s, fval, time])
    loc = loc+1  


# Frank-Wolfe Algorithm
loc = 0
df = pd.DataFrame(columns=('n','d', 's','supp', 'upper bound', 'time'))

for s in range(20,40,10): # set the values of s
    print("This is case ", loc+1)
    x, supp, mindual, time  = frankwolfe(n ,d, s) 
    df.loc[loc] = np.array([n, d, s, supp, mindual, time])
    loc = loc+1  

# Sampling Algorithm
loc = 0
df = pd.DataFrame(columns=('n','d', 's', 'objective value', 'time'))

N = 100 # the number of repetitions for sampling 
for s in range(20,40,10): # set the values of s
    print("This is case ", loc+1)
    fval, xsol, time  = sampling(n,d, s, N) 
    df.loc[loc] = np.array([n, d, s, fval, time])
    loc = loc+1  
