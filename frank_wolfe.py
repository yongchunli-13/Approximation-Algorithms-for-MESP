#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The implementation of the Frank-Wolfe algorithm

import user
import numpy as np
import datetime
import local_search

# assign local names to functions in the user and local_search file
gen_data = user.gen_data
grad = user.grad_fw
localsearch  = local_search.localsearch


# Function frankwolfe needs input n, d and s; outputs the solution, 
# size of its support, upper bound and running time of the Frank-Wolfe algorithm
def frankwolfe(n, d, s): 
    
    start = datetime.datetime.now()
    
    # run local search
    Obj_f, x, X, Xs, ltime = localsearch(n, d, s)
        
    gamma_t = 0.0  
    t = 0.0
    mindual = 1e+10
    dual_gap = 1 # duality gap
    Obj_f = 1 # primal value
    alpha = 1e-4 # target accuracy
    
    while(dual_gap/Obj_f > alpha):
        Obj_f, subgrad, y, dual_gap = grad(x, s, d)
        
        t = t + 1
        gamma_t = 1/(t+2) # step size
        
        x = [(1-gamma_t)*x_i for x_i in x] 
        y = [gamma_t*y_i for y_i in y]
        x = np.add(x,y).tolist() # update x
        mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
        
        print('primal value = ', Obj_f, ', duality gap = ', dual_gap)

        
    supp = (n-x.count(0)) # size of support of output
    
    end = datetime.datetime.now()
    time = (end-start).seconds

    return x, supp, mindual, time




   
