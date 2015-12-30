"""
Copyright (c) 2016, Youngmin Park, Bard Ermentrout
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ympark1988@gmail.com


TODO find a way to include noisy parameter(s)
"""

import matplotlib.pylab as mp
import numpy as np
import time as tt

def ESolve(f,x0,t,threshold=None,reset=None,spike_tracking=False,**args):
    """
    solve given SDE (both IF and non-IF)
    inputs are similar to those used in Odeint:
    x: initial condition (can be array or list)
    t: time vector (initial time, final time, steps implicit)
    f: RHS
    args: tuple of arguments (parameters) for RHS of f
    threshold: max membrane potential before reset
    reset: membrane potential reset
    spike_tracking: (for IF models only) if true, return list of length t of 1s and 0s.
    The entries with 1 denote a spike.

    This function assumes the time array is always divided into bins
    of equal length.    
    """
    # generate default args list if empty
    #print args
    if args == {}:
        args = {'args':[]}

    # h
    dt = t[1]-t[0]

    # create solution vector
    x0 = np.array(x0)
    N = len(t)
    Nsol = np.shape(x0)
    
    if Nsol == () or Nsol == 1:
        Nsol = 1
        sol = np.zeros((N,Nsol))


        # initial condition
        sol[0] = x0

        if spike_tracking:
            spikes = np.zeros(N)        
        
        # solve
        time = t[0]
        
        # input to f (RHS) in loop
        inputs = [sol[0],time]
        for j in range(len(args['args'])):
            inputs.append(args['args'][j])
    else:
        Nsol = len(x0)
        sol = np.zeros((N,Nsol))
        sol[0,:]=x0
        #print sol[0,:]
        
        inputs = [x0]
        # initial condition

        if spike_tracking:
            spikes = np.zeros(N)
            
        time = t[0]
        
        # input for f (RHS)
        inputs.append(time)
        for j in range(len(args['args'])):
            inputs.append(args['args'][j])


    
    #inputs = np.array(inputs)
    #print sol[0,:], "before"
    for i in range(1,N):        
        # iterate
        inputs[0]=sol[i-1,:]
        inputs[1]=time
        #print inputs
        
        sol[i,:] = sol[i-1,:]+dt*f(*inputs)
        #print sol[i,:]
        #tt.sleep(3)
        # catch spikes
        if threshold != None and reset != None:
            if np.any(sol[i,:]) >= threshold:
                sol[i,:][sol[i,:]>=threshold] = reset
                if spike_tracking:
                    spikes[i] = 1

        time += dt

    if spike_tracking:
        return sol,spikes
    else:
        return sol
 


def example(x,t,a,b):
    """
    an example right hand side function
    """
    return x*a - t*b

def example2(x,t):
    """
    another example RHS
    """
    return x*(1-x)

def example3(x,t,a,b):
    """
    3d RHS
    """
    return a*x[0]+b*x[1]-x[2]-3*t

def example4():
    pass

def main():
    # an example non-autonomous function
    x0 = .1
    t0=0
    tend=1
    dt=.1
    t = np.linspace(t0,tend,int((tend-t0)/dt))

    # use the same syntax as odeint
    #sol = LSolve(example,x0,t,args=(1,1))
    #sol = ESolve(example2,x0,t)
    #sol = ESolve(example,x0,t,args=(1,1))
    x0 = [1,2,3]
    sol = ESolve(example3,x0,t,args=(1,1))
    
    mp.figure()
    mp.title("Example solution")
    mp.plot(t,sol)

    mp.show()

if __name__ == "__main__":
    main()
