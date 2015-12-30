"""
Copyright (c) 2016, Youngmin Park, Bard Ermentrout
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Youngmin Park ympark1988@gmail.com

This script file will be to explore the coupled Lambda-omega system 
as discussed in our Monday meeting 5/19/2014

As of 9/18/14, I will use this script to explore and confirm some of our new results

TODO: (9/26/2014) fixed phi_experiment. Fix phi_theory.

"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as mp

import scipy
from scipy import integrate
from langevin_1D import LSolve
#from euler_1D import ESolve
from euler import *
import time

def lambda_omega_uncoupled(u,t,a,q):
    """
    u: first and second coordinates
    t: time
    a: input current
    q: isochron "twist"

    One lambda-omega system:
    \begin{equation}
    \dot{x} = x \lambda(r) - \omega(r) y,\\
    \dot{y} = y \lambda(r) + \omega(r) x,
    \end{equation}
    where $\lambda(r) = a - r^2$ and $\omega(r) = 1+qr^2$
    
    """
    r = np.sqrt(u[0]**2 + u[1]**2)
    lambdar = r*(1 - r**2)
    #omegar = 1 + q*r**2
    #omegar = 1 + q*(1-r**2)
    omegar = 1 + q*(r**2-1)

    return [u[0]*lambdar - omegar*u[1],
            u[1]*lambdar + omegar*u[0]]


def lamom_coupled(u,t,a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile=None):
    """
    New lambda_omega_coupled function
    same function as lambda_omega_coupled, but with q time-varying
    y: first through fourth coordinates (in order of x, y, \hat{x}, \hat{y})
    t: time
    a: input current
    q: a function of q0,q1,f,eps,t
    alpha: coupling constant
    beta: coupling constant
    q0,q1,c: parameters for function q
    
    Two coupled lambda omega systems:
    \begin{equation}
    \left ( \begin{array}{c}
    \cdot{x} \\ \cdot{y}
    \end{array} \right ) = 
    \left ( \begin{array}{c}
    \cdot{x} \\ \cdot{y}
    \end{array} \right ) + 
    \left ( \begin{array}{cc}
    \alpha & -\beta \\ \beta & \alpha
    \end{array} \right )
    \left ( \begin{array}{c}
    \hat{x} - x \\ \hat{y} - y
    \end{array} \right )

    \end{equation}
    
    where $\hat{x}$ and $\hat{y}$ are the same equation. However,
    the vector in the coupling term is as

    \begin{equation}
    \left ( \begin{array}{c}
    x - \hat{x} \\ y - \hat{y}
    \end{array} \right )
    \end{equation}    
    """

    # assign readable variables
    x=u[0];y=u[1]
    xhat=u[2];yhat=u[3]
    
    # uncoupled terms
    q = Q(t,eps,q0,q1,f,dt,partype,noisefile)
    #q2 = Q(t,eps,q0,q1,f,dt,partype,noisefile)

    #q = q0 + q1*np.cos(c*eps*t)
    def f(x,y,t,a,q):
        return lambda_omega_uncoupled([x,y],t,a,q)[0]
    def g(x,y,t,a,q):
        return lambda_omega_uncoupled([x,y],t,a,q)[1]

    # coupling matrix
    cmat = np.array([[alpha, -beta],[beta,alpha]])

    # coupling vector
    cvec = np.array([xhat-x,yhat-y])

    # full equations
    #dot = np.transpose(np.array([f(x,y,t,a,q),g(x,y,t,a,q)])) + \
    #      eps*np.dot(cmat,cvec)
    #(xdot,ydot) = (dot[0][0],dot[0][1])

    #hatdot = np.transpose(np.array([f(xhat,yhat,t,a,q),g(xhat,yhat,t,a,q)])) + \
    #         eps*np.dot(cmat,-cvec)
    #(xhatdot,yhatdot) = (hatdot[0][0],hatdot[0][1])

    (xdot,ydot) = np.array([f(x,y,t,a,q),g(x,y,t,a,q)]) + \
        eps*np.dot(cmat,cvec)
     
    (xhatdot,yhatdot) = np.array([f(xhat,yhat,t,a,q),g(xhat,yhat,t,a,q)]) + \
        eps*np.dot(cmat,-cvec)

    # return right hand side

    return np.array([xdot,ydot,xhatdot,yhatdot])

def Q(t,eps,q0,q1,f,dt,partype,noisefile):
    """
    the q parameter, now a function of time:
    Q(t) = q0+q1 cos(c eps t)
    t: time
    q0,q1,c: par
    eps: coupling strength
    """
    if partype=='p':
        # periodic
        return q0 + q1*np.cos(f*eps*t)
    elif partype=='s':
        # stochastic
        assert(noisefile != None)
        N = noisefile[0]
        t0 = noisefile[1]
        tend = noisefile[2]
        idx = int(N*t/(tend-t0))+3
        #print 't:',t,'idx:',idx,'N:',N,'t0:',t0,'tend:',tend
        #time.sleep(5)
        q = q0 + q1*noisefile[idx]
        return q
    elif partype=='qp':
        # quasi-periodic
        q = q0+(q1/2.)*(np.cos(eps*f*t)+np.cos(np.sqrt(2)*eps*f*t))
        return q
        #q = q0+(q1-q0)*np.cos(eps*f*t)
        #print 'WARNING: QP UNDEFINED'
        #return None
    elif partype == 'c':
        # constant
        return q0 + q1

def Hoddapprox(phi,t,a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile=None):
    """
    WLOG radius=1
    alpha,beta:params
    q: tau-dependent param
    phi: phase difference
    """

    #q = q0 + q1*np.cos(c*eps*t)
    q = Q(t,eps,q0,q1,f,dt,partype,noisefile)

    #phi = (1+a*q)*t
    #print (alpha-q*beta),np.sin(phi*(1.+a*q))
    #return eps*( -2.*np.sqrt(a)*(alpha-q*beta)*np.sin(phi*(1.+a*q)) )
    
    # this one is right
    b1 = 0.50178626*q -0.49707682
    b2 = -0.00106664*q + 0.00105652
    #b1 = 0.09642291*(q)**3 -0.44689531*(q)**2+  0.84727802*(q) -0.48798273
    #b2 = -0.001055*(q-1)
    return 4*eps*(b1*np.sin(phi) + b2*np.sin(2*phi))

def Hodd(phi,t,a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile=None):
    """
    WLOG radius=1
    alpha,beta:params
    q: tau-dependent param
    phi: phase difference
    """

    #q = q0 + q1*np.cos(c*eps*t)
    q = Q(t,eps,q0,q1,f,dt,partype,noisefile)

    # what is this
    #phi = (1+a*q)*t
    #print (alpha-q*beta),np.sin(phi*(1.+a*q))
    #return eps*( -2.*np.sqrt(a)*(alpha-q*beta)*np.sin(phi*(1.+a*q)) )
    
    # return RHS
    return eps*( 2.*(beta*q-1.)*np.sin(phi) )
    


def OU(z,t,mu):
    """
    OU noise RHS
    """
    return -z*mu

def main():
    alpha=1.;beta=1.;eps=.0025
    noisefile=None;filenum=4
    a=1.;dt=.05#;q=.1

    # use one period for sample plots
    TruePeriod = 2*np.pi#/(1+q*a)

    q0=1.1;q1=1 # fixed
    partype = 'p' # choose s,p,qp,c
    # always use XPP's .tab files!!
    if partype == 'p':
        # periodic
        f=1
        print 'q0='+str(q0)+' q1='+str(q1)+' f='+str(f)
        total = TruePeriod*2000
        t = np.linspace(0,total,total/dt)
    elif partype == 's':
        # stochastic
        f=1
        print 'q0='+str(q0)+' q1='+str(q1)+' f='+str(f)
        noisefile = np.loadtxt("ounormed"+str(filenum)+"_mu1k.tab")
        total = noisefile[2]
        t = np.linspace(0,total,total/dt)
        print "Assuming noise data generated with mu=1000"
        print "Data mean for seed="+str(filenum)+": "+str(np.mean(noisefile[3:]))
    elif partype == 'qp':
        # quasi-periodic
        f=1
        print 'q0='+str(q0)+' q1='+str(q1)+' f='+str(f)
        total = TruePeriod*2000
        t = np.linspace(0,total,total/dt)
    elif partype == 'c':
        f=1
        print 'q0='+str(q0)+' q1='+str(q1)+' f='+str(f)
        total = TruePeriod*2000
        t = np.linspace(0,total,total/dt)

    # analytic initial condition
    #initc = [np.sqrt(a),0,-2/np.sqrt(2),2/np.sqrt(2)]
    initc = [2/np.sqrt(2),2/np.sqrt(2),-2/np.sqrt(2),2/np.sqrt(2)]

    # plot sample trajectory of coupled system


    # get coupled lamom
    #lcsolcoupled = integrate.odeint(lamom_coupled,
    #                                initc,t,args=(a,alpha,beta,eps,q0,q1,c,dt,partype,noisefile))
    if partype == 's':
        lcsolcoupled = ESolve(lamom_coupled,initc,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))
        
        phi1init = np.arctan2(initc[1],initc[0])
        phi2init = np.arctan2(initc[3],initc[2])
        # compute Hodd
        # get theory phase
        phi_theory = ESolve(Hodd,
                            phi2init-phi1init,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))

    elif partype == 'p' or partype == 'qp' or partype == 'c':
        lcsolcoupled = integrate.odeint(lamom_coupled,initc,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))
        
        phi1init = np.arctan2(initc[1],initc[0])
        phi2init = np.arctan2(initc[3],initc[2])
        # compute Hodd
        # get theory phase
        phi_theory = integrate.odeint(Hodd,
                            phi2init-phi1init,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))


    """
    mp.figure()
    p1, = mp.plot(lcsolcoupled[:,0],lcsolcoupled[:,1])
    p2, = mp.plot(lcsolcoupled[:,2],lcsolcoupled[:,3])
    mp.title('Coupled lambda-omega with a='+str(a)+',q='+str(q)+',alpha='+str(alpha)+',beta='+str(beta)+',eps='+str(eps))
    mp.xlabel('x or xhat')
    mp.ylabel('y or yhat')

    mp.legend([p1,p2],['(x,y)','(xhat,yhat)'])
    """

    """
    mp.figure()
    p1, = mp.plot(t,lcsolcoupled[:,0])
    p2, = mp.plot(t,lcsolcoupled[:,2])
    mp.title('Coupled lambda-omega with a='+str(a)+',q='+str(q)+',alpha='+str(alpha)+',beta='+str(beta)+',eps='+str(eps))
    mp.xlabel('t')
    mp.ylabel('x and xhat')

    mp.legend([p1,p2],['x','xhat'])
    """
    mp.figure()
    theta1 = np.arctan2(lcsolcoupled[:,1],lcsolcoupled[:,0])
    theta2 = np.arctan2(lcsolcoupled[:,3],lcsolcoupled[:,2])
    phi_exp = np.mod(theta2-theta1+np.pi,2*np.pi)-np.pi
    phi_theory  = np.mod(phi_theory+np.pi,2*np.pi)-np.pi
    #corl = scipy.stats.pearsonr(phi_exp,phi_theory)[0]
    #mp.plot(t,theta1)
    #mp.plot(t,theta2)
    #mp.plot(np.linspace(0,t[-1],len(noisefile)),noisefile,color='r')
    p1,=mp.plot(t,phi_theory,lw=2)
    p2,=mp.plot(t,phi_exp,lw=2)
    
    mp.legend([p1,p2],["phi_theory","phi_experiment"],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.   )
    mp.xlabel("t")
    mp.ylabel("phi")
    max_y=np.amax([phi_theory,phi_exp])
    min_y=np.amin([phi_theory,phi_exp])
    max_x=np.amax(t);min_x=np.amin(t)
    marginy = (max_y-min_y)/20.
    marginx = (max_x-min_x)/20.
    x_coord = np.amin(t) + marginx # x-coord of text
    y_coord = min_y - marginy # y-coord of text
    mp.text(x_coord,y_coord,"phi theory vs experiment. al="+str(alpha)+",be="+str(beta))
    mp.text(x_coord,y_coord-.8*marginy,"eps="+str(eps)+",q0="+str(q0)+",q1="+str(q1)+",f="+str(f)+",mu="+str(100))
    mp.ylim(y_coord-marginy,max_y+marginy)
    mp.xlim(min_x,max_x)

    #mp.figure()
    #mp.title("phi_theory histogram")
    #mp.hist(phi_theory,bins=100,normed=True)

    mp.show()

if __name__ == '__main__':
    main()
