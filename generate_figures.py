"""
Run to generate figures
Requires TeX; may need to install texlive-extra-utils on linux

the main() function at the end calls the preceding individual figure functions.

figures are saved as both png and pdf.

Copyright (c) 2016, Youngmin Park, Bard Ermentrout
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mticker

import matplotlib.pylab as mp
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])

#from matplotlib import rcParams

#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm}']
matplotlib.rcParams.update({'figure.autolayout': True})

sizeOfFont = 20
fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}
lamomfsize=40 #lambda omega figure size


import phase_model
import lambda_omega
import euler

import numpy as np
import matplotlib.pyplot as plt

#from matplotlib import pyplot as plt
from scipy.integrate import odeint

# modified specgram (use my_specgram)
from specgram_mod import *

# default parms (trb2)
default_gm0=.3;default_gm1=.5;
default_eps=.0025;default_f=.5

# default parms (lamom2)
default_eps_lamom=0.0025;default_f_lamom=1.
default_a=1.;default_alpha=1.
default_beta=1.

def trb2_p_fig(gm0=default_gm0,gm1=default_gm1,eps=default_eps,f=default_f,partype='p'):
    """
    two weakly coupled trab models, periodic slowly varying parameter figure
    data files created using trb2simple.ode and trb2simple_just1.ode
    """
    # initialize
    #filename = "trb2_psi_maxn_qp"+str(filenum)
    #filename = "trb2_psi_maxn_p1_ref.dat"

    filename = "trb2_psi_maxn_p1_ref2.dat" # with reviewer's fix
    #filename = "trb2_psi_maxn_p1_refined_2tables.dat"

    dat = np.loadtxt(filename)
    psi0=np.mean(dat[:,1][:int(5/.05)])
    T=dat[:,0][-1]
    N = len(dat[:,0])
    t = np.linspace(0,T,N)
    noisefile = None
    
    # generate data for plots
    sol = euler.ESolve(phase_model.happrox,psi0,t,args=(gm0,gm1,f,eps,partype,noisefile))
    full_model = np.abs(np.mod(dat[:,1]+.5,1)-.5) # [0] to make regular row array
    slow_phs_model = np.abs(np.mod(sol+.5,1)-.5)[:,0]

    # create plot object
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,5)

    ## plot data+theory
    ax1.scatter(dat[:,0]/1000.,full_model*2*np.pi,s=.5,facecolor="gray")
    ax1.plot(np.linspace(0,dat[:,0][-1]/1000.,N),slow_phs_model*2*np.pi,lw=5,color="#3399ff")
    ax1.set_ylabel(r'$\bm{|\phi(t)|}$',fontsize=20)
    ax1.set_xlabel(r'$\bm{t (s)}$',fontsize=20)

    # set tick intervals
    myLocatorx = mticker.MultipleLocator(2000/1000.)
    #myLocatory = mticker.MultipleLocator(.5)
    ax1.xaxis.set_major_locator(myLocatorx)
    #ax1.yaxis.set_major_locator(myLocatory)

    # make plot fit window

    ax1.set_yticks(np.arange(0,0.5,.125)*2*np.pi)
    x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$"]
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax1.set_yticklabels(x_label, fontsize=lamomfsize)

    ax1.set_ylim(np.amin([full_model])*2*np.pi,np.amax([full_model])*2*np.pi)
    ax1.set_xlim(dat[:,0][0]/1000.,dat[:,0][-1]/1000.)

    ## plot P param
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=20,color='red')
    
    # slowly varying parameter
    gm = gm0+(gm1-gm0)*np.cos(eps*f*t)

    # set tick intervals
    myLocatory2 = mticker.MultipleLocator(.1)
    ax2.yaxis.set_major_locator(myLocatory2)
    
    # make param plot fit window
    ax2.set_xlim(dat[:,0][0]/1000.,dat[:,0][-1]/1000.)
    ax2.set_ylim(np.amin(gm),np.amax(gm))
    
    # plot param + stability line 
    ax2.plot(t/1000.,gm,lw=4,color="red",linestyle='--',dashes=(10,2))
    ax2.plot([dat[:,0][0]/1000.,dat[:,0][-1]/1000.],[0.3,0.3],lw=2,color='red')

    # set ticks to red
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    
    # beautify
    ax1.tick_params(labelsize=20,top='off')
    ax1.tick_params(axis='x',pad=8)
    ax2.tick_params(labelsize=20,top='off')
    plt.gcf().subplots_adjust(bottom=0.15)
    


    return fig

def trb2newpar_p_fig(gm0=default_gm0,gm1=default_gm1,eps=default_eps,f=default_f,partype='p'):
    """
    two weakly coupled trab models, periodic slowly varying parameter figure, with parameters in interval [0.05,0.3]
    data files created using trb2_new_params/trb2simple_newpar.ode
    """

    # initialize
    # no more switch from stable/unstable. There always exists a stable point
    #filename = "trb2_new_params/trb2newpar_psi_p.dat" # no normalization by variance
    filename = "trb2_new_params/trb2newpar_psi_p2.dat" # includes normalization by variance
    dat = np.loadtxt(filename)
    psi0=np.mean(dat[:,1][:int(5/.05)])
    T=dat[:,0][-1]
    N = len(dat[:,0])
    dt = T/(1.*N)
    t = np.linspace(0,T,N)
    noisefile = None
    
    # generate data for plots
    sol = euler.ESolve(phase_model.happrox_newpar,psi0,t,args=(gm0,gm1,f,eps,partype,noisefile))
    full_model = np.abs(np.mod(dat[:,1]+.5,1)-.5) # [0] to make regular row array
    slow_phs_model = np.abs(np.mod(sol+.5,1)-.5)[:,0]

    # create plot object
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,5)
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ## plot data+theory
    ax1.scatter(dat[:,0]/1000.,full_model*2*np.pi,s=.5,facecolor="gray")
    ax1.plot(np.linspace(0,dat[:,0][-1]/1000.,N),slow_phs_model*2*np.pi,lw=5,color="#3399ff")
    myLocatorx = mticker.MultipleLocator(2000/1000.)
    #myLocatory = mticker.MultipleLocator(.5)
    ax1.xaxis.set_major_locator(myLocatorx)
    #ax1.yaxis.set_major_locator(myLocatory)

    ax1.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi"]
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax1.set_yticklabels(x_label, fontsize=lamomfsize)

    ax1.set_ylabel(r'$\bm{|\phi(t)|}$',fontsize=20)
    ax1.set_xlabel(r'$\bm{t (s)}$',fontsize=20)

    # make plot fit window
    ax1.set_ylim(np.amin([full_model])*2*np.pi,np.amax(full_model)*2*np.pi)
    ax1.set_xlim(dat[:,0][0]/1000.,dat[:,0][-1]/1000.)

    ## plot P param
    ax2 = ax1.twinx()
    gm = gm0+(gm1-gm0)*np.cos(eps*f*t)
    
    ax2.set_xlim(dat[:,0][0]/1000.,dat[:,0][-1]/1000.)
    ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=20,color='red')
    ax2.plot(t/1000.,gm,lw=4,color="red",linestyle='--',dashes=(10,2))

    myLocatory2 = mticker.MultipleLocator(.05)
    ax2.yaxis.set_major_locator(myLocatory2)

    #ax2.plot([dat[:,0][0],dat[:,0][-1]],[0.3,0.3],lw=2,color='red')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    
    # beautify
    ax1.tick_params(labelsize=20,top='off')
    ax1.tick_params(axis='x',pad=8)
    ax2.tick_params(labelsize=20,top='off')
    plt.gcf().subplots_adjust(bottom=0.15)


    return fig


def trb2_qp_fig(gm0=default_gm0,gm1=default_gm1,eps=default_eps,f=default_f,partype='qp'):
    """
    two weakly coupled trab models, quasi-periodic slowly varying parameter figure
    data files created using trb2simple.ode and trb2simple_just1.ode
    """

    # initialize
    #filename = "trb2_psi_maxn_qp"+str(filenum)
    #filename = "trb2_psi_maxn_qp_ref.dat"
    filename = "trb2_psi_maxn_qp_ref2.dat" # with reviewer fix
    dat = np.loadtxt(filename)
    psi0=np.mean(dat[:,1][:int(5/.05)])
    T=dat[:,0][-1]
    N = len(dat[:,0])
    t = np.linspace(0,T,N)
    noisefile = None
    
    # generate data for plots
    sol = euler.ESolve(phase_model.happrox,psi0,t,args=(gm0,gm1,f,eps,partype,noisefile))
    full_model = np.abs(np.mod(dat[:,1]+.5,1)-.5) # [0] to make regular row array
    slow_phs_model = np.abs(np.mod(sol+.5,1)-.5)[:,0]

    # create plot object
    rc('font', weight='bold')
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,5)

    # plot data+theory
    ax1.scatter(dat[:,0]/1000.,full_model*2*np.pi,s=.5,facecolor="gray")
    ax1.plot(np.linspace(0,dat[:,0][-1]/1000.,N),slow_phs_model*2*np.pi,lw=5,color="#3399ff")
    ax1.set_ylabel(r'$\bm{|\phi(t)|}$',fontsize=20)
    ax1.set_xlabel(r'$\bm{t (s)}$',fontsize=20)

    #myLocatorx = mticker.MultipleLocator(5000/1000.)
    myLocatory = mticker.MultipleLocator(.5)
    #ax1.xaxis.set_major_locator(myLocatorx)
    ax1.yaxis.set_major_locator(myLocatory)

    ax1.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi"]
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax1.set_yticklabels(x_label, fontsize=lamomfsize)

    # make plot fit window
    ax1.set_ylim(np.amin([full_model])*2*np.pi,np.amax(full_model)*2*np.pi)
    ax1.set_xlim(dat[:,0][0]/1000.,dat[:,0][-1]/1000.)

    ## plot QP param
    ax2 = ax1.twinx()
    gm = gm0+((gm1-gm0)/2)*(np.cos(eps*f*t)+np.cos(np.sqrt(2)*eps*f*t))
    
    ax2.plot(t/1000.,gm,lw=4,color="red",linestyle='--',dashes=(10,2))
    ax2.plot([dat[:,0][0]/1000.,dat[:,0][-1]/1000.],[0.3,0.3],lw=2,color='red')

    myLocatory2 = mticker.MultipleLocator(.1)
    ax2.yaxis.set_major_locator(myLocatory2)

    ax2.set_xlim(dat[:,0][0]/1000.,dat[:,0][-1]/1000.)
    ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=20,color='red')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    # beautify
    ax1.tick_params(labelsize=20,top='off')
    ax1.tick_params(axis='x',pad=8)
    ax2.tick_params(labelsize=20,top='off')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    return fig


def trb2_s_fig(filenum=4,gm0=default_gm0,gm1=default_gm1,eps=default_eps,f=default_f,partype='s'):
    """
    two weakly coupled trab models, stochastic "slowly" varying parameter figure
    data files created using

    trb2simple.ode
    trb2simple_just1.ode
    generateou.ode
    """

    # initialize
    #filename = "trb2_psi_maxn_s"+str(filenum)+".dat"
    #filename = "trb2_psi_maxn_s1.dat"
    #filename = "trb2_psi_maxn_s"+str(filenum)+"_mu1k.dat"
    filename = "trb2_psi_maxn_s"+str(filenum)+"_mu1k2.dat" # with reviewer edit
    dat = np.loadtxt(filename)
    psi0=np.mean(dat[:,1][:int(5/.05)])
    T=dat[:,0][-1]
    N = len(dat[:,0])
    dt = T/(1.*N)
    t = np.linspace(0,T,N)
    #noisefile = np.loadtxt("ounormed"+str(filenum)+".tab")
    noisefile = np.loadtxt("ounormed"+str(filenum)+"_mu1k.tab")
    
    # generate data for plots
    sol = euler.ESolve(phase_model.happrox,psi0,t,args=(gm0,gm1,f,eps,partype,noisefile))
    full_model = np.abs(np.mod(dat[:,1]+.5,1)-.5) # [0] to make regular row array
    slow_phs_model = np.abs(np.mod(sol+.5,1)-.5)[:,0]

    # create plot object
    fig = plt.figure()
    fig.set_size_inches(10,7.5)

    gs = gridspec.GridSpec(2,3)

    ax1 = plt.subplot(gs[:1,:])

    # plot data+theory
    ax1.scatter(dat[:,0]/1000.,full_model*2*np.pi,s=.5,facecolor="gray")
    ax1.plot(np.linspace(0,dat[:,0][-1]/1000.,N),slow_phs_model*2*np.pi,lw=4,color="#3399ff")
    ax1.set_ylabel(r'$\bm{|\phi(t)|}$',fontsize=20)


    ax1.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$",   r"$\pi$"]
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax1.set_yticklabels(x_label, fontsize=lamomfsize)
    
    # make plot fit window
    ax1.set_ylim(np.amin(full_model)*2*np.pi,np.amax(full_model)*2*np.pi)#np.amax(full_model))
    ax1.set_xlim(dat[:,0][0]/1000.,dat[:,0][-1]/1000.)
    
    #myLocatory = mticker.MultipleLocator(.5)
    #ax1.yaxis.set_major_locator(myLocatory)

    ## plot s param
    ax2 = plt.subplot(gs[1,:])

    s_N = len(noisefile[3:])

    ax2.plot(np.linspace(0,dat[:,0][-1]/1000.,s_N),(gm0+(gm1-gm0)*noisefile[3:]),lw=1,color="red")
    ax2.plot([dat[:,0][0]/1000.,dat[:,0][-1]/1000.],[0.3,0.3],lw=3,color='red',linestyle='--',dashes=(10,2))

    myLocatorx = mticker.MultipleLocator(2000/1000.)
    ax2.xaxis.set_major_locator(myLocatorx)

    ax2.set_xlim(dat[:,0][0]/1000.,dat[:,0][-1]/1000.)
    ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=20,color='red')

    myLocatory2 = mticker.MultipleLocator(.1)
    ax2.yaxis.set_major_locator(myLocatory2)

    ax2.set_xlabel(r'$\bm{t (s)}$',fontsize=20)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')    

    ax1.tick_params(labelsize=20,
                    top='off',
                    right='off')
    ax1.xaxis.set_ticklabels([])

    #ax2.set_xticks([])
    #ax2.set_yticks([])
    ax2.tick_params(labelsize=20,
                    top='off',
                    right='off')
    ax2.tick_params(axis='x',pad=8)
    ax2.set_frame_on(False)    


    return fig


def lamom2_p_fig(q0,q1,eps=default_eps_lamom,
                 f=default_f_lamom,a=default_a,alpha=default_alpha,
                 beta=default_beta,partype='p'):
    """
    two weakly coupled lambda-omega models, periodic slowly varying parameter figure

    the model is simulated in this function. calls functions from lambda_omega.py
    """

    # initialize
    #filename = "trb2_psi_maxn_qp"+str(filenum)
    trueperiod = 2*np.pi
    T = trueperiod*2000
    dt = 0.05
    N = int(T/dt)
    t = np.linspace(0,T,N)
    noisefile = None
    initc = [2/np.sqrt(2),2/np.sqrt(2),-2/np.sqrt(2),2/np.sqrt(2)]
    
    # generate data for plots
    lcsolcoupled = odeint(lambda_omega.lamom_coupled,initc,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))
    
    phi1init = np.arctan2(initc[1],initc[0])
    phi2init = np.arctan2(initc[3],initc[2])
    # compute hodd
    # get theory phase
    phi_theory = odeint(lambda_omega.Hodd,
                        phi2init-phi1init,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))

    theta1 = np.arctan2(lcsolcoupled[:,1],lcsolcoupled[:,0])
    theta2 = np.arctan2(lcsolcoupled[:,3],lcsolcoupled[:,2])
    phi_exp = np.mod(theta2-theta1+np.pi,2*np.pi)-np.pi
    phi_theory  = np.mod(phi_theory+np.pi,2*np.pi)-np.pi

    # create plot object
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,5)
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ## plot data+theory

    ax1.plot(t,phi_exp,lw=5,color='black')
    ax1.plot(t,phi_theory,lw=5,color="#3399ff",ls='dashdot',dashes=(10,5))
    if q0 == 0.9:
        ax1.set_ylabel(r'$\bm{\phi(t)}$',fontsize=lamomfsize)

    
    ax1.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$",   r"$\pi$"]
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax1.set_yticklabels(x_label, fontsize=lamomfsize)
    padding = 0.1
    ax1.set_ylim(-0.1,np.pi+0.1)
    #ax1.set_xlabel(r'$\bm{t}$',fontsize=20)
    #xtick_locs = np.arange(0,T+2000,2000,dtype='int') 
    #ytick_locs = np.arange(0,np.pi+0.5,0.5)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    #plt.xticks(xtick_locs, [r"" % x for x in xtick_locs])
    #plt.yticks(ytick_locs, [r"$\mathbf{%s}$" % x for x in ytick_locs])    

    #fig = plt.figure(figsize=(15,7.5))
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # make plot fit window
    #ax1.set_ylim(np.amin([full_model]),np.amax(full_model))
    #ax1.set_xlim(dat[:,0][0],dat[:,0][-1])

    ## plot P param
    ax2 = ax1.twinx()

    q = q0+q1*np.cos(eps*f*t)
    
    # dumb hack to get bold right-side axis labels # used boldmath instead
    #minval=np.amin(q);maxval=np.amax(q);increment=(maxval-minval)/8.
    #ytick_loc2 = np.arange(minval,maxval+increment,increment)
    #ytick_lab2 = []
    # http://stackoverflow.com/questions/6649597/python-decimal-places-putting-floats-into-a-string

    #for val in ytick_loc2:
    #    ytick_lab2.append(r'\boldmath ${0:.2f}$'.format(val))
    
    #ax2.set_yticks(ytick_loc2)
    #ax2.set_yticklabels(ytick_lab2)
    
    ax2.set_xlim(0,T)
    ax2.set_ylim(np.amin(q),np.amax(q))
    if q0 == 1.1:
        ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=lamomfsize,color='red')
    ax2.plot(t,q,lw=4,color="red",linestyle='--',dashes=(10,2))
    ax2.plot([t[0],t[-1]],[1,1],lw=2,color='red')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
        
    # beautify
    ax1.tick_params(labelsize=lamomfsize,top='off')
    ax2.tick_params(labelsize=lamomfsize,top='off')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    #axes.set_xticks([])
    #axes.set_yticks([])
    #axes.set_frame_on(False)
    ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax1.set_frame_on(False)
    #ax1.tick_params(labelsize=16)
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    #ax2.set_frame_on(False)
    #ax2.tick_params(labelsize=16)

    return fig

def lamom2_qp_fig(q0,q1,eps=default_eps_lamom,
                 f=default_f_lamom,a=default_a,alpha=default_alpha,
                 beta=default_beta,partype='qp'):
    """
    two weakly coupled lambda-omega models, quasi-periodic slowly varying parameter figure

    the model is simulated in this function. calls functions from lambda_omega.py
    """

    # initialize
    #filename = "trb2_psi_maxn_qp"+str(filenum)
    trueperiod = 2*np.pi
    T = trueperiod*2000
    dt = 0.05
    N = int(T/dt)
    t = np.linspace(0,T,N)
    noisefile = None
    initc = [2/np.sqrt(2),2/np.sqrt(2),-2/np.sqrt(2),2/np.sqrt(2)]
    
    # generate data for plots
    lcsolcoupled = odeint(lambda_omega.lamom_coupled,initc,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))
    
    phi1init = np.arctan2(initc[1],initc[0])
    phi2init = np.arctan2(initc[3],initc[2])
    # compute hodd
    # get theory phase
    phi_theory = odeint(lambda_omega.Hodd,
                        phi2init-phi1init,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))

    theta1 = np.arctan2(lcsolcoupled[:,1],lcsolcoupled[:,0])
    theta2 = np.arctan2(lcsolcoupled[:,3],lcsolcoupled[:,2])
    phi_exp = np.mod(theta2-theta1+np.pi,2*np.pi)-np.pi
    phi_theory  = np.mod(phi_theory+np.pi,2*np.pi)-np.pi

    # create plot object
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,5)
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ## plot data+theory
    ax1.plot(t,phi_exp,lw=5,color='black')
    ax1.plot(t,phi_theory,lw=5,color="#3399ff",ls='dashdot',dashes=(10,5))
    if q0 == 0.9:
        ax1.set_ylabel(r'$\bm{\phi(t)}$',fontsize=lamomfsize)

    ax1.set_xlabel(r'$\bm{t}$',fontsize=lamomfsize)

    ax1.xaxis.set_major_locator(MultipleLocator(4000))

    ax1.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$",   r"$\pi$"]
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax1.set_yticklabels(x_label, fontsize=lamomfsize)
    ax1.set_ylim(-0.1,np.pi+0.1)
    #xtick_locs = np.arange(0,T+2000,2000,dtype='int') 
    #ytick_locs = np.arange(0,np.pi+0.5,0.5)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    #plt.xticks(xtick_locs, [r"" % x for x in xtick_locs])
    #plt.yticks(ytick_locs, [r"$\mathbf{%s}$" % x for x in ytick_locs])    

    #fig = plt.figure(figsize=(15,7.5))
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # make plot fit window
    #ax1.set_ylim(np.amin([full_model]),np.amax(full_model))
    #ax1.set_xlim(dat[:,0][0],dat[:,0][-1])

    ## plot P param
    ax2 = ax1.twinx()

    q = q0+(q1/2.)*(np.cos(eps*f*t)+np.cos(np.sqrt(2)*eps*f*t))
    
    # dumb hack to get bold right-side axis labels
    #minval=np.amin(q);maxval=np.amax(q);increment=(maxval-minval)/8.
    #ytick_loc2 = np.arange(minval,maxval+increment,increment)
    #ytick_lab2 = []
    # http://stackoverflow.com/questions/6649597/python-decimal-places-putting-floats-into-a-string
    #for val in ytick_loc2:
    #    ytick_lab2.append(r'\boldmath ${0:.2f}$'.format(val))
    
    #ax2.set_yticks(ytick_loc2)
    #ax2.set_yticklabels(ytick_lab2)
    
    ax2.set_xlim(0,T)
    ax2.set_ylim(np.amin(q),np.amax(q))
    if q0 == 1.1:
        ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=lamomfsize,color='red')
    ax2.plot(t,q,lw=4,color="red",linestyle='--',dashes=(10,2))
    ax2.plot([t[0],t[-1]],[1,1],lw=2,color='red')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
        
    # beautify
    ax1.tick_params(labelsize=lamomfsize,top='off')
    ax2.tick_params(labelsize=lamomfsize,top='off')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    #axes.set_xticks([])
    #axes.set_yticks([])
    #axes.set_frame_on(False)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax1.set_frame_on(False)
    #ax1.tick_params(labelsize=16)
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    #ax2.set_frame_on(False)
    #ax2.tick_params(labelsize=16)

    return fig

def lamom2_s_fig(q0,q1,filenum,eps=default_eps_lamom,
                 f=default_f_lamom,a=default_a,alpha=default_alpha,
                 beta=default_beta,partype='s'):
    """
    two weakly coupled lambda-omega models, stochastic "slowly" varying parameter figure

    the model is simulated in this function. calls functions from lambda_omega.py

    filenum: seed
    """

    # initialize
    #filename = "trb2_psi_maxn_s"+str(filenum)+".dat"
    #filename = "trb2_psi_maxn_s1.dat"
    dt=.05

    noisefile = np.loadtxt("ounormed"+str(filenum)+"_mu1k.tab")
    total = noisefile[2]
    t = np.linspace(0,total,total/dt)
    initc = [2/np.sqrt(2),2/np.sqrt(2),-2/np.sqrt(2),2/np.sqrt(2)]
    
    # generate data for plots
    lcsolcoupled = euler.ESolve(lambda_omega.lamom_coupled,initc,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))

    phi1init = np.arctan2(initc[1],initc[0])
    phi2init = np.arctan2(initc[3],initc[2])
    # compute Hodd
    # get theory phase
    phi_theory = euler.ESolve(lambda_omega.Hodd,
                        phi2init-phi1init,t,args=(a,alpha,beta,eps,q0,q1,f,dt,partype,noisefile))
    
    theta1 = np.arctan2(lcsolcoupled[:,1],lcsolcoupled[:,0])
    theta2 = np.arctan2(lcsolcoupled[:,3],lcsolcoupled[:,2])
    phi_exp = np.mod(theta2-theta1+np.pi,2*np.pi)-np.pi
    phi_theory  = np.mod(phi_theory+np.pi,2*np.pi)-np.pi

    # create plot object
    fig = plt.figure()
    gs = gridspec.GridSpec(2,3)
    #ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=2)
    #ax2 = plt.subplot2grid((3,3),(2,0),colspan=3)

    ax1 = plt.subplot(gs[:1,:])
    # bold tick labels

    ax1.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$",   r"$\pi$"]
    ax1.set_yticklabels(x_label, fontsize=lamomfsize)
    
    #ytick_locs = np.arange(np.amin(phi_theory),np.amax(phi_theory),
    #                       (np.amax(phi_theory)-np.amin(phi_theory))/8.)
    #plt.yticks(ytick_locs, [r"$\mathbf{%1.1f}$" % x for x in ytick_locs])

    ax2 = plt.subplot(gs[1,:])
    #fig, axarr = plt.subplots(2, sharex=True)
    #axarr[0] = plt.subplot2grid(
    fig.set_size_inches(10,7.5)
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # plot data+theory
    ax1.plot(t,phi_exp,lw=5,color="black")
    ax1.plot(t,phi_theory,lw=5,color="#3399ff",ls='dashdot',dashes=(10,5))
    if q0 == .9:
        ax1.set_ylabel(r'$\bm{\phi(t)}$',fontsize=lamomfsize)

    #ax1.yaxis.set_major_locator(MultipleLocator(0.4))
    # make plot fit window
    #ax1.set_ylim(np.amin(full_model),0.3)#np.amax(full_model))
    #ax1.set_xlim(dat[:,0][0],dat[:,0][-1])
    ax1.set_xlim(0,total)
    ax1.set_ylim(-0.1,np.pi+0.1)
    # plot s param
    q = q0+(q1)*noisefile[3:]
    print 'mean =',np.mean(q),'for seed='+str(filenum)
    #ax2 = plt.subplots(2,1,1)
    #ax2 = ax1.twinx()
    s_N = len(noisefile[3:])
    s_N_half = s_N#int(s_N/2.)
    
    ax2.plot(np.linspace(0,t[-1],s_N),q,lw=1,color="red")
    ax2.plot([t[0],t[-1]],[1,1],lw=3,color='red',linestyle='--',dashes=(10,2))
    #ax2.set_xlim(dat[:,0][0],dat[:,0][-1])
    if q0 == .9:
        ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=lamomfsize,color='red')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax2.set_xlabel(r'$\bm{t}$',fontsize=lamomfsize)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')    
    ax2.yaxis.set_major_locator(MultipleLocator(0.4))
    ax2.xaxis.set_major_locator(MultipleLocator(4000))
    ax2.set_xlim(0,total)
    #xtick_locs = np.arange(t[0], t[-1], 2000,dtype='int')
    #minval=np.amin(q);maxval=np.amax(q)
    #ytick_locs = np.arange(minval,maxval,(maxval-minval)/8.)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    #plt.yticks(ytick_locs, [r"$\mathbf{%1.1f}$" % x for x in ytick_locs])

    #axes.set_xticks([])
    #axes.set_yticks([])
    #axes.set_frame_on(False)
    ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax1.set_frame_on(False)    
    ax1.tick_params(labelsize=lamomfsize,
                    top='off',
                    right='off')

    #ax2.set_xticks([])
    #ax2.set_yticks([])
    ax2.tick_params(labelsize=lamomfsize,
                    top='off',
                    right='off')
    ax2.set_frame_on(False)    


    return fig

def lo_inhom():
    """
    weakly coupled lambda-omega with slight frequency difference
    data generated in XPP
    """

    phi_full_0025=np.loadtxt('phi-full-0025.dat')
    phi_reduce_0025=np.loadtxt('phi-reduce-0025.dat')

    phi_full_025=np.loadtxt('phi-full-025.dat')
    phi_reduce_025=np.loadtxt('phi-reduce-025.dat')

    # create plot object
    fig = plt.figure()
    fig.set_size_inches(10,7.5)
    gs = gridspec.GridSpec(2,3)

    ax1 = plt.subplot(gs[:1,:])
    # plot data+theory for eps=.025
    ax1.plot([0,phi_full_025[-1,0]],[np.pi,np.pi],color='gray',lw=1.7)
    ax1.plot([0,phi_full_025[-1,0]],[0,0],color='gray',lw=1.7)
    ax1.plot([0,phi_full_025[-1,0]],[2*np.pi,2*np.pi],color='gray',lw=1.7)
    ax1.plot(phi_full_025[:,0],phi_full_025[:,1],lw=3,color="black")
    ax1.plot(phi_reduce_025[:,0],phi_reduce_025[:,1],lw=2,color="#3399ff",ls='dashdot',dashes=(10,1))

    # bold axis labels
    min1=np.amin(phi_full_025[:,1]);max1=np.amax(phi_full_025[:,1])
    padding1 = (max1-min1)/16.
    xtick_locs1 = np.arange(phi_full_025[0,0],phi_full_025[-1,0], 2000,dtype='int')
    #ytick_locs1 = np.arange(min1,max1,np.pi/2)#padding1*2)

    ax1.set_yticks(np.arange(0,1+.25,.25)*2*np.pi)
    x_label = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",   r"$2\pi$"]
    ax1.set_yticklabels(x_label, fontsize=20)

    plt.xticks(xtick_locs1, [r"$\mathbf{%s}$" % x for x in xtick_locs1])
    #plt.yticks(ytick_locs1, [r"$\mathbf{%1.1f}$" % x for x in ytick_locs1])

    # make plot fit window
    ax1.set_ylim(min1-padding1,max1+padding1)#np.amax(full_model))
    
    # axis labels
    ax1.set_ylabel(r'$\bm{\phi(t)}$',fontsize=20)
    ax1.set_xlabel(r'$\bm{t}$',fontsize=20)

    ax2 = plt.subplot(gs[1,:])

    # plot data+theory for eps=.0025
    ax2.plot([0,phi_full_0025[-1,0]],[np.pi,np.pi],color='gray',lw=1.7)
    ax2.plot([0,phi_full_0025[-1,0]],[0,0],color='gray',lw=1.7)
    ax2.plot([0,phi_full_0025[-1,0]],[2*np.pi,2*np.pi],color='gray',lw=1.7)
    ax2.plot(phi_full_0025[:,0],phi_full_0025[:,1],lw=3,color="black")
    ax2.plot(phi_reduce_0025[:,0],phi_reduce_0025[:,1],lw=2,color="#3399ff",ls='dashdot',dashes=(10,2))
        
    # bold tick labels    
    min2=np.amin(phi_full_0025[:,1]);max2=np.amax(phi_full_0025[:,1])
    padding2 = (max2-min2)/16.
    xtick_locs2 = np.arange(phi_full_0025[0,0],phi_full_0025[-1,0], 20000,dtype='int')
    #ytick_locs2 = np.arange(min2,max2,2*padding2)

    ax2.set_yticks(np.arange(0,1+.25,.25)*2*np.pi)
    x_label = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",   r"$2\pi$"]
    ax2.set_yticklabels(x_label, fontsize=20)

    plt.xticks(xtick_locs2, [r"$\mathbf{%s}$" % x for x in xtick_locs2])
    #plt.yticks(ytick_locs2, [r"$\mathbf{%1.1f}$" % x for x in ytick_locs2])
    
    # make plot fit window
    ax2.set_ylim(min2-padding2,max2+padding2)#np.amax(full_model))
    
    # axis labels
    ax2.set_ylabel(r'$\bm{\phi(t)}$',fontsize=20)
    ax2.set_xlabel(r'$\bm{t}$',fontsize=20)
    

    #axes.set_xticks([])
    #axes.set_yticks([])
    #axes.set_frame_on(False)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax1.set_frame_on(False)    
    ax1.tick_params(labelsize=20,
                    top='off',
                    right='off')

    #ax2.set_xticks([])
    #ax2.set_yticks([])
    ax2.tick_params(labelsize=20,
                    top='off',
                    right='off')
    #ax2.set_frame_on(False)    


    return fig


def trb2_prc_hodd():
    """
    comparison of traub model PRCs for different parameter values + Fourier approximation
    at gm=0.1 and gm=0.5
    """

    adj1 = np.loadtxt('trb2_adjoint.gm_0.1.dat')
    adj5 = np.loadtxt('trb2_adjoint.gm_0.5.dat')
    
    hfun1 = np.loadtxt('trb2_hfun.gm_0.1.dat')
    hfun5 = np.loadtxt('trb2_hfun.gm_0.5.dat')

    # create plot object
    fig = plt.figure()
    fig.set_size_inches(10,7.5)
    gs = gridspec.GridSpec(2,3)

    ax1 = plt.subplot(gs[:1,:])
    # plot adjoint gm=.1,gm=.5
    ax1.plot(np.linspace(0,2*np.pi,len(adj1[:,1])),adj1[:,1],lw=6,color="blue")
    ax1.plot(np.linspace(0,2*np.pi,len(adj5[:,1])),adj5[:,1],lw=6,color="red")#,ls='dashdot',dashes=(10,3))
    ax1.text(.55*2*np.pi,1.2,r'$\bm{q=0.5}$',fontsize=24)
    ax1.text(.18*2*np.pi,.25,r'$\bm{q=0.1}$',fontsize=24)
    # text label for gm
    #ax1.text()

    # bold axis labels
    min1=np.round(np.amin(adj5[:,1]),1);max1=np.round(np.amax(adj5[:,1]),1)
    padding1 = (max1-min1)/16.
    #padding_alt = np.round((max1-min1)/5.,decimals=1)
    #xtick_locs1 = np.linspace(0,1,6)#,dtype='int')
    #ytick_locs1 = np.arange(min1,max1+padding1,padding1*4)
    #ytick_locs1 = np.arange(min1,max1+padding_alt,padding_alt)
    #plt.xticks(xtick_locs1, [r"$\mathbf{%s}$" % x for x in xtick_locs1])
    #plt.yticks(ytick_locs1, [r"$\mathbf{%1.1f}$" % x for x in ytick_locs1])
    #plt.yticks(ytick_locs1, [r"$\mathbf{%1.1f}$" % x for x in ytick_locs1])

    # make plot fit window
    ax1.set_ylim(min1-padding1,max1+padding1)#np.amax(full_model))
    ax1.set_xlim(0,2*np.pi)

    # axis labels
    ax1.set_ylabel(r'$\bm{Z}$',fontsize=20)
    ax1.set_xlabel(r'$\bm{\phi}$',fontsize=20)

    
    ax1.set_xticks(np.arange(0,1+.25,.25)*2*np.pi)
    x_label = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",   r"$2\pi$"]
    ax1.set_xticklabels(x_label, fontsize=20)


    ax2 = plt.subplot(gs[1,:])

    # actual hfunctions
    hodd1 = -(np.flipud(hfun1[:,1])-hfun1[:,1])/2.
    hodd5 = -(np.flipud(hfun5[:,1])-hfun5[:,1])/2.
    # approx. hfunctions # call from phase model
    b11=.7213877067760438022;b15=-6.24915908247
    b21=0.738313204983;b25=1.43126232962
    phi = np.linspace(0,2*np.pi,30)
    happroxgm1=2*(b11*np.sin(phi)+b21*np.sin(2*phi))
    happroxgm5=2*(b15*np.sin(phi)+b25*np.sin(2*phi))

    # gm1
    ax2.plot(np.linspace(0,2*np.pi,len(hfun1[:,1])),hodd1,lw=7,color="blue")
    ax2.plot(phi,-happroxgm1,lw=5,color='black',ls='dashed',dashes=(5,2))
    ax2.plot(phi,-happroxgm1,lw=3,color='cyan',ls='dashed',dashes=(5,2))
    ax2.text(.1*2*np.pi,-6,r'$\bm{q=0.1}$',fontsize=24)

    #gm 5
    #ax2.plot(np.linspace(0,1,len(hfun5[:,1])),hodd5,lw=7,color="red",ls='dashdot',dashes=(10,3))
    ax2.plot(np.linspace(0,2*np.pi,len(hfun5[:,1])),hodd5,lw=7,color="red")
    ax2.plot(phi,-happroxgm5,lw=5,color='black',ls='dashed',dashes=(5,2))
    ax2.plot(phi,-happroxgm5,lw=3,color='#ffd80a',ls='dashed',dashes=(5,2))
    ax2.text(.4*2*np.pi,11,r'$\bm{q=0.5}$',fontsize=24)

    #ax2.plot(phi,-happroxgm1,lw=3,color='#3399ff',marker='s',markersize=10)
    #ax2.plot(phi,-happroxgm5,lw=3,color='#ff9999',marker='D',markersize=10)

    """
    plot horizontal zero line + zero intersections
    """
    # get idx of zero crossings of Hodd for q=0.1:
    zero_crossings = np.where(np.diff(np.sign(hodd1)))[0]

    # horizontal line at Hodd=0
    ax2.plot([0,2*np.pi],[0,0],color='gray',zorder=-3,lw=3)
    xx = np.linspace(0,2*pi,len(hfun1[:,1]))
    if len(zero_crossings) > 0:
        for idx in zero_crossings:
            # plot zero crossings (above horizontal line in zorder)
            ax2.scatter(xx[idx],hodd1[idx],s=300,zorder=-2,facecolor='black',edgecolor='black')
    

    # bold tick labels    
    min2=np.round(np.amin(hodd5));max2=np.round(np.amax(hodd5))
    padding2 = (max2-min2)/16.
    #padding2_alt = (max2-min2)/5.
    #xtick_locs2 = np.linspace(0,1,6)#,dtype='int')
    #ytick_locs2 = np.arange(min2,max2+padding2_alt,padding2_alt)
    #plt.xticks(xtick_locs2, [r"$\mathbf{%s}$" % x for x in xtick_locs2])
    #plt.yticks(ytick_locs2, [r"$\mathbf{%1.1f}$" % x for x in ytick_locs2])
    
    # make plot fit window
    ax2.set_ylim(min2-padding2,max2+padding2)#np.amax(full_model))
    ax2.set_xlim(0,2*np.pi)
    
    # axis labels
    ax2.set_ylabel(r'$\bm{H_{odd}(\phi)}$',fontsize=20)
    ax2.set_xlabel(r'$\bm{\phi}$',fontsize=20)

    ax2.set_xticks(np.arange(0,1+.25,.25)*2*np.pi)
    ax2.set_xticklabels(x_label, fontsize=20)
    

    #axes.set_xticks([])
    #axes.set_yticks([])
    #axes.set_frame_on(False)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax1.set_frame_on(False)    
    ax1.tick_params(labelsize=20,
                    axis='x',pad=10)
    ax1.tick_params(labelsize=20,
                    top='off',
                    right='off',
                    axis='both')

    #ax2.set_xticks([])
    #ax2.set_yticks([])
    ax2.tick_params(labelsize=20,
                    axis='x',pad=10)
    ax2.tick_params(labelsize=20,
                    top='off',
                    right='off',
                    axis='both')
    #ax2.set_frame_on(False)    


    return fig



def trb50_specgram():
    """
    spectrogram of 50 weakly coupled traub models.
    """
    dt = 0.1
    #x = np.loadtxt('vtot-stot-gmod-v25.dat') # all signals
    x = np.loadtxt('gooddata50.dat') # all signals
    t = x[:,0]/1000-2 # convert to units of s
    vtot = x[:,1] # total voltage signal
    stot = x[:,2] # total syn signal
    g = x[:,3] # param
    #t = np.linspace(0,100,100/dt)
    #vtot = sin(t*20*np.pi*dt)
    NFFT = 4096       # the length of the windowing segments (units of ms/dt)
    no = 4000
    Fs = int(1000.0/dt)  # the sampling frequency in Hz?

    #fig = plt.figure(figsize=(15,7.5))
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    fig = mp.figure()
    fig.set_size_inches(10,7.5)
    # plot Vtot
    ax1 = mp.subplot(211)
    ax1.set_title('')
    ax1.set_ylabel(r'$\textbf{Membrane Potential}$',fontsize=20)
    #ax1.set_xticks([])
    mp.plot(t,vtot)
    #xtick_locs = range(5000, 20000, 2000) 
    ytick_locs = np.arange(-85,-40,5)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    plt.yticks(ytick_locs, [r"$\mathbf{%s}$" % x for x in ytick_locs])
    
    # plot param
    ax2 = ax1.twinx()
    #ax3 = mp.subplot(313)
    ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=20,color='red')
    #ax2.set_xlabel(r'\textbf{t (s)}')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax2.plot(t,g,lw=5,color='red')

    # dumb hack to get bold right-side axis labels
    #minval=np.amin(g);maxval=np.amax(g);increment=(maxval-minval)/8.
    ytick_loc2 = np.arange(0,.6,.1)#np.arange(minval,maxval+increment,increment)
    ytick_lab2 = []
    # http://stackoverflow.com/questions/6649597/python-decimal-places-putting-floats-into-a-string
    for val in ytick_loc2:
        ytick_lab2.append(r'\boldmath ${0:.1f}$'.format(val))
    
    ax2.set_yticks(ytick_loc2)
    ax2.set_yticklabels(ytick_lab2)

    
    # plot spectrogram
    ax3 = mp.subplot(212, sharex=ax1)
    minfreq=15;maxfreq=120
    Pxx, freqs, bins, im = my_specgram(vtot, NFFT=NFFT, Fs=Fs, noverlap=no,minfreq=minfreq,maxfreq=maxfreq)#,
    #cmap=cm.gist_heat)
    #ax2.specgram(vtot, NFFT=NFFT, Fs=Fs, noverlap=no)#,
    ax3.set_ylabel(r'$\textbf{Frequency}$', fontsize=20)
    ax3.set_ylim(minfreq,maxfreq)
    
    # colorbar
    cbar = fig.colorbar(im, orientation='horizontal',shrink=.8,pad=.25)
    cbar.set_label(r'$\textbf{Intensity}$')

    # bold x,y-ticks
    xtick_locs3 = np.arange(0,14,2)
    ytick_locs3 = np.arange(20,140,20)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    plt.xticks(xtick_locs3, [r"$\mathbf{%s}$" % x for x in xtick_locs3])
    plt.yticks(ytick_locs3, [r"$\mathbf{%s}$" % x for x in ytick_locs3])
    ax3.set_xlabel(r'$\textbf{Time (Seconds)}$',fontsize=20)
    

    # beautify
    ax1.tick_params(labelsize=20,top='off',labelbottom='off')
    ax2.tick_params(labelsize=20,top='off')
    ax3.tick_params(labelsize=20,top='off')



    return fig


def trb50_op():
    """
    order parameter of 50 weakly coupled traub models
    """

    fig = mp.figure()
    fig.set_size_inches(10,5)
    dat = np.loadtxt('gm-op.dat')
    t=dat[:,0];op=dat[:,1];gm=dat[:,2]
    mp.plot(t,op,color='black',lw=3)
    mp.plot(t,gm,color='red',lw=3)
    mp.text(25,.53,'$\mathbf{q(t)}$',color='red',fontsize=20)

    mp.xlim(t[0],t[-1])
    mp.xlabel(r'\textbf{Time (Seconds)}',fontsize=20)
    mp.ylabel(r'\textbf{Order Parameter}',fontsize=20)

    xtick_locs = np.arange(20,75,10)
    ytick_locs = np.arange(0,1.02,0.2)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    plt.yticks(ytick_locs, [r"$\mathbf{%s}$" % x for x in ytick_locs])
    mp.tick_params(labelsize=20,top='off')
    #ax1 = mp.subplot(111)
    #ax1.plot(t,op)
     
    #ax2 = ax1.twinx()
    #ax2.plot(t,gm,color='red')



    return fig


def trb50_specgram_op():
    """
    Combined specgram, order parameter fig
    """
    dt = 0.1
    #x = np.loadtxt('vtot-stot-gmod-v25.dat') # all signals
    x = np.loadtxt('gooddata50.dat') # all signals
    t = x[:,0]/1000-2 # convert to units of s
    vtot = x[:,1] # total voltage signal
    stot = x[:,2] # total syn signal
    g = x[:,3] # param
    #t = np.linspace(0,100,100/dt)
    #vtot = sin(t*20*np.pi*dt)
    NFFT = 4096       # the length of the windowing segments (units of ms/dt)
    no = 4000
    Fs = int(1000.0/dt)  # the sampling frequency in Hz?

    #fig = plt.figure(figsize=(15,7.5))
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    fig = plt.figure()
    fig.set_size_inches(10,12.5)
    # plot Vtot
    ax1 = fig.add_subplot(311)
    ax1.set_title('')
    ax1.set_ylabel(r'$\textbf{Membrane Potential}$',fontsize=20)
    #ax1.set_xticks([])
    ax1.plot(t,vtot)
    #xtick_locs = range(5000, 20000, 2000) 
    #ytick_locs = np.arange(-85,-40,5)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    #plt.yticks(ytick_locs, [r"$\mathbf{%s}$" % x for x in ytick_locs])

    sublabelsize=25 # subfigure label (a),(b),(c) font size
    #from matplotlib.font_manager import FontProperties
    ax1.text(-1.7,-40,r'$\textbf{(a)}$',fontsize=sublabelsize)
    
    # plot param
    ax2 = ax1.twinx()
    #ax3 = mp.subplot(313)
    ax2.set_ylabel(r'$\bm{q(t)}$',fontsize=20,color='red')
    #ax2.set_xlabel(r'\textbf{t (s)}')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax2.plot(t,g,lw=5,color='red')

    # dumb hack to get bold right-side axis labels
    #minval=np.amin(g);maxval=np.amax(g);increment=(maxval-minval)/8.
    #ytick_loc2 = np.arange(0,.6,.1)#np.arange(minval,maxval+increment,increment)
    #ytick_lab2 = []
    # http://stackoverflow.com/questions/6649597/python-decimal-places-putting-floats-into-a-string
    #for val in ytick_loc2:
    #    ytick_lab2.append(r'\boldmath ${0:.1f}$'.format(val))
    
    #ax2.set_yticks(ytick_loc2)
    #ax2.set_yticklabels(ytick_lab2)






    
    # plot spectrogram
    ax3 = fig.add_subplot(312, sharex=ax1)
    minfreq=15;maxfreq=120
    Pxx, freqs, bins, im = my_specgram(vtot, NFFT=NFFT, Fs=Fs, noverlap=no,minfreq=minfreq,maxfreq=maxfreq)#,
    #cmap=cm.gist_heat)
    #ax2.specgram(vtot, NFFT=NFFT, Fs=Fs, noverlap=no)#,
    ax3.set_ylabel(r'$\textbf{Frequency}$', fontsize=20)
    ax3.set_ylim(minfreq,maxfreq)
    
    # colorbar
    cbar = fig.colorbar(im, orientation='horizontal',shrink=.8,pad=.25)
    cbar.set_label(r'$\textbf{Intensity}$',size=15)
    #print dir(cbar)
    cbar.ax.tick_params(labelsize=20)

    # bold x,y-ticks
    #xtick_locs3 = np.arange(0,14,2)
    #ytick_locs3 = np.arange(20,140,20)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    #plt.xticks(xtick_locs3, [r"$\mathbf{%s}$" % x for x in xtick_locs3])
    #plt.yticks(ytick_locs3, [r"$\mathbf{%s}$" % x for x in ytick_locs3])
    ax3.set_xlabel(r'$\textbf{Time (Seconds)}$',fontsize=20)
    ax3.text(-1.7,120,r'$\textbf{(b)}$',fontsize=sublabelsize)
    
    ## plot OP
    ax4 = fig.add_subplot(313)
    dat = np.loadtxt('gm-op.dat')
    
    # convert units of ms/eps to Seconds 
    t=dat[:,0]*1./(0.0025*1000)
    op=dat[:,1];gm=dat[:,2]
    ax4.plot(t,op,color='black',lw=3)
    ax4.plot(t,gm,color='red',lw=3)
    ax4.text(9.7,.53,r'$\bm{q(t)}$',color='red',fontsize=20)

    ax4.set_xlim(t[0],t[-1])
    ax4.set_xlabel(r'\textbf{Time (Seconds)}',fontsize=20)
    ax4.set_ylabel(r'\textbf{Order Parameter}',fontsize=20)
    ax4.text(5,1,r'$\textbf{(c)}$',fontsize=sublabelsize)
        
    #xtick_locs = np.arange(20,75,10)
    #ytick_locs = np.arange(0,1.02,0.2)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    #plt.yticks(ytick_locs, [r"$\mathbf{%s}$" % x for x in ytick_locs])

    #ax1 = mp.subplot(111)
    #ax1.plot(t,op)
     
    #ax2 = ax1.twinx()
    #ax2.plot(t,gm,color='red')



    # beautify
    ax1.tick_params(labelsize=20,top='off',labelbottom='off')
    ax2.tick_params(labelsize=20,top='off')
    ax3.tick_params(labelsize=20,top='off')
    ax4.tick_params(labelsize=20,top='off')


    return fig



def fd_diagram():
    """
    f-d parameter space.
    """

    fd_wn0 = np.loadtxt('fd_wn0.dat')
    fd_wn1 = np.loadtxt('fd_wn1.dat')

    fig = plt.figure()
    fig.set_size_inches(10,7.5)
    # plot Vtot
    ax1 = fig.add_subplot(111)
    #ax1.set_title(r'\textbf{(a)}',x=-.1,y=1.08)

    ax1.set_xlabel(r'$\textbf{d}$',fontsize=20)
    ax1.set_ylabel(r'$\textbf{f}$',fontsize=20)
    
    #ax1.set_xticks([])
    ax1.scatter(fd_wn0[:,0],fd_wn0[:,1],marker='+',color='red')
    ax1.scatter(fd_wn1[:,0],fd_wn1[:,1],marker='x',color='green')

    ax1.set_xlim(0,.15)
    ax1.set_ylim(0,2)
    #xtick_locs = range(5000, 20000, 2000) 
    #ytick_locs = np.arange(-85,-40,5)
    #plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
    #plt.yticks(ytick_locs, [r"$\mathbf{%s}$" % x for x in ytick_locs])

    sublabelsize=25 # subfigure label (a),(b),(c) font size
    #from matplotlib.font_manager import FontProperties
    #ax1.text(-1.7,-40,r'$\textbf{(a)}$',fontsize=sublabelsize)

    #ax1.tick_params(labelsize=20,top='off',labelbottom='off')

    return fig


def generate_figure(function, args, filenames, title="", title_pos=(0.5,0.95)):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;
    fig = function(*args)
    #fig.text(title_pos[0], title_pos[1], title, ha='center')
    if type(filenames) == list:
        for name in filenames:
            if name.split('.')[-1] == 'ps':
                fig.savefig(name, orientation='landscape')
            else:
                fig.savefig(name)
    else:
        if name.split('.')[-1] == 'ps':
            fig.savefig(filenames,orientation='landscape')
        else:
            fig.savefig(filenames)

def main():
    
    figures = [
        #(trb2newpar_p_fig, [.175,.3,default_eps,default_f,'p'], ['trb2newpar_p.png']),
        #(trb2_p_fig, [], ['trb2_p_fig.png']),
        #(trb2_qp_fig, [], ['trb2_qp_fig.png']),
        #(trb2_s_fig, [], ['trb2_s4_fig.png']),
        #(lamom2_p_fig, [0.9,1.], ['lamom2_p_fig1.pdf','lamom2_p_fig1.eps']),
        #(lamom2_p_fig, [1.1,1.], ['lamom2_p_fig2.pdf','lamom2_p_fig2.eps']),
        #(lamom2_qp_fig, [0.9,1.], ['lamom2_qp_fig1.pdf','lamom2_qp_fig1.eps']),
        #(lamom2_qp_fig, [1.1,1.], ['lamom2_qp_fig2.pdf','lamom2_qp_fig2.eps']),
        #(lamom2_s_fig, [0.9,1.,1], ['lamom2_s1_fig1.pdf','lamom2_s1_fig1.eps']),
        #(lamom2_s_fig, [0.85,1.,2], ['lamom2_s2_fig1.pdf','lamom2_s2_fig1.eps']),
        #(lo_inhom,[],['lo-inhom.pdf']),
        #(trb2_prc_hodd,[],['trb2_prc_hodd.pdf']),
        #(trb50_specgram,[],['trb50_specgram.pdf']),
        #(trb50_op,[],['trb50_op.pdf']),
        #(trb50_specgram_op,[],['network3_ymp.pdf']),
        #(fd_diagram,[],['fd_diagram.pdf','fd_diagram.eps']),
        ]
    for fig in figures:
        generate_figure(*fig)

if __name__ == "__main__":
    main()
