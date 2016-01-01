"""
Copyright (c) 2016, Youngmin Park, Bard Ermentrout
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ympark1988@gmail.com

Slowly varying phase model.

Use this script to compare full model to theory.

TODO:
None

Depricated notes:
Get bi_p12 and bi_p3 from datplot.py. Choose c=1 for b1 and c=2 for b2.

"""
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pylab as mp
mp.rc('text', usetex=True)
import matplotlib.pyplot as plt
#import time
import copy

from fourier_approx import *
from scipy.integrate import odeint, ode
from euler import *

# define coefficients b_i (for sine)
b0_1=0;b0_3=0
b1_1=0.721387113706;b1_3=-1.5028098729
b2_1=0.738312597998;b2_3=1.03494013487

b_at_1 = [b0_1,b1_1,b2_1,b2_1,b1_1]
b_at_3 = [b0_3,b1_3,b2_3,b2_3,b1_3]

# Define coefficiens a_i (for cosine)
a0_1=19.6011939665;a0_3=17.4255017198
a1_1=-3.32476526025;a1_3=-6.97305767558
a2_1=-0.255371105623;a2_3=-0.83690237427

a_at_1 = [a0_1,a1_1,a2_1,a2_1,a1_1]
a_at_3 = [a0_3,a1_3,a2_3,a2_3,a1_3]

def bi(gm,i):
    """
    gm: parameter value
    i: index, 0, 1, or 2 (start at 1 to keep b0 as the constant)
    
    note: b_at_3[i] means b_i evaluated at gm=0.3
    """
    return 5*(b_at_3[i]-b_at_1[i])*gm + 1.5*b_at_1[i]-.5*b_at_3[i]


def happrox(psi,t,gm0,gm1,f,eps,partype,noisefile=None):
    """
    approximation to -2Hodd
    sum_i b_i(gm) sin(i psi)
    
    psi: phase difference in full model. \psi \in [0,1)
    t: time
    gm0,gm1: slow parameter terms
    f: periodic/quasi-periodic frequency
    eps: coupling strength
    partype: string. to choose between quasi-per,per,and stoch
    noisefile: if partype=='s', then must specify noisefile
    """
    # slowly varying parameter (from trb2simple.ode)
    if partype=='periodic' or partype=='p':
        gm = gm0+(gm1-gm0)*np.cos(eps*f*t)
    elif partype=='qp' or partype=='quasiperiodic':
        gm = gm0+((gm1-gm0)/2)*(np.cos(eps*f*t)+np.cos(np.sqrt(2)*eps*f*t))
    elif partype=='s' or partype=='stochastic':
        assert(noisefile != None)
        N = noisefile[0]
        t0 = noisefile[1]
        tend = noisefile[2]

        # get index
        idx = int(N*t/(tend-t0))+3

        # slowly varing parameter
        gm = gm0 + (gm1-gm0)*noisefile[idx]

    # RHS

    tot = eps*2*(bi(gm,1)*np.sin(2*np.pi*psi)+bi(gm,2)*np.sin(2*np.pi*2*psi))
    return tot

def happrox_newpar(psi,t,gm0,gm1,f,eps,partype,noisefile=None):
    """
    new approximation to -2Hodd
    sum_i b_i(gm) sin(i psi)
    
    psi: phase difference in full model. \psi \in [0,1)
    t: time
    gm0,gm1: slow parameter terms
    f: periodic/quasi-periodic frequency
    eps: coupling strength
    partype: string. to choose between quasi-per,per,and stoch
    noisefile: if partype=='s', then must specify noisefile
    """
    # slowly varying parameter (from trb2simple.ode)
    if partype=='periodic' or partype=='p':
        gm = gm0+(gm1-gm0)*np.cos(eps*f*t)
    elif partype=='qp' or partype=='quasiperiodic':
        gm = gm0+((gm1-gm0)/2)*(np.cos(eps*f*t)+np.cos(np.sqrt(2)*eps*f*t))
    elif partype=='s' or partype=='stochastic':
        assert(noisefile != None)
        N = noisefile[0]
        t0 = noisefile[1]
        tend = noisefile[2]

        # get index
        idx = int(N*t/(tend-t0))+3

        # slowly varing parameter
        gm = gm0 + (gm1-gm0)*noisefile[idx]

    # RHS
    # linear fit did not work well. Hence quadratic fit.
    #0.05 0.946492310506
    #0.3 -1.50220208
    #m1=(-1.50220208-.946492310506)/(.3-.05);b1=m1*(gm-.05)+.946492310506

    #0.05 0.696962361891
    #0.3 1.03493602
    #m2=(1.03493602-0.696962361891)/(.3-.05);b2=m2*(gm-.05)+0.696962361891

    #0.05 0.223065149737
    #0.3 0.22916746
    #m3=(0.22916746-0.223065149737)/(.3-.05);b3=m3*(gm-.05)+0.223065149737

    #0.05 0.0699006023343
    #0.3 0.01561291
    #m4=(-0.02709481-0.0699006023343)/(.3-.05);b4=m4*(gm-.05)+0.0699006023343

    #0.05 0.0202243199847
    #0.3 -0.02709481
    #m5=(-0.02991668-0.0202243199847)/(.3-.05);b5=m5*(gm-.05)+0.0202243199847
    
    #b1 = -29.45045773*gm**2 + 0.80215421*gm + 0.94897777
    #b2 = 2.90001234*gm**2 + 0.35451794*gm + 0.67310092
    #b3 = 0.87912035*gm**2 -0.31112169*gm + 0.23994085
    #b4 = -0.16888056*gm**2 -0.17721699*gm +  0.08141014
    #b5 = -0.50866254*gm**2 -0.02235143*gm + 0.0238174
    #b6 = -0.56594122*gm**2 + 0.06418721*gm + 0.0009969
    #b7 = -0.50583443*gm**2 + 0.09786764*gm -0.00774121
    #b8 = -0.3993536*gm**2 + 0.09897358*gm -0.01002817
    #b9 = -0.28318736*gm**2 + 0.08312107*gm -0.00932904
    #b10 = -0.17679462*gm**2 + 0.06078856*gm -0.00742644
    #b1 = -45.64112171*gm**3  -6.34463987*gm**2  -2.67156303*gm +  1.09651805
    #b2 = 2.10057065*gm**3 + 1.83659844*gm**2 + 0.51439106*gm + 0.66631058
    #b3 = 5.48687909*gm**3 -1.89861219*gm**2 + 0.10648124*gm + 0.22220387
    #b4 = 3.32420768*gm**3 -1.8517607*gm**2 +  0.07578638*gm + 0.07066425
    def cubic(x,a3,a2,a1,a0):
        return a3*x**3 + a2*x**2 + a1*x + a0
    b1 = cubic(gm,-42.61640431,-7.94776356,-2.3814985,1.09031998)
    b2 = cubic(gm,1.21754145,2.10365281,0.49233105,0.66832328)
    b3 = cubic(gm,5.096827,-1.72966978,0.07646528,0.22475099)
    b4 = cubic(gm,3.14096481,-1.75666984,0.05311828,0.0730698)
    b5 = cubic(gm,1.78134396,-1.40235937,0.10339942,0.0198824)
    b6 = cubic(gm,8.15454337e-01,-9.71446193e-01,1.17405655e-01,-8.88598339e-08)
    b7 = cubic(gm,0.11257057,-0.55893078,0.09949942,-0.00673432)
    b8 = cubic(gm,-0.36173782,-0.21630085,0.0666423,-0.00777079)
    b9 = cubic(gm,-0.6396974,0.03694153,0.0316292,-0.00643063)
    b10 = cubic(gm,-0.75140434,0.19720189,0.00240786,-0.00437724)
    b11 = cubic(gm,-0.73667622,0.27523212,-0.0176746,-0.00242751)
    b12 = cubic(gm,-0.63762746,  0.28948784, -0.02845605, -0.00092166)

    blist = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12]
    intmult = np.arange(1,len(blist)+1,1)
    
    tot = eps*4*np.sum(blist*np.sin(2*np.pi*intmult*psi))
    #tot = eps*4*(b1*np.sin(2*np.pi*1*psi)+\
    #             b2*np.sin(2*np.pi*2*psi)+\
    #             b3*np.sin(2*np.pi*3*psi)+\
    #             b4*np.sin(2*np.pi*4*psi)+\
    #             b5*np.sin(2*np.pi*5*psi)+\
    #             b6*np.sin(2*np.pi*6*psi)+\
    #             b7*np.sin(2*np.pi*7*psi))
    return tot




def main():
    #f=6#f=4#f=3#f=2.5#f=2#f=1.5#f=1#f=.5
    newpar = False
    eps=.0025;f=.5
    if not(newpar):
        gm0=.3;gm1=.5
    else:
        gm0=.175;gm1=.325
    noisefile=None;filenum=1
    print "Initializing sims..."
    # choose from 'qp','p','s':quasi-per, per, stoch
    partype = 'p' # if s, pick seed above (filenum)

    if partype == 'qp':
        # full sim
        filename = "trb2_psi_maxn_qp"+str(filenum)
    elif partype == 'p':
        # full sim
        if not(newpar):
            if f != 0.5:
                filename = "trb2_psi_f"+str(f)+"_p"+str(filenum)
            else:
                #filename = "trb2_psi_maxn_p"+str(filenum)
                filename = "trb2_psi_maxn_p"+str(filenum)+"_refined"
        else:
            filename = 'trb2_new_params/trb2newpar_psi_p'
            #"trb2_new_params/trb2newpar_psi_f"+str(f)+"_p"
    elif partype == 's':
        # full sim
        filename = "trb2_psi_maxn_s"+str(filenum)+"_mu1k"
        #filename = "trb2_psi_maxn_s"+str(filenum)
        #dat = np.loadtxt("../psi_minv.dat") # seed=1
        #dat = np.loadtxt("../psi_minn.dat") # seed=1
        #dat = np.loadtxt("../psi_maxn.dat") # seed=1
        print 'Assuming mu=1k'
        #print 'Assuming mu=100'
        # load slow par
        noisefile = np.loadtxt("ounormed"+str(filenum)+"_mu1k.tab")
        #noisefile = np.loadtxt("ounormed"+str(filenum)+".tab")

    # load full sim
    dat = np.loadtxt(filename+".dat")

    # init
    psi0=.9#np.mean(dat[:,1][:int(5/.05)])
    N = len(dat[:,0])
    cutoffidx = N
    T=dat[:cutoffidx,0][-1]

    dt = T/(1.*N)
    t = np.linspace(0,T,N)
    #print N,len(dat[:,0])
    # solve
    sol = ESolve(happrox_newpar,psi0,t,args=(gm0,gm1,f,eps,partype,noisefile))
    #sol_test = ESolve(happrox2,psi0,t,args=(gm0,gm1,f,eps,partype,noisefile))

    # begin plotting
    full_model = np.abs(np.mod(dat[:,1]+.5,1)-.5) # [0] to make regular row array
    slow_phs_model = np.abs(np.mod(sol+.5,1)-.5)[:,0]
    #slow_phs_model_test = np.abs(np.mod(sol_test+.5,1)-.5)[:,0]
    corl = scipy.stats.pearsonr(full_model,slow_phs_model)[0]
    
    #mp.figure()
    #mp.plot(slow_phs_model)
    #mp.plot(slow_phs_model_test)
    #mp.show()

    
    """
    PLOTTING
    """
    print "Generating plots..."

    # plot
    # see http://matplotlib.org/examples/api/two_scales.html for docs on multiple scales
    fig, ax1 = plt.subplots()
    fig.set_size_inches(15,7.5)
    ax1.scatter(dat[:cutoffidx,0],full_model[:cutoffidx],s=.5,facecolor="gray")
    ax1.plot(np.linspace(0,T,N),slow_phs_model,lw=5,color="blue")
    ax1.set_ylabel(r'$\psi(t)$',fontsize=16)
    ax1.set_xlabel(r'$t$',fontsize=16)

    # make plot fit window
    ax1.set_ylim(np.amin([full_model]),np.amax(full_model))
    ax1.set_xlim(dat[:,0][0],T)
    #ax1.set_xlim(0,T)

    # noise function
    ax2 = ax1.twinx()
    if partype == 'qp':
        ax2.plot(t,gm0+((gm1-gm0)/2)*(np.cos(eps*f*t)+np.cos(np.sqrt(2)*eps*f*t)),lw=2,color="red")    
    elif partype == 'p':
        ax2.plot(t,gm0+(gm1-gm0)*np.cos(eps*f*t),lw=2,color="red")
    elif partype == 's':
        y = gm0+(gm1-gm0)*noisefile[3:]
        ax2.plot(np.linspace(0,T,len(y)),y,lw=1,color="red")
    ax2.plot([dat[:,0][0],T],[0.3,0.3],lw=2,color='red',linestyle='--')


    ax2.set_xlim(dat[:,0][0],T)
    ax2.set_ylabel(r'$\gamma(t)$',fontsize=16,color='red')


    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    # title
    if partype == 'p' or partype == 'qp':
        mp.title("Full model (black) vs phase model (green). Correlation coeff="+str(corl))
    elif partype == 's':
        mp.title("Full model (black) vs phase model (green). Correlation coeff="+str(corl)+", seed="+str(filenum))
        
    
    fig.savefig(filename+".png",dpi=100)
    #fig.savefig(filename+".pdf",dpi=100)
    #plt.show()
    #mp.show()

if __name__ == "__main__":
    main()


# keeping this here in case I need to use other integrators
### WARNING: FOR SCIPY.ODE, RHS MUST TAKE t,x
### IN CONTRAST TO ODEINT WHICH TAKES x,t
"""
r = ode(happrox).set_integrator('vode',nsteps=N)
r.set_initial_value(psi0,0).set_f_params(gm0,gm1,f,eps,'qp')
sol = np.zeros(N)
sol[0]=psi0
idx = 1
# run integration

while r.successful() and r.t < T-dt:
    r.integrate(r.t+dt)
    #print r.t
    sol[idx] = r.y
    idx += 1
"""
