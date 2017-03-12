# generate long movie of weakly coupled traub models
# by default, the data files correspond to figure 8 of the paper.
# starttime sets the time at which you start saving frames.

# avconv -r 10 -start_number 1 -i test%d.jpg -b:v 1000k test.mp4


import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.integrate import odeint
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])

#from matplotlib import rcParams

#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm} \usepackage{xcolor} \definecolor{blue1}{HTML}{3399FF}']
matplotlib.rcParams.update({'figure.autolayout': True})

sizeOfFont = 20
fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}
lamomfsize=15 #lambda omega figure size


import euler
import lambda_omega

pi = np.pi
sin = np.sin
cos = np.cos

movdir = 'mov/' # save frames to this dir
#starttime = 7666 # starting time in ms
#endtime = 7966 # starting time in ms
starttime = 800
endtime = 2000

# other start time at 9500 and end at 9800.

### DATA

trueperiod = 2*np.pi
T = trueperiod*2000
dt = 0.05
N = int(T/dt)
t = np.linspace(0,T,N)
noisefile = None
initc = [2/np.sqrt(2),2/np.sqrt(2),-2/np.sqrt(2),2/np.sqrt(2)]

q0=.9;q1=1.;eps=.0025;
f=1.;a=1.;alpha=1.
beta=1.;partype='p'

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


counter=1
j=int(starttime/dt)
nfinal = int(endtime/dt)

skipn = 10

while j < int(starttime/dt)+16000:

    q = q0+q1*np.cos(eps*f*t)

    fig = plt.figure(figsize=(7,7))

    ax11 = plt.subplot2grid((2,2),(0,0))
    ax12 = plt.subplot2grid((2,2),(0,1))
    ax21 = plt.subplot2grid((2,2),(1,0),colspan=2)

    ax11.set_title(r"\textbf{Oscillator 1}")
    ax12.set_title(r"\textbf{Oscillator 2}")
    ax21.set_title(r"\textbf{Phase Difference and Slow Parameter}")
    ax21b = ax21.twinx()

    
    ax21.set_xlabel(r"\textbf{Time (ms)}",fontsize=15)

    ax21.set_ylabel(r"$\mathbf{\phi}$",fontsize=15)
    ax21b.set_ylabel(r'$\bm{q(t)}$',fontsize=15,color='red')

    ax21.set_xlim(0,T)
    ax21b.set_xlim(0,T)

    ax21.set_ylim(0,pi)
    ax21b.set_ylim(np.amin(q),np.amax(q))


    ax11.set_xticks([])
    ax12.set_xticks([])
    ax12.set_yticks([])

    ax11.set_yticks([])
    ax21.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi"]
    ax21.set_yticklabels(x_label, fontsize=15)


    #ax11.scatter(vdat[k,1],ndat[k,1],color='red',s=50)


    j += skipn
    k = j


    if q0 == 0.9:
        ax21.set_ylabel(r'$\bm{\phi(t)}$',fontsize=lamomfsize)
    if q0 == 1.1:
        ax21b.set_ylabel(r'$\bm{q(t)}$',fontsize=lamomfsize,color='red')


    xx = np.linspace(-pi,pi,75)
    ax11.plot(cos(xx),sin(xx), color='black', lw=3) # circle
    ax12.plot(cos(xx),sin(xx), color='black', lw=3) # circle

    # oscillators 1 and 2
    ax11.scatter(lcsolcoupled[:,1][j],lcsolcoupled[:,0][j],color='red',s=50)
    ax12.scatter(lcsolcoupled[:,3][j],lcsolcoupled[:,2][j],color='red',s=50)


    ax21.plot(t[:j],phi_exp[:j],lw=5,color='black',label='Numerics')
    ax21.plot(t[:j],phi_theory[:j],lw=5,color="#3399ff",ls='dashdot',dashes=(10,5),label='Theory')
    ax21b.plot(t[:j],q[:j],lw=4,color="red",linestyle='--',dashes=(10,2))
    #ax21b.plot([t[0,1],t[-1]],[1,1],lw=2,color='red')


    for tl in ax21b.get_yticklabels():
        tl.set_color('r')
        
    # beautify
    ax21.tick_params(labelsize=lamomfsize,top='off')
    ax21b.tick_params(labelsize=lamomfsize,top='off')
    plt.gcf().subplots_adjust(bottom=0.15)



    ax21.legend()
    
    fig.savefig(movdir+str(counter)+".png",dpi=100)
    
    #plt.pause(.01)
    print t[k],k,counter

    #ax11.clear()
    #ax12.clear()
    #ax21.clear()
    #ax21b.clear()
    plt.cla()
    plt.close()

    counter += 1




plt.show()
