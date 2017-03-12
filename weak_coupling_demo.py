# generate long movie of weakly coupled traub models
# by default, the data files correspond to figure 8 of the paper.
# starttime sets the time at which you start saving frames.

# avconv -r 10 -start_number 1 -i test%d.jpg -b:v 1000k test.mp4


import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
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
import phase_model



movdir = 'mov/' # save frames to this dir
#starttime = 7666 # starting time in ms
#endtime = 7966 # starting time in ms
starttime = 9500
endtime = 9800

# other start time at 9500 and end at 9800.

### DATA

#filename = "trb2_new_params/trb2newpar_psi_p.dat" # no normalization by variance
filename = "trb2_new_params/trb2newpar_p_psi_new.dat" # includes normalization by variance
dat = np.loadtxt(filename)
psi0=np.mean(dat[:,1][:int(5/.05)])
T=dat[:,0][-1]
N = len(dat[:,0])
dt = T/(1.*N)
t = np.linspace(0,T,N) # time

# solution values
vdat = np.loadtxt("trb2_new_params/trb2newpar_p_voltage.dat")
ndat = np.loadtxt("trb2_new_params/trb2newpar_p_n.dat")
vpdat = np.loadtxt("trb2_new_params/trb2newpar_p_voltagep.dat")
npdat = np.loadtxt("trb2_new_params/trb2newpar_p_np.dat")

vmax = np.amax(vdat[:,1])+10
vmin = np.amin(vdat[:,1])-10

nmax = np.amax(ndat[:,1])+.05
nmin = np.amin(ndat[:,1])-.05


# lookup table
vlo = np.loadtxt("trb2_new_params/VLo2.tab")[3:]
nlo = np.loadtxt("trb2_new_params/nLo2.tab")[3:]
vhi = np.loadtxt("trb2_new_params/VHi2.tab")[3:]
nhi = np.loadtxt("trb2_new_params/nHi2.tab")[3:]


# approx phase
theta1 = np.loadtxt("trb2_new_params/trb2newpar_p_theta1.dat")
theta2 = np.loadtxt("trb2_new_params/trb2newpar_p_theta2.dat")


# slow param
eps=.0025
gm0=.175
gm1=.3
f=.5
gm = gm0+(gm1-gm0)*np.cos(eps*f*t)


# theoretical phase
# generate data for plots
partype='p';noisefile=None
sol = euler.ESolve(phase_model.happrox_newpar,psi0,t,args=(gm0,gm1,f,eps,partype,noisefile))
slow_phs_model = np.abs(np.mod(sol+.5,1)-.5)[:,0]


minval = np.amin(dat[:,1])*2*np.pi-.05
maxval = np.amax(dat[:,1])*2*np.pi+.05

minvalp = np.amin(gm)-.05
maxvalp = np.amax(gm)+.05

x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi"]

counter=1
j=int(starttime/dt)
nfinal = int(endtime/dt)

skipn = 2
skipn2 = 1000

while j < int(starttime/dt)+1400:

    fig = plt.figure(figsize=(7,7))

    ax11 = plt.subplot2grid((2,2),(0,0))
    ax12 = plt.subplot2grid((2,2),(0,1))
    ax21 = plt.subplot2grid((2,2),(1,0),colspan=2)
    ax21b = ax21.twinx()

    ax11.set_title(r"\textbf{Oscillator 1}")
    ax12.set_title(r"\textbf{Oscillator 2}")
    ax21.set_title(r"\textbf{Phase Difference and Slow Parameter}")

    ax11.set_xlabel(r"\textbf{Voltage (mV)}",fontsize=15)
    ax12.set_xlabel(r"\textbf{Voltage (mV)}",fontsize=15)
    ax21.set_xlabel(r"\textbf{Time (ms)}",fontsize=15)

    ax11.set_ylabel(r"$\mathbf{n}$",fontsize=15)
    ax12.set_ylabel(r"$\mathbf{n}$",fontsize=15)
    ax21.set_ylabel(r"$\mathbf{\phi}$",fontsize=15)
    ax21b.set_ylabel(r'$\bm{q(t)}$',fontsize=15,color='red')

    ax11.set_xlim(vmin,vmax)
    ax12.set_xlim(vmin,vmax)
    ax21.set_xlim(t[0],t[-1])
    ax21b.set_xlim(t[0],t[-1])

    ax11.set_ylim(nmin,nmax)
    ax12.set_ylim(nmin,nmax)
    ax21.set_ylim(minval,maxval)
    ax21b.set_ylim(minvalp,maxvalp)

    ax21.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    ax21.set_yticklabels(x_label, fontsize=15)


    plt.locator_params(nticks=4)
    ax11.xaxis.set_ticks(np.arange(-80,80,40)) # fix x label spacing
    ax12.xaxis.set_ticks(np.arange(-80,80,40))


    j += skipn
    k = j
    #g1.matshow(np.reshape(sol[k,:N],(rN,rN)))

    # oscillators 1,2
    ax11.scatter(vdat[k,1],ndat[k,1],color='red',s=50)
    ax12.scatter(vpdat[k,1],npdat[k,1],color='red',s=50)

    ax11.plot(vlo,nlo,lw=2) # lookup tables
    ax12.plot(vlo,nlo,lw=2)

    ax11.text(-80,0.35,r"\textbf{Approx. phase=}") # real time phase
    ax11.text(-80,0.3,r"$\quad$\textbf{"+str(theta1[j,1])+r"*2pi}")

    ax12.text(-80,0.35,r"\textbf{Approx. phase=}")
    ax12.text(-80,0.3,r"$\quad$\textbf{"+str(theta2[j,1])+r"*2pi}")


    # phase diff full + theory + param
    ax21.plot(t[:k][::skipn],dat[:k,1][::skipn]*2*np.pi,color='black',lw=2,label='Numerics')
    N = len(slow_phs_model)
    ax21.plot(np.linspace(0,dat[:,0][-1],N)[:k][::skipn2],slow_phs_model[:k][::skipn2]*2*np.pi,lw=5,color="#3399ff",label='Theory')

    ax21b.plot(t[:k][::skipn2],gm[:k][::skipn2],color='red',lw=2,label='Parameter')

    ax21.legend()
    
    fig.savefig(movdir+str(counter)+".png",dpi=80)
    
    #plt.pause(.01)
    print t[k],k,counter


    plt.cla()
    plt.close()

    counter += 1




plt.show()
