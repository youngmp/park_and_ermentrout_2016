"""
This script is for the dirty work.

Get Fourier coefficients
Plot sine/cosine coefficients as a function of gm
Plot other data

"""

import matplotlib.pylab as mp
import numpy as np
import copy
from fourier_approx import *

def main():
    parvals = np.arange(.1,.5+.02,.02)
    Ndat = len(parvals)
    grayscale_val = 0.0

    # plot sine and attempt a fit
    # consider using maximum log-likelihood
    #b1_amp = np.loadtxt("fourier/b1_amp.dat")

    """
    t = (parvals-0.12)/0.36
    y = (1-t)+t*19
    f1 = -8.16660825042733 + 9.34618477203906*np.cos(y/10.)
    f2 = 1.465128339976848 - 0.1851492051170480*(y) - 0.02657459947389852*(y)**2
    f3 = 1.273107726417173 - 0.082821337025290*y - 0.0390434704842670*y**2 + 0.000415629033678951*y**3
    err1 = np.linalg.norm(f1-b1_amp)
    err2 = np.linalg.norm(f2-b1_amp)
    err3 = np.linalg.norm(f3-b1_amp)
    
    print "cos, quad, cubic errors"
    print err1,err2,err3
    """
    #mp.figure()
    #mp.title("blue: cosine fit; green: quadratic fit; dots: data")
    #mp.scatter(parvals,b1_amp)
    #mp.plot(parvals,f1,lw=2,color="blue")
    #mp.plot(parvals,f2,lw=2,color="green")



    counter = 0
    #difflist = np.zeros(Ndat-1)
    amplist = np.zeros(Ndat)

    for gm in parvals:
        hfun = np.loadtxt("fourier/hfun.gm_"+str(gm)+".dat")
        #hfun = np.loadtxt("fourier_i/hfun.i_"+str(gm)+".dat")
        orbit = np.loadtxt("fourier/orbit.gm_"+str(gm)+".dat")
        #adjoint = np.loadtxt("fourier/adjoint.gm_"+str(i)+".dat")
        print 'period at gm=',gm, ':', orbit[-1,0]
        """
        #begin Fourier approximation
        """
        if True:
            dat = hfun[:,1]
            N = len(dat)
            dom = np.linspace(0,N,N)
            
            fcoeff = np.fft.fft(dat)
            ffreq = np.fft.fftfreq(dat.size)
            
            n = 5
            x = brentq(amp_cutoff,0,np.amax(np.abs(fcoeff)),args=(n,fcoeff))
            coeff_array_idx = np.absolute(fcoeff) > x
            
            # get coefficients
            bc = fcoeff[coeff_array_idx]/N
            freqc = ffreq[coeff_array_idx]
            
            idxc = np.nonzero(coeff_array_idx)[0]
            #print gm,bc
            # write coefficients to file
            if False:
                # save coefficient, freq, idx, and size N
                np.savetxt("hfun.fourier.coeff.gm_"+str(gm)+".dat",(bc,freqc,idxc))
                #np.savetxt("hfun.fourier.freq.gm_"+str(i)+".dat",freqc)
                #np.savetxt("hfun.fourier.idx.gm_"+str(i)+".dat",idxc)
                np.savetxt("hfun.fourier.N.gm_"+str(gm)+".dat",np.array([N]))

            
            # come back to time domain
            c3 = manual_ift_sin(bc,freqc,idxc,N)
            c4 = manual_ift_cos(bc,freqc,idxc,N)
            

            # test inverse
            temp = copy.deepcopy(fcoeff)
            #print coeff_array_idx[:10]
            idx = np.argsort(abs(temp))

            c4 = np.fft.ifft(temp,n=N)
            n = np.linspace(0,N-1,N)
            if gm-.0001<.3 and gm+.0001>.3:
                
                mp.figure()
                mp.title('')
                mp.plot(2*np.imag(fcoeff)/N)
        


        """
        #end fourier approximation
        """

        """
        #Find amplitude of sine/cosine fuction as a function of gm
        """
        if True:
            # choose coefficient index. c=0 is the constant coefficient.
            coeffidx = 1
            amplist[counter] = np.imag(bc[coeffidx])
            print np.imag(bc[1:6]), '5 sine coeff excluding 0, n=',counter, 'gm =',gm
            print freqc[1:6]

        if False:
            # choose coefficient index. c=0 is the constant coefficient.
            coeffidx = 2
            amplist[counter] = np.real(bc[coeffidx])
            #print bc[coeffidx]

            
        """
        #misc and plotting
        """

        #phivals = np.linspace(0,1,len(dat))
        neg2hodd = (np.flipud(hfun[:,1])-hfun[:,1])
        #mp.plot(np.linspace(0,1,len(dat)),neg2hodd,color=str(grayscale_val))

        #mp.plot(np.linspace(0,1,len(dat)),-neg2hodd/2.,color=str(grayscale_val))
        #mp.plot(np.linspace(0,1,N),c3,color=str(grayscale_val))

        mp.plot(np.linspace(0,1,len(dat)),hfun[:,1],color=str(grayscale_val))
        mp.plot(np.linspace(0,1,N),c3+c4,color=str(grayscale_val))

        #mp.plot(np.linspace(0,1,len(hfun[:,1])),hfun[:,1],color=str(grayscale_val))
        
        # check if increasing or decreasing at origin
        #mp.title("blue:slope of neg2hodd; green:slope of approx. (at origin)")
        #mp.scatter(i,c3[1]-c3[0],color=(0,0.25+grayscale_val,0))
        #mp.scatter(i,neg2hodd[1]-neg2hodd[0],color=(0,0,0.25+grayscale_val))
        #mp.xlabel("gm")
        #mp.ylabel("(slope at origin)*dt")
        
        # Plot frequency values as a function of gm.
        #mp.scatter(gm,freqc[coeffidx],color=(0,grayscale_val,0))


        #mp.plot(np.linspace(0,1,len(orbit[:,0])),orbit[:,1],color=str(grayscale_val))
        #mp.plot(np.linspace(0,1,len(adjoint[:,0])),adjoint[:,1],color=str(grayscale_val))
        grayscale_val += 0.75/Ndat
        counter += 1


    # print parvals
    #print 0, amplist[0]
    #print .3, amplist[(parvals < 0.3+.01) * (parvals > 0.3-.01)][0]

    gm_amplist = np.zeros((len(amplist),2))
    for k in range(len(amplist)):
        gm_amplist[k,0] = parvals[k]
        gm_amplist[k,1] = amplist[k]
    if False:
        np.savetxt("b"+str(coeffidx)+"_amp.dat",gm_amplist)
    if True:
        mp.figure()
        # plot amplitudes; modify for pos values that should be neg
        #minval = np.amin(amplist) # need for coeff=1
        #idx = np.where(amplist==minval)[0][0] # need for coeff=1
        #amplist[idx:] = -amplist[idx:] # need for coeff=1
        #print parvals[0],parvals[tempidx]
        #print amplist[0],amplist[np.where(parvals==0.3)]
        for i in range(len(parvals)):
            print parvals[i],amplist[i]

        mp.plot(parvals, amplist)

    mp.show()

if __name__ == "__main__":
    main()
