"""
This script is for the dirty work.

Get Fourier coefficients
Plot sine/cosine coefficients as a function of gm
Plot other data

"""

import matplotlib.pylab as mp
import numpy as np
from fourier_approx import *

def main():
    parvals = np.arange(.05,.3+.0125,.0125)
    Ndat = len(parvals)
    grayscale_val = 0.0

    counter = 0
    #difflist = np.zeros(Ndat-1)
    amplist = np.zeros(Ndat)

    for gm in parvals:
        hfun = np.loadtxt("fourier/hfun.gm_"+str(gm)+".dat")
        #hfun = np.loadtxt("fourier_i/hfun.i_"+str(gm)+".dat")
        #orbit = np.loadtxt("fourier/orbit.gm_"+str(i)+".dat")
        #adjoint = np.loadtxt("fourier/adjoint.gm_"+str(i)+".dat")
        """
        #begin Fourier transformation
        """
        dat = hfun[:,1]
        N = len(dat)
        if True:
            dom = np.linspace(0,N,N)
            
            fcoeff = np.fft.fft(dat)
            
            ffreq = np.fft.fftfreq(dat.size)
            
            n = 40
            x = brentq(amp_cutoff,0,np.amax(np.abs(fcoeff)),args=(n,fcoeff))
            coeff_array_idx = np.absolute(fcoeff) > x
            
            # get coefficients
            bc = fcoeff[coeff_array_idx]/N

            freqc = ffreq[coeff_array_idx]
            
            idxc = np.nonzero(coeff_array_idx)[0]
            #print gm,bc[:6]
            # write coefficients to file
            if False:
                # save coefficient, freq, idx, and size N
                np.savetxt("hfun.fourier.coeff.gm_"+str(gm)+".dat",(bc,freqc,idxc))
                #np.savetxt("hfun.fourier.freq.gm_"+str(i)+".dat",freqc)
                #np.savetxt("hfun.fourier.idx.gm_"+str(i)+".dat",idxc)
                np.savetxt("hfun.fourier.N.gm_"+str(gm)+".dat",np.array([N]))

            # come back to time domain
            c3 = manual_ift_sin(bc,freqc,idxc,N)
            n = np.linspace(0,N-1,N)
        """
        #end fourier transformation
        """


        """
        #Find amplitude of sine/cosine fuction as a function of gm
        """
        if True:
            # choose coefficient index. c=0 is the constant coefficient.

            
            coeffidx = 12
            
            amplist[counter] = np.imag(bc[coeffidx])
            #amplist[counter] = (np.imag(fcoeff)/N)[:90][coeffidx]
            
            #print np.imag(bc[1:]), '5 sine coeff excluding 0, n=',counter
            #print freqc[1:], 'frec'
            #print idxc[1:], 'idxc'
            #print (idxc/(1.*N))[1:], 'should be same as freqc'

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
        mp.plot(np.linspace(0,1,len(dat)),neg2hodd,color=str(grayscale_val))
        #mp.plot(np.linspace(0,1,N),2*c3,color=str(grayscale_val))
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
        #for i in range(len(parvals)):
        #    print parvals[i],amplist[i]
        mp.plot(parvals, amplist)

        # do quadratic fit
        coeff = np.polyfit(parvals,amplist,3)
        print coeff,coeffidx
        print 'amplist:',amplist
        mp.plot(parvals,coeff[0]*parvals**3+coeff[1]*parvals**2+coeff[2]*parvals+coeff[3])
    mp.show()

if __name__ == "__main__":
    main()
