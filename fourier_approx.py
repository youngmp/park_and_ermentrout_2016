"""
Copyright (c) 2016, Youngmin Park, Bard Ermentrout
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ympark1988@gmail.com

Plot data and compute Fourier coefficients

(Thank you Mario for the help and inspiration)

TODO:
1. Let user choose coefficients up to x% of the L2 norm
2. Add code to write coefficients to file
"""
try:
    import matplotlib.pylab as mp
    matplotlib_module = True
except ImportError:
    print "You must have matplotlib installed to generate plots"
    matplotlib_module = False
try:
    import numpy as np
    #np.random.seed(0)
    numpy_module = True
except ImportError:
    print "You must have numpy installed to run this script"
    numpy_module = False
try:
    from scipy.optimize import brentq
    scipy_module = True
except ImportError:
    print "You must have numpy installed to run this script"
    scipy_module = False

def manual_ift(bc,freqc,idxc,N):
    """
    manual inverse fourier transform
    bc: nonzero output from np.fft.fft
    ffreqc: corresponding freq. component
    idxc: corresponding index of freq. component
    """
    # define domain
    n = np.linspace(0,N-1,N)
    tot = 0

    c = 0 # counter
    # for select k, compute value at each n
    for k in idxc:
        tot += np.real(bc[c])*np.cos(k*2*np.pi*n/N) - np.imag(bc[c])*np.sin(k*2*np.pi*n/N)
        c += 1
    return tot

def manual_ift_sin(bc,freqc,idxc,N):
    """
    manual inverse fourier transform for sine terms only
    bc: nonzero output from np.fft.fft
    ffreqc: corresponding freq. component
    idxc: corresponding index of freq. component
    """
    # define domain
    n = np.linspace(0,N-1,N)
    tot = 0

    c = 0 # counter
    # for select k, compute value at each n
    for k in idxc:
        #print np.imag(bc[c])
        tot += np.imag(bc[c])*np.sin(k*2*np.pi*n/N)
        #print k,np.imag(bc[c]),'sin'
        c += 1
    return tot

def manual_ift_cos(bc,freqc,idxc,N):
    """
    manual inverse fourier transform for cosine terms only
    bc: nonzero output from np.fft.fft
    ffreqc: corresponding freq. component
    idxc: corresponding index of freq. component
    """
    # define domain
    n = np.linspace(0,N-1,N)
    tot = 0

    c = 0 # counter
    # for select k, compute value at each n
    for k in idxc:
        tot += np.real(bc[c])*np.cos(k*2*np.pi*n/N)
        #print k,np.real(bc[c]),'cos'
        c += 1

    return tot


def amp_cutoff(x,n,fcoeff):
    # goal: find ideal x s.t. sum(coeff_array_idx) = n
    """
    fcoeff: output from np.fft.fft
    x: cutoff for magnitude of fourier coefficient
    n: desired number of fourier coefficients
    """
    coeff_array_idx = np.absolute(fcoeff) > x
    return sum(coeff_array_idx) - n

def main():
    # load/define data
    dat = np.loadtxt("fourier/hfun.gm_0.5.dat")
    dat = dat[:,1]
    #dat2 = np.sin(np.linspace(0,10,N))
    #dom = np.linspace(0,1000,1000)
    N = len(dat)
    dom = np.linspace(0,N,N)
    #print np.shape(dat), np.shape(dat2)

    # get Fourier transform and frequencies
    fcoeff = np.fft.fft(dat)
    ffreq = np.fft.fftfreq(dat.size)

    # find cutoff x for desired number of coefficients
    n = 5 # desired # of coefficients
    x = brentq(amp_cutoff,0,np.amax(np.abs(fcoeff)),args=(n,fcoeff))
    # array corresponding to desired coefficients
    coeff_array_idx = np.absolute(fcoeff) > x

    # build list of desired coefficients
    b = fcoeff*coeff_array_idx

    # extract corresponding frequencies
    freq = ffreq*coeff_array_idx   
    
    # build lits of only desired coeff & freq
    bc = fcoeff[coeff_array_idx]/N
    freqc = ffreq[coeff_array_idx]
    idxc = np.nonzero(coeff_array_idx)[0]
    
    #print bc
    # come back to time domain
    c = np.fft.ifft(b)

    # or
    c2 = manual_ift(bc,freqc,idxc,N)

    # for sine/cosine component only:
    c3 = manual_ift_sin(bc,freqc,idxc,N)
    c4 = manual_ift_cos(bc,freqc,idxc,N)
    
    
    # add option to write coefficients to file
    if True:
        pass

    if matplotlib_module:
        mp.figure()
        mp.plot(dat)
        mp.plot(c4-c3)
        #mp.plot(c)
        #mp.plot(c2)
        #mp.plot(c3)
        #mp.plot(c4)
        
        mp.show()
    
if __name__ == "__main__":
    main()
