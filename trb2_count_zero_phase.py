import numpy as np
import matplotlib.pylab as mp

theta1_p = np.loadtxt('trb2_theta1_p.dat')[:,1]
theta1_p_newpar = np.loadtxt('trb2_theta1_p_newpar.dat')[:,1]
theta1_qp = np.loadtxt('trb2_theta1_qp.dat')[:,1]
theta1_s = np.loadtxt('trb2_theta1_s4.dat')[:,1]
theta1_s_x = np.loadtxt('trb2_theta1_s4.dat')[:,0]

mp.plot(theta1_s)
mp.plot(theta1_p)
mp.show()

# brute force local minima. since each point is about 1/300 of one period, and the function is monotonic, we are safe to check neighboring points only.

# from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array

theta1_p_tot = np.sum(np.r_[True, theta1_p[1:] < theta1_p[:-1]] & np.r_[theta1_p[:-1] < theta1_p[1:], True])
theta1_qp_tot = np.sum(np.r_[True, theta1_qp[1:] < theta1_qp[:-1]] & np.r_[theta1_qp[:-1] < theta1_qp[1:], True])
theta1_p_newpar_tot = np.sum(np.r_[True, theta1_p_newpar[1:] < theta1_p_newpar[:-1]] & np.r_[theta1_p_newpar[:-1] < theta1_p_newpar[1:], True])
theta1_s_tot = np.sum(np.r_[True, theta1_s[1:] < theta1_s[:-1]] & np.r_[theta1_s[:-1] < theta1_s[1:], True])

print 'p total cycles', theta1_p_tot
print 'qp total cycles', theta1_qp_tot
print 'p newpar total cycles', theta1_p_newpar_tot
print 's total cycles', theta1_s_tot
