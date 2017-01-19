from math import *
from numpy import *
from scipy import *
from scipy.integrate import quad
from scipy.interpolate import splrep, splev

# Paolo Serra: 04 / 08 / 2014
# program to compute both angular power spectra and
# cross-spectra for different dark matter tracers

# Define the model for dS / dz
mu = 0.201577E+01
sigma = 0.595684E+00
L0 = 3.75e5
# Planck freqs to be used (in Hz)
nu = [3.53e11, 5.45e11, 8.57e11]
nfreq = len(nu)
jy2muk = [287.2262, 57.9766, 2.26907]
# amplitudes of the redshift distribution from
# cosmomc best fit parameters
A = [0.321364E+00, 0.886381E+00, 0.170816E+01]
# cosmological model used
om = 0.3188
H0 = 67.07  # km / s / Mpc
chistar = 14184.9421574498
# speed of light in km / s
clight = 299792.458

#  read in the CAMB matter power spectrum file at redshift 0, P(k,z=0); the
knum, pk = loadtxt("test_matterpower1.dat", unpack = True)
# transform the output in 1./Mpc and Mpc^3
knum = knum * H0 / 100.0
pk = pk / (H0 / 100.0)**3

#min and max k-values
kmin = min(knum)
kmax = max(knum)
print(kmin, kmax)
# python module to spline a function
tck = splrep(knum, pk)

#number of z-intervals
numz = 100
z_int = zeros(numz)

dSdz_int = zeros([nfreq, numz])
intdsdz = zeros(numz)
zmin = -2
zmax = 1

#parameters for an analytic galaxy redshift distribution
for i in range(numz):
    logz = - zmin / (numz-1.) * i -2.0
    z_int[i] = 7.0 * 10**(logz)
    intdsdz[i] = L0 / (sigma * sqrt(2.0 * math.pi)) * exp(-(z_int[i] - mu)**2 / (2.0 * sigma**2))
    for j in range(nfreq):
        dSdz_int[j, i] = A[j] * intdsdz[i]
#normalization for the dSdz distribution
#outfile = open("dsdz.txt","w")
#for i in range(numz):
#    outfile.write('%g' % z_int[i] + " " + " " + '%g' % dSdz_int[0, i] + " " + " " + '%g' % dSdz_int[1, i] + " " + " " + dSdz_int[2, i] + " " + " " + "\n")

chi = zeros(numz)
hubble = zeros(numz)
vol = zeros(numz)
D = zeros(numz)
# total num of frequencies among auto and cross-spectra
maxnfreq = int(nfreq * (nfreq + 1) / 2)
intg_auto = zeros([maxnfreq, numz])
intg_cross = zeros([nfreq, numz])

# normalization term for the calculation of the growth factor, see below
norm = quad(lambda x:(1.0+x) / (om * (1.0 + x)**3 + (1.0 - om))**(3./2.), 0.0, +inf)

numk = 100
el = zeros(numk)
cl_cross = zeros([nfreq, numk])
cl_auto = zeros([maxnfreq, numk])
intcl_cross = zeros([nfreq, numk])
intcl_auto = zeros([maxnfreq, numk])

# cycle over the angular scales, up to l = 2048
for j in range(numk):
    kth = 10**(3.28 * j / (numk - 1)) / 21600.0
# cycle over the redshift
    for i in range(numz):
        z = z_int[i]
        # Hubble factor H(z) in Km/s/Mpc
        hubble[i] = H0 * (om * (1.+z)**3 + (1. - om))**(0.5)
#       integral to compute the comoving distance
        intdis = quad(lambda x:1.0 / (sqrt(om * (1.0 + x)**3 +(1.0-om))), 0.0, z)

        #comoving distance in Mpc
        chi[i] = clight / H0 * intdis[0]
        # wavenumber k in function of the inverse of angular scale and comoving distance. In general: k=2*pi*k_theta/chi(z)
        k_int = 360.0 * 60.0 * kth / chi[i]
        int3dis = quad(lambda x:(1.0+x)/(om*(1.0+x)**3+(1.0-om))**(1.5),z,+inf)
    #growth factor, see Barkana Loeb 0010468 or http://www.astronomy.ohio-state.edu/~dhw/A873/notes8.pdf
        D[i] = sqrt(om *(1.+z)**3 + (1.0 - om)) * int3dis[0] / norm[0]
#comoving volume element in Mpc^3
        vol[i] = chi[i]**2 * clight / hubble[i]

        if k_int >= kmin and k_int <= kmax:
            intgd1 = splev(k_int, tck) * D[i]**2 * (1.0 + z) * (3.0 * om * (H0 / clight)**2 /(360.0 * 60.0 * kth)**2 * (chistar - chi[i])/ (chistar * chi[i]))
#            print chistar, chi[i], chistar - chi[i]
            intgd2 = splev(k_int, tck) / vol[i] * D[i]**2
            q = -1
            for k1 in range(nfreq):
                intg_cross[k1, i]= intgd1 * dSdz_int[k1, i]

                for k2 in range(k1, nfreq):
                    q += 1
                    intg_auto[q, i] = intgd2 * dSdz_int[k1, i] * dSdz_int[k2, i]
        else:
            print(kint, kmin, kmax)
            intg_cross[k1, i] = 0.
            intg_auto[q, i] = 0

# best fit bias for CIB and LRG: see Lagache et al. (2011), Planck collaboration, pag. 22 and Ho et al. (2012)
    biasCIB = 2.5
    for k1 in range(nfreq):
        q = -1
        intcl_cross[k1, j] = trapz(intg_cross[k1, :], z_int[:])
        cl_cross[k1, j] = biasCIB * intcl_cross[k1, j]
        for k2 in range(k1, maxnfreq):
            q += 1
            intcl_auto[q, j] = trapz(intg_auto[q, :], z_int[:])
            cl_auto[q, j]  = biasCIB**2 * intcl_auto[q, j]
    el[j] = 360.0 * 60.0 * kth
    multiel = 1.#el[j]*(el[j]+1.0)/(2.0*pi)
#   total CIB cl based on linear term+shot-noise taken from Lagache et al. (2011)
# simple power law; it provides a decent fit to the data!
#    power_law[j]=multiel*11400.0*(el[j]/1000.0)**(-1.09)


#interpolation of C_l functions
q = -1
cross = zeros(numk)
auto = zeros([maxnfreq, numk])
cross0 = splrep(el, cl_cross[0, :])
cross1 = splrep(el, cl_cross[1, :])
cross2 = splrep(el, cl_cross[2, :])
auto0 = splrep(el, cl_auto[0, :])
auto1 = splrep(el, cl_auto[1, :])
auto2 = splrep(el, cl_auto[2, :])
auto3 = splrep(el, cl_auto[3, :])
auto4 = splrep(el, cl_auto[4, :])
auto5 = splrep(el, cl_auto[5, :])
#        auto[q, :] = splrep(el[:], cl_auto[q, :])
lmax = 2048
ell = zeros(lmax)
cl_cross0 = zeros(lmax)
cl_cross1 = zeros(lmax)
cl_cross2 = zeros(lmax)
cl_auto0 = zeros(lmax)
cl_auto1 = zeros(lmax)
cl_auto2 = zeros(lmax)
cl_auto3 = zeros(lmax)
cl_auto4 = zeros(lmax)
cl_auto5 = zeros(lmax)
clpolxphi = zeros(lmax)
outfile1 = open("all_spectra.txt","w") #Paolo
outfile2 = open("spectraTxphi_allell_2048.txt","w")
for i in range(lmax):
    ell[i] = 1.0 * i
#    if i <= 16:
#        cl_cross0[i] = 0
#        cl_cross1[i] = 0
#        cl_cross2[i] = 0
#        cl_auto0[i] = 0
#        cl_auto1[i] = 0
#        cl_auto2[i] = 0
#        cl_auto3[i] = 0
#        cl_auto4[i] = 0
#        cl_auto5[i] = 0
#        clpolxphi[i] = 0
#    else:
    # in muK
    cl_cross0[i] = ell[i]**3 * splev(ell[i], cross0) / jy2muk[0]
    cl_cross1[i] = ell[i]**3 * splev(ell[i], cross1) / jy2muk[1]
    cl_cross2[i] = ell[i]**3 * splev(ell[i], cross2) / jy2muk[2]
    cl_auto0[i] = splev(ell[i], auto0) #+ (605 - 164)
    cl_auto1[i] = splev(ell[i], auto1) #+ (1508 - 452)
    cl_auto2[i] = splev(ell[i], auto2) #+ (2502 - 872)
    cl_auto3[i] = splev(ell[i], auto3) #+ (4239 - 1249)
    cl_auto4[i] = splev(ell[i], auto4) #+ (7377 - 2407)
    cl_auto5[i] = splev(ell[i], auto5) #+ (14593 - 4539)
    outfile1.write('%g' % ell[i] + " " + " " + '%g' % cl_auto0[i] + " " + " " + '%g' % cl_auto1[i]+  " "  + " " + " " + '%g' % cl_auto2[i] + " " + " " + '%g' % cl_auto3[i] + " " + " " + '%g' % cl_auto4[i] + " " + " " + '%g' % cl_auto5[i] + " " + " " + '%g' % cl_cross0[i] + " " + " " + '%g' % cl_cross1[i] + " " + " " + '%g' % cl_cross2[i] + "\n")

    outfile2.write('%g' % ell[i] + " " + " " + '%g' % cl_cross0[i] + " " + " " + '%g' % cl_cross1[i] + " " + " " + " " + " " + '%g' % cl_cross2[i] + " " + " " + " " + '%g' % cl_cross0[i] + " " + " " + " " + " " + '%g' % cl_cross1[i] + " " + " " + " " + " " + '%g' % cl_cross2[i] + " "  + " " + " " + "\n")
