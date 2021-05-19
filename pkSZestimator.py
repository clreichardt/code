import numpy as np
from scipy.spatial import distance as dist
from scipy.signal import fftconvolve
from scipy.integrate import simps, quad
from scipy.special import spherical_jn
from colossus.cosmology import cosmology
from colossus.halo import mass_adv
from colossus.lss.mass_function import modelTinker08 as Tinker08
from colossus.lss.mass_function import massFunction
from utils import *

def DESrichtomass(richness, z):
    M0 = 3.081e14 #Msun
    F = 1.356
    G = -0.30
    return M0*((richness/40)**F)*(((1+z)/1.35)**G)

def j0(x):
    return spherical_jn(0, x)

def j1(x):
    return spherical_jn(1, x)

def fz(z, gamma=4./7.):
    '''
    Lahav et al (1991) approximation for the growth factor
    '''
    return cosmo.Om(z)**gamma

def bias_nu(nu, deltac=1.686, delta_v=200.):
    """
    Halo bias as a function of normalized mass overdensity \nu (Eq.6 from 1001.3162)
    Parameters
    ----------
    nu: float array
        Array of normalized mass overdensity \nu
    Returns
    -------
    b : float array
        Halo bias
    """
    y = np.log10(delta_v)
    A = 1.0 + 0.24*y*np.exp(-(4./y)**4.)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
    c = 2.4
    return (1. - A*nu**a/(nu**a + deltac**a) + B*nu**b + C*nu**c)

def bias_avg(Mmin, Mmax, z):
    size = int(1e5)
    M = np.linspace(Mmin,Mmax,size)
    hmf = massFunction(M*h, z, q_in='M', q_out='dndlnM', mdef='200c', model='tinker08')*(h**3)/M
    nu = get_nu(z, Mmin=Mmin, Mmax=Mmax, lenght=size)
    bh = bias_nu(nu)
    numerator = simps(M*hmf*bh, M)
    denominator = simps(M*hmf, M)
    return numerator/denominator

def rho_bar(z): #[M_sun/kpc^3]
    """
    Returns the comoving matter density at redshift z [M_sun/kpc^3].
    """
    return cosmo.rho_c(z) * cosmo.Om(z) * (h**2)

def W_k_tophat(k, R):
    """
    Returns the Fourier Transform of a tophat window function.
    """
    return 3*(np.sin(k*R) - (k*R)*np.cos(k*R))/((k*R)**3)

def get_nu(z, deltac = 1.686, Mmin = 9e13, Mmax=4e14, lenght=1000):
    M = np.linspace(Mmin, Mmax, lenght)
    R = (3*M/(4*np.pi*rho_bar(z)))**(1./3.) #kPc
    R = R * 1e-3 #MPc
    sigma = cosmo.sigma(R, z)
    return deltac / sigma

def xibarintegrand(r, z):
    return cosmo.correlationFunction(r*h, z) * r * r

def velocitycorrelation(r, z, gamma):
    a = 1/(1+z)
    factor = (a * r * cosmo.Hz(z) * fz(z, gamma=gamma))/3
    integral = np.zeros(len(r))
    for i in range(len(r)):
        xint = np.linspace(1e-3/0.6,r[i],1e5)
        integral[i] = simps(xibarintegrand(xint, z), x = xint)
        integral[i] *= 3/(r[i]**3)
    return -1* factor * integral

def v12(r,z,Mmin,Mmax, gamma):
    vcorr = velocitycorrelation(r, z, gamma=gamma)
    mcorr = cosmo.correlationFunction(r*h, z)
    b = bias_avg(Mmin, Mmax, z)
    numerator = 2 * b * vcorr
    denominator = 1 + (b**2)*mcorr
    return numerator/denominator

def TpkSZmodel(tau, r, z, sigmadc, Mmin, Mmax, gamma):
    Tcmb0 = 2.726e6  #uK
    Tcmb = Tcmb0*(1+z)
    c=3e5 #km/s
    v = v12(r, z, Mmin, Mmax, gamma)
    if sigmadc != 0:
        sigmar = np.sqrt(2) * sigmadc
        damp = 1 - np.exp(-((r)**2)/(2*(sigmar**2)))
    else:
        damp = 1
    return tau * Tcmb * v * damp / c

def TpkSZ_calc(r, tau, z, sigmadc, richmin, richmax, gamma=4./7.):
    '''
    pkSZ = zeros(len(r))
    for i in range(len(r)):
    '''
    M200min = DESrichtomass(richmin, z)*h
    M200max = DESrichtomass(richmax, z)*h
    Mmin, Rnew, Cnew = mass_adv.changeMassDefinitionCModel(M200min, z, '200m', '200c')
    Mmax, Rnew, Cnew = mass_adv.changeMassDefinitionCModel(M200max, z, '200m', '200c')
    pkSZ = TpkSZmodel(tau, r, z, sigmadc, Mmin/h, Mmax/h, gamma)
    return pkSZ

def sigmafit(r, TpkSZ, TpkSZcov, rmin=1, sep_good=None, rmax=301, richmin=20, richmax=60, meanz=0.4866, photoz=None, subsamples=150, printbool=False, plot=False, cosmo=cosmo):
    tau=1
    if photoz is not None:
        sigmadc = 3e5*photoz*(1+meanz)/cosmo.Hz(meanz)
    else:
        sigmadc = 0
    TpkSZfit = TpkSZ[np.where(r>40)]
    rfit = r[np.where(r>40)]
    invc = np.zeros((TpkSZfit.size, TpkSZfit.size))
    invc = ((subsamples - len(rfit) - 2)/(subsamples - 1))*np.linalg.inv(TpkSZcov[TpkSZ.size - TpkSZfit.size:,TpkSZ.size - TpkSZfit.size:])
    rfitbin = np.zeros(len(rfit)+1)
    for i in range(len(rfit)):
        if sep_good is not None:
            if i == 0:
                deltar = (rfit[i+1] - rfit[i])/2
                rfitbin[i] = rfit[i] - deltar
            elif i != len(rfit)-1:
                rfitbin[i] = rfit[i] - deltar
            else:
                rfitbin[i] = sep_good
        else:
            if i == 0:
                deltar = (rfit[i+1] - rfit[i])/2
            rfitbin[i] = rfit[i] - deltar
    rfitbin[-1] = rmax
    pkSZ = np.zeros(len(rfit))
    for i in range(len(rfit)):
        R = np.linspace(rfitbin[i],rfitbin[i+1],100)
        pkSZfull = TpkSZ_calc(R, tau, meanz, sigmadc, richmin, richmax)
        pkSZ[i] = sum(pkSZfull)/len(R)
    sigmatau = 1/np.matmul(pkSZ.T, np.matmul(invc,pkSZ))
    taubest = sigmatau*np.matmul(TpkSZfit.T, np.matmul(invc,pkSZ))
    StoN = taubest/np.sqrt(sigmatau)
    if printbool:
        print('The S/N of the tau fit is: %.2f'%StoN)
    if plot == True:
        R = np.linspace(rmin,rmax,100)
        pkSZtheo = TpkSZ_calc(R, taubest, meanz, sigmadc, richmin, richmax)
        return taubest, np.sqrt(sigmatau), R, pkSZtheo
    else:
        return taubest, np.sqrt(sigmatau), StoN

def gammafit(gamma, r, TpkSZ, TpkSZcov, tau, rmin=1, sep_good=None, rmax=301, richmin=20, richmax=60, meanz=0.4866, photoz=None, subsamples=150, printbool=False, plot=False, cosmo=cosmo):
    if photoz is not None:
        sigmadc = 3e5*photoz*(1+meanz)/cosmo.Hz(meanz)
    else:
        sigmadc = 0
    TpkSZfit = TpkSZ[np.where(r>40)]
    rfit = r[np.where(r>40)]
    invc = np.zeros((TpkSZfit.size, TpkSZfit.size))
    invc = ((subsamples - len(rfit) - 2)/(subsamples - 1))*np.linalg.inv(TpkSZcov[TpkSZ.size - TpkSZfit.size:,TpkSZ.size - TpkSZfit.size:])
    rfitbin = np.zeros(len(rfit)+1)
    for i in range(len(rfit)):
        if sep_good is not None:
            if i == 0:
                deltar = (rfit[i+1] - rfit[i])/2
                rfitbin[i] = rfit[i] - deltar
            elif i != len(rfit)-1:
                rfitbin[i] = rfit[i] - deltar
            else:
                rfitbin[i] = sep_good
        else:
            if i == 0:
                deltar = (rfit[i+1] - rfit[i])/2
            rfitbin[i] = rfit[i] - deltar
    rfitbin[-1] = rmax
    pkSZ = np.zeros(len(rfit))
    chisq = np.zeros(len(gamma))
    for j in range(len(gamma)):
        for i in range(len(rfit)):
            R = np.linspace(rfitbin[i],rfitbin[i+1],100)
            pkSZfull = TpkSZ_calc(R, tau, meanz, sigmadc, richmin, richmax, gamma=gamma[j])
            pkSZ[i] = sum(pkSZfull)/len(R)
        vec = TpkSZfit - pkSZ
        chisq[j] = np.matmul(vec, np.matmul(invc, vec))
        if j == 0:
            chisqbest = chisq[j]
            bestgamma = gamma[j]
        elif chisqbest > chisq[j]:
            chisqbest = chisq[j]
            bestgamma = gamma[j]
    ind = np.where(chisq >= chisqbest+1)
    sigmagamma = min(abs(bestgamma - gamma[ind]))
    StoN = bestgamma/sigmagamma
    if printbool:
        print('The S/N of the gamma fit is: %.2f'%StoN)
    if plot == True:
        R = np.linspace(rmin,rmax,100)
        pkSZtheo = TpkSZ_calc(R, tau, meanz, sigmadc, richmin, richmax, gamma=bestgamma)
        return bestgamma, sigmagamma, R, pkSZtheo, chisq, chisqbest
    else:
        return bestgamma, sigmagamma, StoN, chisq, chisqbest

def parallelsigmafit(r, TpkSZboots, TpkSZcovboot, rmin, sep_good, rmax, richmin, richmax, meanz, photoz, subsamples, plotsign):
    taufitparallel, _, _, _ = sigmafit(r, TpkSZboots, TpkSZcovboot, rmin=rmin, sep_good=sep_good, rmax=rmax, richmin=richmin, richmax=richmax, meanz=meanz, photoz=photoz, subsamples=subsamples, plot=plotsign)
    return taufitparallel
