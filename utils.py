import numpy as np
import sys
sys.path.append('/home/eschiappucci/spt3g_software/build')
from scipy.special import kn
from colossus.halo import mass_adv
from colossus.cosmology import cosmology

from spt3g import core, mapmaker, mapspectra, maps
from astropy.io import fits
from scipy.integrate import simps, quad
from scipy import interpolate

RA_SPT3g_min  = -50   #310
RA_SPT3g_max  = 50
DEC_SPT3g_min = -70
DEC_SPT3g_max = -42
boxsize         = 40 # arcmin
reso            = 0.25 #spt150Tmap.x_res/core.G3Units.arcmin #arcmin
M0 = 3.081e14 #Msun
F = 1.356
G = -0.30

cosmo = cosmology.setCosmology('planck18-only')#planck18-only WMAP7
print(cosmo.getName())
h = cosmo.Hz(0)/100

def betaprof(theta, thetac=0.5, beta=1):
    '''
    Calculates the projected isothermal Beta model given by (Cavaliere 1976)
    '''
    return (1 + (theta**2)/(thetac**2))**(-beta)

def beta1fourier(t, thetac=np.deg2rad(0.5/60)):
    '''
    Fourier Transform of the beta profile with beta = 1
    '''
    return kn(0,thetac*t) * (thetac**2)

def nfwprof(theta, A=1, thetac=0.5):
    '''
    Calculates the projected NFW profile given by (Bartelmann 1996)
    '''
    x = theta/thetac
    assert(x[0]>0)
    fx = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < 1:
            fx[i] = 1 - 2*np.arctanh(np.sqrt((1-x[i])/(1+x[i])))/np.sqrt(1 - x[i]**2)
        elif x[i] == 1:
            fx[i] = 0
        elif x[i] > 1:
            fx[i] = 1 - 2*np.arctanh(np.sqrt((x[i]-1)/(1+x[i])))/np.sqrt(x[i]**2 - 1)
    return A*fx/(x**2 - 1)

def gaussianbeam(theta, sigma=1, norm=True):
    '''
    Calculates the gaussian beam
    '''
    if norm == True:
        A = 1/np.sqrt(2*np.pi*sigma**2)
    else:
        A = 1
    return A*np.exp(- (theta**2)/(2*sigma**2))

def transfunc(ell, elllow=500, ellhigh=20000):
    '''
    Estimate of the Tranfer functionn in ell space
    '''
    return np.exp(-(elllow/ell)**6) * np.exp(-(ell/ellhigh)**6)

def int2d(data, xrang = np.zeros(1), yrang = np.zeros(1)):
    '''
    Computes the 2D integral of a 2D data array by using a rectangular grid and Simpson's method
    '''
    if (xrang.size != 1) & (yrang.size != 1):
        assert(data.shape[0] == xrang.size)
        assert(data.shape[1] == yrang.size)
        return simps([simps(data_x,xrang) for data_x in data],yrang)
    else:
        return simps(simps(data, range(data.shape[0])), range(data.shape[1]))

def fn_load_halo(fname, zmin=0., zmax=1., lambdamin=10, lambdamax=100, lambdatilde=False, nobj=None, like_SPT3g=False):
    """
    Returns a RA, DEC, Z with redmapper catalog info.
    """
    cat = fits.open(fname)[1].data
    cat = cat[(cat.Z_LAMBDA >= zmin) & (cat.Z_LAMBDA <= zmax)]
    cat = cat[(cat.LAMBDA_CHISQ/cat.SCALEVAL >= lambdamin) & (cat.LAMBDA_CHISQ/cat.SCALEVAL < lambdamax)]
    if like_SPT3g:
        cat.RA[(cat.RA>180)] = cat.RA[(cat.RA>180)] - 360
        cat = cat[((cat.RA >= RA_SPT3g_min) & (cat.RA <= RA_SPT3g_max))]
        cat = cat[(cat.DEC >= DEC_SPT3g_min) & (cat.DEC <= DEC_SPT3g_max)]

    if nobj is not None:
        cat = cat[np.random.randint(0,len(cat),size=nobj)]

    ra    = cat.RA
    dec   = cat.DEC
    zs    = cat.Z_LAMBDA
    zs_err = cat.Z_LAMBDA_E
    if lambdatilde == True:
        lamb  = cat.LAMBDA_CHISQ/cat.SCALEVAL
    else:
        lamb  = cat.LAMBDA_CHISQ

    return ra, dec, zs, zs_err, lamb

def fn_load_ptsrc(fname, RAcol='col2', DECcol='col3', SNcol='col5'):
    catalog_point = ascii.read(fname)
    rasrc, decsrc, SN  = catalog_point['col2'], catalog_point['col3'], catalog_point['col5']
    rasrc[(rasrc>180)] = rasrc[(rasrc>180)] - 360
    return rasrc, decsrc, SN

def cluster_discard(rasrc, decsrc, ra, dec, z, photoz, lamb, apodsize=1, srcrange=10/60):
    cond = np.concatenate((np.where(RA_SPT3g_min+apodsize > rasrc)[0], np.where(RA_SPT3g_max-apodsize < rasrc)[0], np.where(DEC_SPT3g_min+apodsize > decsrc)[0], np.where(DEC_SPT3g_max-apodsize < decsrc)[0]))
    rasrc = np.delete(rasrc, cond)
    decsrc = np.delete(decsrc, cond)

    cond = np.concatenate((np.where(RA_SPT3g_min+apodsize > ra)[0], np.where(RA_SPT3g_max-apodsize < ra)[0], np.where(DEC_SPT3g_min+apodsize > dec)[0], np.where(DEC_SPT3g_max-apodsize < dec)[0]))
    ra = np.delete(ra, cond)
    dec = np.delete(dec, cond)
    z = np.delete(z, cond)
    photoz = np.delete(photoz, cond)
    lamb = np.delete(lamb, cond)

    print(len(ra))
    for i in range(rasrc.size):
        cond = np.where((rasrc[i]+srcrange > ra) & (rasrc[i]-srcrange < ra) & (decsrc[i]+srcrange > dec) & (decsrc[i]-srcrange < dec))[0]
        ra = np.delete(ra, cond)
        dec = np.delete(dec, cond)
        z = np.delete(z, cond)
        photoz = np.delete(photoz, cond)
        lamb = np.delete(lamb, cond)
    return rasrc, decsrc, ra, dec, z, photoz, lamb

def padproj(sptmap, padding=(4000,4000), squaremap=True, proj0=True):
    sptmap = np.pad(np.asarray(sptmap), padding, mode='constant')

    if squaremap == True:
        sptmap = sptmap[:,int((sptmap.shape[1]/2)-(sptmap.shape[0]/2)):int((sptmap.shape[1]/2)+(sptmap.shape[0]/2))]

    sptmap = maps.FlatSkyMap(sptmap.copy(order='C'), 0.25 * core.G3Units.arcmin, weighted=False, proj=maps.MapProjection.Proj5, alpha_center=0 * core.G3Units.deg, delta_center=-57.5 * core.G3Units.deg, coord_ref=maps.MapCoordReference.Equatorial, units=core.G3TimestreamUnits.Tcmb, pol_type=maps.MapPolType.T,)
    if proj0 == True:
        MapProj0 = map_3g.Clone(False)
        MapProj0.proj = maps.MapProjection.Proj0
        maps.reproj_map(map_3g, MapProj0, interp=True)
        sptmap = MapProj0
    return sptmap

def makemask(apodmap, rasrc, decsrc, reso=0.25, boxsize=40, gausssigma=2):
    for i in tqdm(range(rasrc.size)):
        if (i == 0):
            col = boxsize/reso
            row = boxsize/reso
            indsy = np.arange(-row/2,row/2)
            indsx = np.ones(int(col))
            Y = np.outer(indsy, indsx)
            indsy = np.ones(int(row))
            indsx = np.arange(-col/2,col/2)
            X = np.outer(indsy, indsx)
            r = reso*np.sqrt(X**2 + Y**2)
            gaussmask = 1 - gaussianbeam(r, sigma=2, norm = False)
            del r, X, Y, indsx, indsy
        ra_tmp = float(rasrc[i])
        dec_temp = float(decsrc[i])
        x, y = apodmap.angle_to_xy(ra_tmp*core.G3Units.deg, dec_temp*core.G3Units.deg)
        apodmask[int(y-boxsize/(2*reso)):int(y+boxsize/(2*reso)),int(x-boxsize/(2*reso)):int(x+boxsize/(2*reso))] = gaussmask

    return apodmask

def circaptfilt(r, rfilt=0.5, norm=True):
    '''
    Calculates the circular aperture filter using an r vector
    '''
    filt = np.zeros(r.shape)
    filt[r <= rfilt] = 1
    filt[(rfilt < r) & (r <= np.sqrt(2)*rfilt)] = -1
    if norm == True:
        return filt/(np.pi*(rfilt**2))
    elif norm == False:
        return filt

def GetMatchFilter2d(map2d, bells, bl, mapreso=0.25, beamwidth=1.27, thetacore=0.5, beta=1., lowl=500, highl=20000):
    '''
    Calculates the match filter of a 2D noisy map measured witha a Gaussian Beam.
    This assume this signal follows a beta-profile and includes a band pass transfer function in x direction
    '''
    ells = np.arange(50, 20050, 50)
    spt150ps = mapspectra.map_analysis.calculate_powerspectra({"T":map2d}, lbins=ells)
    spt150ps = spt150ps['TT']
    ells = (ells[:-1]+ells[1:])/2
    sigmareal = np.deg2rad(beamwidth/60)/2.355
    sigmafourier = 1 /sigmareal
    ell = np.ones((1, map2d.shape[0]))*np.fft.fftfreq(map2d.shape[0], mapreso*np.pi/(180*60))*2*np.pi
    ellx = np.ones(map2d.shape)* ell.T
    ell = np.ones((1, map2d.shape[1]))*np.fft.fftfreq(map2d.shape[1], mapreso*np.pi/(180*60))*2*np.pi
    elly = (np.ones(map2d.shape).T)*ell.T
    elly = elly.T
    L = np.hypot(ellx, elly)
    tfunc = transfunc(elly, elllow=lowl, ellhigh=highl)
    del ellx, elly
    ellx = np.fft.fftfreq(map2d.shape[0], mapreso*np.pi/(180*60))*2*np.pi
    elly = np.fft.fftfreq(map2d.shape[1], mapreso*np.pi/(180*60))*2*np.pi
    limits = [-max(ellx), max(ellx), -max(elly), max(elly)]
    fi = interpolate.interp1d(bells, bl, kind='quadratic', fill_value='extrapolate')
    gauss_2d = fi(ells)
    gauss_2d[np.where(gauss_2d<1e-7)] = 1e-7
    beta_2d = beta1fourier(ells, thetac=np.deg2rad(thetacore/60))
    signal = gauss_2d * beta_2d
    fi = interpolate.interp1d(ells, signal, kind='quadratic', fill_value='extrapolate')
    signal = fi(L.ravel())
    signal[np.where(abs(L.ravel())>12000)] = 0
    signal = signal.reshape(L.shape)
    signal *= tfunc
    fi = interpolate.interp1d(ells, spt150ps, kind='quadratic', fill_value='extrapolate')
    noise_2d = fi(L.ravel())
    noise_2d[np.where(abs(L.ravel())>max(ells))] = 1e-20
    noise_2d = noise_2d.reshape(L.shape)
    del gauss_2d, beta_2d, ells, spt150ps, L
    signal[np.isnan(signal)] = 0
    sigma_psi = 1/int2d(np.fft.fftshift(abs(signal)**2 / noise_2d),xrang=np.fft.fftshift(ellx),yrang=np.fft.fftshift(elly) )
    psi_2d = sigma_psi * signal / noise_2d
    return psi_2d, limits

def GetNoiseMatchFilter2d(noise_2d, bells, bl, mapreso=0.25, thetacore=0.5, beta=1., lowl=500, highl=20000):
    '''
    Calculates the match filter of a 2D noisy map measured witha a Gaussian Beam.
    This assume this signal follows a beta-profile and includes a band pass transfer function in x direction
    '''
    ell = np.ones((1, noise_2d.shape[0]))*np.fft.fftfreq(noise_2d.shape[0], mapreso*np.pi/(180*60))*2*np.pi
    ellx = np.ones(noise_2d.shape)* ell.T
    ell = np.ones((1, noise_2d.shape[1]))*np.fft.fftfreq(noise_2d.shape[1], mapreso*np.pi/(180*60))*2*np.pi
    elly = (np.ones(noise_2d.shape).T)*ell.T
    elly = elly.T
    L = np.hypot(ellx, elly)
    tfunc = transfunc(ellx, elllow=lowl, ellhigh=highl)
    del ellx, elly, ell
    fi = interpolate.interp1d(bells, bl, kind='quadratic', fill_value='extrapolate')
    gauss_2d = fi(L.ravel())
    gauss_2d = gauss_2d.reshape(L.shape)
    beta_2d = beta1fourier(L, thetac=np.deg2rad(thetacore/60))
    signal = gauss_2d * beta_2d * tfunc
    del gauss_2d, beta_2d, L
    signal[np.isnan(signal)] = 0
    sigma_psi = 1/int2d(np.fft.fftshift(abs(signal)**2 / noise_2d),xrang=np.fft.fftshift(ell),yrang=np.fft.fftshift(ell) )
    psi_2d = sigma_psi * signal / noise_2d

    return psi_2d, limits


def GetILCweights2d(mapsvec, M, ells, avec, bvec=None, reso = 0.25, lowl = 50, highl = 20050, deltal = 50, lowpass=None, highpass=None):
    '''
    Calculates the match filter of 2D noisy maps with given beam functions.
    This assume this signal follows a beta-profile, with beta=1., and includes a band pass transfer function in x direction.
    '''
    if bvec is not None:
        assert(avec.shape == bvec.shape)
        N = mapsvec.shape[0]
        a = np.ones((1, N, len(ells)))
        b = np.ones((1, N, len(ells)))
        aT = np.ones((N, 1, len(ells)))
        bT = np.ones((N, 1, len(ells)))
        for i in range(0,N):
            a[0,i] = avec[i]
            b[0,i] = bvec[i]
            aT[i,0] = avec[i]
            bT[i,0] = bvec[i]
        M = np.linalg.inv(M.T).T
        bnlb = np.matmul(bT.T, np.matmul(M.T, b.T))[:,0,0]
        anla = np.matmul(aT.T, np.matmul(M.T, a.T))[:,0,0]
        anlb = np.matmul(aT.T, np.matmul(M.T, b.T))[:,0,0]
        sigma = ((anla*bnlb) - anlb**2)
        psi = ((bnlb * np.matmul(aT.T, M.T).T) - (anlb * np.matmul(bT.T, M.T).T)) / sigma
        ell = np.ones((1, mapsvec.shape[1]))*np.fft.fftfreq(mapsvec.shape[1], reso*np.pi/(180*60))*2*np.pi
        limits = [-max(ell.ravel()), max(ell.ravel())]
        ellx = np.ones(mapsvec[0].shape)* ell.T
        ell = np.ones((1, mapsvec.shape[2]))*np.fft.fftfreq(mapsvec.shape[2], reso*np.pi/(180*60))*2*np.pi
        elly = (np.ones(mapsvec[0].shape).T)*ell.T
        elly = elly.T
        L = np.hypot(ellx, elly)
        del ellx, elly, ell
        weights = np.ones(mapsvec.shape)
        for i in range(0,N):
            x = np.append(ells, ells[-1] + 500)
            y = np.append(psi[i,0], 0)
            fi = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')
            weight_temp = fi(L.ravel())
            weight_temp[np.where(abs(L.ravel())>max(x))] = 0
            weights[i] = weight_temp.reshape(L.shape)
            if (lowpass is not None) and (highpass is not None):
                if i == 0:
                    tfunc = transfunc(L, elllow=lowpass, ellhigh=highpass)
                weights[i] *= tfunc
        return weights, limits
    else:
        N = mapsvec.shape[0]
        a = np.ones((1, N, len(ells)))
        aT = np.ones((N, 1, len(ells)))
        for i in range(0,N):
            a[0,i] = avec[i]
            aT[i,0] = avec[i]
        M = np.linalg.inv(M.T).T
        anla = np.matmul(aT.T, np.matmul(M.T, a.T))[:,0,0]
        psi = np.matmul(aT.T, M.T).T / anla
        ell = np.ones((1, mapsvec.shape[1]))*np.fft.fftfreq(mapsvec.shape[1], reso*np.pi/(180*60))*2*np.pi
        limits = [-max(ell.ravel()), max(ell.ravel())]
        ellx = np.ones(mapsvec[0].shape)* ell.T
        ell = np.ones((1, mapsvec.shape[2]))*np.fft.fftfreq(mapsvec.shape[2], reso*np.pi/(180*60))*2*np.pi
        elly = (np.ones(mapsvec[0].shape).T)*ell.T
        elly = elly.T
        L = np.hypot(ellx, elly)
        del ellx, elly, ell
        weights = np.ones(mapsvec.shape)
        for i in range(0,N):
            x = np.append(ells, ells[-1] + 500)
            y = np.append(psi[i,0], 0)
            fi = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')
            weight_temp = fi(L.ravel())
            weight_temp[np.where(abs(L.ravel())>max(x))] = 0
            weights[i] = weight_temp.reshape(L.shape)
            if (lowpass is not None) and (highpass is not None):
                if i == 0:
                    tfunc = transfunc(L, elllow=lowpass, ellhigh=highpass)
                weights[i] *= tfunc
        return weights, limits

def RaDec2XYZ(ra,dec):
    """
    From (ra,dec) -> unit vector on the sphere
    """
    rar  = np.radians(ra)
    decr = np.radians(dec)

    x = np.cos(rar) * np.cos(decr)
    y = np.sin(rar) * np.cos(decr)
    z = np.sin(decr)

    return np.array([x,y,z]).T

def weight_func(z_i, z_j, sigma_z):
    '''
    See Eq. 20 in (Soergel 2016)
    '''
    return np.exp(-0.5*(z_i-z_j)**2/sigma_z**2)

def richtoradvir(richness, z):
    '''
    Convert Richness to virial radii from the DES RedMapper in (McClintock 2018)
    '''
    h = cosmo.Hz(z)/100
    M200m = M0*((richness/40)**F)*(((1+z)/1.35)**G)   #M_sun
    Mnew, Rnew, Cnew = mass_adv.changeMassDefinitionCModel(M200m*h, z, '200m', 'vir', profile='nfw', c_model='diemer19')
    thetavir = Rnew*1e-3/cosmo.angularDiameterDistance(z) #Angular distance in radians
    return thetavir*3437.75/h  #Return the angular size in arcmin

def masstorich(mass, z, massdef='200c'):
    '''
    Convert mass to richness in DES according to (McClintock 2018)
    '''
    assert(len(mass) == len(z))
    Mnew = np.zeros(len(mass))
    for i in range(len(mass)):
        Mnew[i], Rnew, Cnew = mass_adv.changeMassDefinitionCModel(mass[i]*h, z[i], massdef, '200m')
    Mnew=Mnew/h
    return 40*((Mnew/M0)*(1.35/(1+z))**G)**(1./F)

def GettSZFreeMap(map150, map90, mask=None, f=1.67, beam150=1.27, beam90=1.62, lmax=30000):
    '''
    Calculate a tSZ-free map from a 150GHz and 90GHz CMB map
    '''
    assert(map150.shape == map90.shape)
    ell   = np.linspace(1,lmax+1,map150.shape[0])
    Lx = np.ones(map150.shape)*ell
    Ly = Lx.T
    L = np.hypot(Lx, Ly)
    del Lx, Ly

    # beam
    sigmareal = np.deg2rad(beam150/60)/2.355
    sigmafourier = 1 /sigmareal
    bl150_1d = gaussianbeam(ell, sigma=sigmafourier, norm=False)
    sigmareal = np.deg2rad(beam90/60)/2.355
    sigmafourier = 1 /sigmareal
    bl90_1d  = gaussianbeam(ell, sigma=sigmafourier, norm=False)
    bl150_2d = np.interp(L.ravel(), ell, bl150_1d).reshape(L.shape)
    bl90_2d  = np.interp(L.ravel(), ell, bl90_1d).reshape(L.shape)

    beam_ratio = bl90_2d/bl150_2d
    beam_ratio[np.isnan(beam_ratio)] = 0.

    ft150 = mapspectra.basicmaputils.map_to_ft(map150, apod_mask=mask)
    ft150 *= beam_ratio
    map150_tilde = mapspectra.basicmaputils.ft_to_map(ft150, apod_mask=mask)
    return  (f*map150_tilde - map90) / (f-1)

def GetNoise(halfA, halfB, lmin=2000, lmax=6000, return_cls=False):
    '''
    Calculate the noise level of map given two sub maps that contain random distributions of the full map
    '''
    diff = mapspectra.map_analysis.subtract_two_maps(halfA[0], halfB[0], divide_by_two=True)

    cls = mapspectra.map_analysis.calculate_powerspectra(diff, lmin=500, lmax=10000, apod_mask="from_weight")
    idx = np.where((cls["TT"].bin_centers > lmin) & (cls["TT"].bin_centers < lmax))#[0]
    tt_noise = np.sqrt(np.mean((cls["TT"])[idx])) / (core.G3Units.arcmin * core.G3Units.uK)

    print("The average noise between ells %0d and %0d is:"% (lmin, lmax))
    print("TT: %.1f uK-arcmin" % tt_noise)

    if return_cls:
        return cls

def tSZmod(nu, y=1, Tcmb=2.726, deltarc=0):
    '''
    Calculate the frequency dependence of the tSZ effect
    '''
    hplanck = 6.6263e-34
    kb = 1.3806e-23
    a = hplanck/kb
    x = a*nu*1e9/Tcmb
    ex = np.exp(x)
    frac = (ex + 1)/(ex-1)
    return y*Tcmb*((x*frac)-4)*(1+deltarc)

def masstoradvir(m, z):
    '''
    Go mass to viral radius using the colossus package
    '''
    Mnew, Rnew, Cnew = mass_adv.changeMassDefinitionCModel(m*h, z, '200c', 'vir', profile='nfw', c_model='diemer19')
    thetavir = Rnew*1e-3/cosmo.angularDiameterDistance(z) #Angular distance in radians
    return thetavir*3437.75/h  #Return the angular size in arcmin

def fn_load_halo_flender(fname, mmin=5e14, mmax=5e15, richmin=None, richmax=None, zmin=0.1 , zmax=1., nobj=None, like_SPT3g=0):
    '''
    Returns a RA, DEC, Z with redmapper catalog info from the Flender Simulation catalog
    '''
    cat = fits.open(fname)[1].data
    cat.RA[np.where(cat.RA>180)] -= 360
    cat = cat[(cat.REDSHIFT >= zmin) & (cat.REDSHIFT <= zmax)]
    if like_SPT3g:
        cat = cat[(cat.RA >= RA_SPT3g_min) & (cat.RA <= RA_SPT3g_max)]
        cat = cat[(cat.DEC >= DEC_SPT3g_min) & (cat.DEC <= DEC_SPT3g_max)]
    if nobj is not None:
        ind = range(len(cat))
        np.random.shuffle(ind)
        cat = cat[ind[0:nobj]]

    if richmin is not None and richmax is not None:
        richness = masstorich(cat.M200, cat.REDSHIFT, massdef='200c')
        cat = cat[(richness >= richmin)]
        richness = richness[(richness >= richmin)]
        cat = cat[(richness <= richmax)]
        richness = richness[(richness <= richmax)]
    else:
        cat = cat[(cat.M200 > mmin)]
        cat = cat[(cat.M200 < mmax)]
        richness = cat.M200
    ra    = cat.RA
    dec   = cat.DEC
    zs    = cat.REDSHIFT
    v_los = cat.VLOS

    return ra, dec, zs, v_los, richness

