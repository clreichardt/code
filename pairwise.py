from numpy import *
import numba
from scipy.spatial import distance as dist
from utils import RaDec2XYZ
from astropy.io import ascii
from tqdm import tqdm
from utils import *
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic as binstats

numba.config.NUMBA_NUM_THREADS = 8

def CalculatePairwiseCDist(ra, dec, com_dists, field, sep_min=0, sep_max=300, nbins=20, flip_sign=False, count=True):
    nclusts = len(ra)
    delta_sep = (sep_max-sep_min)/nbins
    bins = arange(sep_min,sep_max+delta_sep,delta_sep)
    sep_max  = min(sep_max,bins[-1])
    
    vec_unit = RaDec2XYZ(ra,dec) # Get unit vectors pointing to the cluster
    vec_dist = (vec_unit.T * com_dists).T # Mpc
    tree = cKDTree(vec_dist)
    pairs = tree.query_pairs(sep_max, output_type='ndarray') 
    
    unita = vec_unit[pairs[:,0]]
    unitb = vec_unit[pairs[:,1]]
    dista = vec_dist[pairs[:,0]]
    distb = vec_dist[pairs[:,1]]
    coma = com_dists[pairs[:,0]]
    comb = com_dists[pairs[:,1]]
    fielda = field[pairs[:,0]]
    fieldb = field[pairs[:,1]]
    
    com_sep = linalg.norm(dista - distb, axis=1)
    costheta = sum(unita*unitb, axis=1)
    
    cij = (coma-comb)*(1+costheta)/(2*com_sep)
    
    if flip_sign:
        fij = fielda + fieldb
    else:
        fij = fielda - fieldb
        
    numerator = fij*cij
    denominator = (cij**2)
    
    numerator, _, _ = binstats(com_sep, numerator, statistic='sum', bins=bins)
    denominator, _, _ = binstats(com_sep, denominator, statistic='sum', bins=bins)
    
    if count:
        counter, _, _ = binstats(com_sep, numerator, statistic='count', bins=bins)
    else:
        counter = None

    pairwise = numerator/denominator
    
    return bins, -1*pairwise, counter

#@numba.guvectorize([(float64[:], float64[:], float64[:], float64[:], float64, float64, int32, boolean, boolean, float64[:], float64[:], float64[:])],'(n),(n),(n),(n),(),(),(),(),()->(n),(n),(n)')
def vectorizedCalculatePairwiseCDist(ra, dec, com_dists, field, sep_min, sep_max, nbins, flip_sign, count, bins, pairwise, counter):
    nclusts = len(ra)
    delta_sep = (sep_max-sep_min)/nbins
    bins = arange(sep_min,sep_max+delta_sep,delta_sep)
    sep_max  = min(sep_max,bins[-1])
    
    vec_unit = RaDec2XYZ(ra,dec) # Get unit vectors pointing to the cluster
    vec_dist = (vec_unit.T * com_dists).T # Mpc
    tree = cKDTree(vec_dist)
    pairs = tree.query_pairs(sep_max, output_type='ndarray') 
    
    unita = vec_unit[pairs[:,0]]
    unitb = vec_unit[pairs[:,1]]
    dista = vec_dist[pairs[:,0]]
    distb = vec_dist[pairs[:,1]]
    coma = com_dists[pairs[:,0]]
    comb = com_dists[pairs[:,1]]
    fielda = field[pairs[:,0]]
    fieldb = field[pairs[:,1]]
    
    com_sep = linalg.norm(dista - distb, axis=1)
    costheta = sum(unita*unitb, axis=1)
    
    cij = (coma-comb)*(1+costheta)/(2*com_sep)
    
    if flip_sign:
        fij = fielda + fieldb
    else:
        fij = fielda - fieldb
        
    numerator = fij*cij
    denominator = (cij**2)
    
    numerator, _, _ = binstats(com_sep, numerator, statistic='sum', bins=bins)
    denominator, _, _ = binstats(com_sep, denominator, statistic='sum', bins=bins)
    
    if count:
        counter, _, _ = binstats(com_sep, numerator, statistic='count', bins=bins)
    else:
        counter = None

    pairwise = -numerator/denominator
    return
    #return bins, -1*pairwise, counter

'''
@numba.jit()   
def CalculatePairwiseCDist(ra, dec, com_dists, field, sep_min=0, sep_max=300, nbins=20, flip_sign=False):
    #assert(len(ra) == len(com_dists))
    
    nclusts = len(ra)
    delta_sep = (sep_max-sep_min)/nbins
    bins = arange(sep_min,sep_max+delta_sep,delta_sep)
    
    vec_unit = RaDec2XYZ(ra,dec) # Get unit vectors pointing to the clusters
    vec_dist = (vec_unit.T * com_dists).T # Mpc
    
    costheta = 1 - dist.cdist(vec_unit, vec_unit, 'cosine') # matrix with cos(theta_{ij}) as entries
    com_sep  = dist.cdist(vec_dist, vec_dist, 'euclidean')  # matrix with comoving separation between pairs of clusters
    sep_max  = min(sep_max,bins[-1])
    
    pairwise    = zeros(len(bins)-1)
    denominator = zeros(len(bins)-1)
    numerator   = zeros(len(bins)-1)
    counter     = zeros(len(bins)-1)
    
    if flip_sign: # NULL TEST !!!!
        for i in range(1, nclusts):
            for j in range(i):
                if (com_sep[i,j] > sep_min) and (com_sep[i,j] < sep_max):
                    # this_bin = digitize(com_sep[i,j], bins) - 1 # find the right comoving separation bin
                    this_bin = int(floor((com_sep[i,j]-sep_min)/delta_sep))
                    f_ij = field[i] + field[j] # Note the plus instead of minus
                    c_ij = (com_dists[i]-com_dists[j]) * (1+costheta[i,j]) / (2*com_sep[i,j])
                    counter[this_bin] += 1
                    numerator[this_bin]   += f_ij * c_ij
                    denominator[this_bin] += c_ij**2
    else:
        for i in range(1, nclusts):
            for j in range(i):
                if (com_sep[i,j] > sep_min) and (com_sep[i,j] < sep_max):
                    # this_bin = digitize(com_sep[i,j], bins) - 1 # find the right comoving separation bin
                    this_bin = int((com_sep[i,j]-sep_min)/delta_sep)
                    f_ij = field[i] - field[j]
                    c_ij = (com_dists[i]-com_dists[j]) * (1+costheta[i,j]) / (2*com_sep[i,j])
                    # print T_ij, c_ij
                    counter[this_bin] += 1
                    numerator[this_bin]   += f_ij * c_ij
                    denominator[this_bin] += c_ij**2
                    
    pairwise = numerator / denominator
    
    return bins, -1*pairwise, counter
'''

@numba.jit()
def GetBootstrap(ra, dec, com_dists, field, sep_min=0, sep_max=300, nbins=20, nboot=25, flip_sign=False):
    nclusts = len(ra)
    delta_sep = (sep_max-sep_min)/nbins
    bins = arange(sep_min,sep_max+delta_sep,delta_sep)

    pw_bs = zeros((nboot,len(bins)-1))

    # Bootstrap estimation of covariance matrix
    #print("...calculate covariance through bootstrap...")
    for i in range(nboot):
        idx = random.randint(0, high=nclusts, size=nclusts) #random.choice(len(ra), size=len(ra))
        ra_tmp = ra[idx]
        dec_tmp = dec[idx]
        com_dists_tmp = com_dists[idx]
        field_tmp = field[idx]
        _, pw_bs[i], _ = CalculatePairwiseCDist(ra_tmp, dec_tmp, com_dists_tmp, field_tmp, sep_min=sep_min, sep_max=sep_max, nbins=nbins, flip_sign=flip_sign, count=False)
    #print("...Done...")
    return pw_bs

@numba.jit()
def GetJackknife(ra, dec, com_dists, field, sep_min=0, sep_max=300, nbins=20, knifebins=50, flip_sign=False):
    nclusts = len(ra)
    delta_sep = (sep_max - sep_min)/nbins
    bins = arange(sep_min, sep_max + delta_sep, delta_sep)
    
    cuts = int(nclusts/knifebins)
    pw_bs = zeros((knifebins,len(bins)-1))
    
    # Bootstrap estimation of covariance matrix
    #print("...calculate covariance through jackknife...")
    c = 0
    for i in range(knifebins):
        byebye = range(c, cuts + c)
        ra_tmp = delete(ra, byebye)
        dec_tmp = delete(dec, byebye)
        com_dists_tmp = delete(com_dists, byebye)
        field_tmp = delete(field, byebye)
        _, pw_bs[i], _ = CalculatePairwiseCDist(ra_tmp, dec_tmp, com_dists_tmp, field_tmp, sep_min=sep_min, sep_max=sep_max, nbins=nbins, flip_sign=flip_sign, count = False)
        c += cuts
    #print("...Done...")
    return pw_bs

def readdata(filename, DECmin=None, DECmax=None, richmin=None, richmax=None, photoz=None, nobj=None, seed=None):
    clustinfo = ascii.read(filename)
    Z, TSZ, richness = asarray(clustinfo['Z']),asarray(clustinfo['TSZ']),asarray(clustinfo['LAMBDA'])
    RA = asarray(clustinfo['RA'])
    DEC = asarray(clustinfo['DEC'])

    if DECmin is not None and DECmax is not None:    
        pos = concatenate((where(clustinfo['DEC'] > DECmax)[0] , where(clustinfo['DEC'] < DECmin)[0]))      
        Z = delete(Z, pos)
        TSZ = delete(TSZ, pos)
        richness = delete(richness, pos)
        RA = delete(RA, pos)
        DEC = delete(DEC, pos)

    if richmin is not None:
        pos = where(richness < richmin)
        Z = delete(Z, pos)
        TSZ = delete(TSZ, pos)
        richness = delete(richness, pos)
        RA = delete(RA, pos)
        DEC = delete(DEC, pos)        
    
    if richmax is not None:
        pos = where(richness > richmax)
        Z = delete(Z, pos)
        TSZ = delete(TSZ, pos)
        richness = delete(richness, pos)
        RA = delete(RA, pos)
        DEC = delete(DEC, pos)        
        
        
    if nobj is not None:
        if seed is not None:
            random.seed(seed)
            
        pos = random.randint(0,len(Z),size=nobj)
        Z = Z[pos]
        RA = RA[pos]
        DEC = DEC[pos]
        TSZ = TSZ[pos]
        richness = richness[pos]
        
    if photoz is not None:
        if seed is not None:
            random.seed(seed)
            
        Z += np.random.normal(loc=0., scale=photoz*(1+Z))
        
    T = zeros(len(Z))
    sigmaz = 0.02
    
    for i in tqdm(range(len(Z))):
        zi = Z[i]
        Ti = TSZ[i]
        T[i] = Ti - sum(TSZ*weight_func(zi, Z, sigmaz))/sum(weight_func(zi, Z, sigmaz))

    h = cosmo.Hz(0)/100
    com_dists = cosmo.comovingDistance(z_max=Z)/h          # Compute comoving distance to each cluster
        
    return RA, DEC, com_dists, T, richness

def pkSZcalc(RA, DEC, com_dists, T, binnumber=16, Jackknife=True, Bootstrap=False, subsamples=150, sign=False, count=False):
    r, TpkSZ, counter = CalculatePairwiseCDist(RA, DEC, com_dists, T, nbins=binnumber, flip_sign=sign, count=count)
    r = delete(r, 0)
    if Jackknife:
        TpkSZknife = GetJackknife(RA, DEC, com_dists, T, nbins=binnumber, knifebins=subsamples, flip_sign=sign)
    elif Bootstrap:
        TpkSZknife = GetBootstrap(RA, DEC, com_dists, T, nbins=binnumber, nboot=subsamples, flip_sign=sign)
    else:
        print('Either Jackknife or Bootstrap must be True')
        return
    TpkSZcov = cov(TpkSZknife.T, bias=True)*(subsamples-1)
    TpkSZcorr = corrcoef(TpkSZknife.T)
    Terror = sqrt(diagonal(TpkSZcov))
    return r, TpkSZ, TpkSZcov, Terror, TpkSZcorr, counter

def pkSZstoN(r, TpkSZ, TpkSZcov, rthreshold=45, subsamples=150, returnSN = False):
    pkSZ = TpkSZ[where(r>rthreshold)]
    invcov = zeros((pkSZ.size, pkSZ.size))
    invcov = ((subsamples - len(pkSZ) - 2)/(subsamples - 1))*linalg.inv(TpkSZcov[TpkSZ.size - pkSZ.size:,TpkSZ.size - pkSZ.size:])
    StoN = sqrt(matmul(pkSZ.T, matmul(invcov, pkSZ)))
    print(StoN)
    if returnSN:
        return StoN
    else:
        return 