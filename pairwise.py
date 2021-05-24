import numpy as np

from scipy.spatial import distance as dist

from astropy.io import ascii
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic as binstats
import pdb
from tqdm import tqdm
from multiprocessing import Pool
from utils import *

def readdata(filename, DECmin=None, DECmax=None, richmin=None, richmax=None, photoz=None, nobj=None, seed=None):
    clustinfo = ascii.read(filename)
    Z, TSZ, richness = np.asarray(clustinfo['Z']),np.asarray(clustinfo['TSZ']),np.asarray(clustinfo['LAMBDA'])
    RA = np.asarray(clustinfo['RA'])
    DEC = np.asarray(clustinfo['DEC'])

    if DECmin is not None and DECmax is not None:
        pos = np.concatenate((np.where(clustinfo['DEC'] > DECmax)[0] , np.where(clustinfo['DEC'] < DECmin)[0]))
        Z = np.delete(Z, pos)
        TSZ = np.delete(TSZ, pos)
        richness = np.delete(richness, pos)
        RA = np.delete(RA, pos)
        DEC = np.delete(DEC, pos)

    if richmin is not None:
        pos = np.where(richness < richmin)
        Z = np.delete(Z, pos)
        TSZ = np.delete(TSZ, pos)
        richness = np.delete(richness, pos)
        RA = np.delete(RA, pos)
        DEC = np.delete(DEC, pos)
    if richmax is not None:
        pos = np.where(richness > richmax)
        Z = np.delete(Z, pos)
        TSZ = np.delete(TSZ, pos)
        richness = np.delete(richness, pos)
        RA = np.delete(RA, pos)
        DEC = np.delete(DEC, pos)

    if nobj is not None:
        if seed is not None:
            np.random.seed(seed)
        pos = np.arange(len(Z))
        np.random.shuffle(pos)
        pos = pos[0:nobj]
        Z = Z[pos]
        RA = RA[pos]
        DEC = DEC[pos]
        TSZ = TSZ[pos]
        richness = richness[pos]

    if photoz is not None:
        if seed is not None:
            np.random.seed(seed)
        Z += np.random.normal(loc=0., scale=photoz*(1+Z))
    T = np.zeros(len(Z))
    sigmaz = 0.02
    for i in tqdm(range(len(Z))):
        zi = Z[i]
        Ti = TSZ[i]
        T[i] = Ti - sum(TSZ*weight_func(zi, Z, sigmaz))/sum(weight_func(zi, Z, sigmaz))

    com_dists = cosmo.comovingDistance(z_max=Z)/h          # Compute comoving distance to each cluster
    return RA, DEC, com_dists, T, richness


def define_bins(sep_min=41, sep_good=None, sep_max=300, nbins=20):
    if sep_good is not None:
        assert(sep_min < sep_good < sep_max)
        delta_sep = (sep_good-sep_min)/(nbins-1)
        bins = np.arange(sep_min,sep_good+delta_sep/2,delta_sep)
        bins = np.append(bins, sep_max)
    else:
        assert(sep_min < sep_max)
        delta_sep = (sep_max-sep_min)/nbins
        bins = np.arange(sep_min,sep_max+delta_sep/2,delta_sep)
        sep_max  = min(sep_max,bins[-1])
    return bins,sep_max

def CalculatePairwiseCDistJackknife(ra, dec, com_dists, field, sep_min=41, sep_good=None, sep_max=300, nbins=20, knifebins=50, flip_sign=False, count=True,bins=None):
    if bins is None:
        bins, sep_max = define_bins(sep_min=sep_min,sep_max=sep_max,nbins=nbins,sep_good=sep_good)
    else:
        nbins = len(bins)-1
    sep_max=bins[-1]
    sep_min=bins[0]
    rbins = 0.5*np.asarray(bins[1:]+bins[:-1])
    vec_unit = RaDec2XYZ(ra,dec) # Get unit vectors pointing to the cluster
    nclusts = len(ra)
    vec_dist = (vec_unit.T * com_dists).T # Mpc

    tree = cKDTree(vec_dist)
    pairs = tree.query_pairs(sep_max, output_type='ndarray')
    dista = vec_dist[pairs[:,0]]
    distb = vec_dist[pairs[:,1]]
    com_sep = np.linalg.norm(dista - distb, axis=1)
    ind = np.where(com_sep<sep_min)

    com_sep = np.delete(com_sep, ind)
    pairs = np.delete(pairs, ind, axis=0)
    unita = vec_unit[pairs[:,0]]
    unitb = vec_unit[pairs[:,1]]
    coma = com_dists[pairs[:,0]]
    comb = com_dists[pairs[:,1]]
    fielda = field[pairs[:,0]]
    fieldb = field[pairs[:,1]]
    costheta = np.sum(unita*unitb, axis=1)
    cij = (coma-comb)*(1+costheta)/(2*com_sep)

    if flip_sign:
        fij = fielda + fieldb
    else:
        fij = fielda - fieldb

    numerator = fij*cij
    denominator = (cij**2)

    bin_numerator, _, _ = binstats(com_sep, numerator, statistic='sum', bins=bins)
    bin_denominator, _, _ = binstats(com_sep, denominator, statistic='sum', bins=bins)

    if count:
        counter, _, _ = binstats(com_sep, numerator, statistic='count', bins=bins)
    else:
        counter = None

    TpkSZ = -1* bin_numerator/bin_denominator

    TpkSZknife = np.zeros((knifebins,len(bins)-1))
    keep = np.zeros(len(ra),dtype=bool) # clusters keep or not
    # now do jackknife resampling:
    #randomly reorder to avoid any location dependence on indices
    randid = np.arange(nclusts)
    np.random.shuffle(randid)
    for i in range(knifebins):
        keep[randid[i::knifebins]]=True
        keep_pairs = np.logical_or(keep[pairs[:,0]],keep[pairs[:,1]])
        numerator_to_drop, _, _ = binstats(com_sep[keep_pairs], numerator[keep_pairs], statistic='sum', bins=bins)
        denominator_to_drop, _, _ = binstats(com_sep[keep_pairs], denominator[keep_pairs], statistic='sum', bins=bins)
        TpkSZknife[i,:]=-1 * (bin_numerator-numerator_to_drop)/(bin_denominator-denominator_to_drop)
        keep[:]=False

    return rbins, TpkSZ, TpkSZknife, counter


def CalculatePairwiseCDistBootstrap(ra, dec, com_dists, field, sep_min=41, sep_good=None, sep_max=300, nbins=20, nboot=25, flip_sign=False, count=True,bins=None):
    if bins is None:
        bins, sep_max = define_bins(sep_min=sep_min,sep_max=sep_max,nbins=nbins,sep_good=sep_good)
    else:
        nbins = len(bins)-1
    sep_max=bins[-1]
    sep_min=bins[0]
    rbins = 0.5*np.asarray(bins[1:]+bins[:-1])
    nclusts = len(ra)

    vec_unit = RaDec2XYZ(ra,dec) # Get unit vectors pointing to the cluster
    vec_dist = (vec_unit.T * com_dists).T # Mpc
    tree = cKDTree(vec_dist)
    pairs = tree.query_pairs(sep_max, output_type='ndarray')

    dista = vec_dist[pairs[:,0]]
    distb = vec_dist[pairs[:,1]]
    com_sep = np.linalg.norm(dista - distb, axis=1)

    sort_ind = np.argsort(com_sep)
    index_bin_edges = np.searchsorted(com_sep[sort_ind],bins,side='right')
    #the value at the index is just abov the bin edge
    #last value is upper edge

    pairs = pairs[sort_ind[index_bin_edges[0]:],:].copy()
    rev_index_bin_edges = index_bin_edges-index_bin_edges[0]

    coma  = com_dists[pairs[:,0]]
    comb  = com_dists[pairs[:,1]]
    fielda = field[pairs[:,0]]
    fieldb = field[pairs[:,1]]

    costheta = sum(vec_unit[pairs[:,0]]*vec_unit[pairs[:,1]], axis=1)
    cij = (coma-comb)*(1+costheta)/(2*com_sep[sort_ind[index_bin_edges[0]:]])

    if flip_sign:
        fij = fielda + fieldb
    else:
        fij = fielda - fieldb
    numerator = (fij*cij)
    denominator = (cij**2)

    bin_numerator= np.zeros(nbins,dtype=float)
    bin_denominator= np.zeros(nbins,dtype=float)

    for i in range(nbins):
        bin_numerator[i]=np.sum(numerator[rev_index_bin_edges[i]:rev_index_bin_edges[i+1]])
        bin_denominator[i]=np.sum(denominator[rev_index_bin_edges[i]:rev_index_bin_edges[i+1]])

    if count:
        counter = rev_index_bin_edges[1:]-rev_index_bin_edges[0:-1]
    else:
        counter = None

    TpkSZ = -1.*bin_numerator/bin_denominator

    TpkSZbootstrap= np.zeros((nboot,len(bins)-1))
    for i in range(nboot):
        #making this 1 extra to ensure weights below is always right length
        idx = np.random.randint(nclusts,size=nclusts+1)
        idx[-1]=nclusts-1 #always want idx to include last one for next function
        weights = np.bincount(idx)
        weights[-1]-=1 #taking off the artifical instance of nclust-1

        weight_pairs = weights[pairs[:,0]]*weights[pairs[:,1]]
        tmp_num = numerator*weight_pairs
        tmp_den = denominator*weight_pairs

    for j in range(nbins):
            TpkSZbootstrap[i,j]=-1 * (np.sum(tmp_num[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]])) / (np.sum(tmp_den[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]]))

    return rbins, TpkSZ, TpkSZbootstrap, counter

def BS_loop(i,nclusts,nbins,pairs,numerator,denominator,rev_index_bin_edges,rng):
    #making this 1 extra to ensure weights below is always right length
    #rng.jumped(i)
    np.random.seed(7919 + 5903*i)
    idx = np.random.randint(nclusts,size=nclusts+1) #rng.randint(nclusts,size=nclusts+1)

    idx[-1]=nclusts-1 #always want idx to include last one for next function
    weights = np.bincount(idx)
    weights[-1]-=1 #taking off the artifical instance of nclust-1

    weight_pairs = weights[pairs[:,0]]*weights[pairs[:,1]]
    tmp_num = numerator*weight_pairs
    tmp_den = denominator*weight_pairs
    BS = np.zeros(nbins)
    for j in range(nbins):
        BS[j]=-1 * (np.sum(tmp_num[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]])) / (np.sum(tmp_den[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]]))
    return BS

def CalculatePairwiseCDistBootstrapParallel(ra, dec, com_dists, field, sep_min=41, sep_good=None, sep_max=300, nbins=20, nboot=25, flip_sign=False, count=True,bins=None,cpu_pool=8):
    if bins is None:
        bins, sep_max = define_bins(sep_min=sep_min,sep_max=sep_max,nbins=nbins,sep_good=sep_good)
    else:
        nbins = len(bins)-1
    sep_max=bins[-1]
    sep_min=bins[0]
    rbins = 0.5*np.asarray(bins[1:]+bins[:-1])
    nclusts = len(ra)

    vec_unit = RaDec2XYZ(ra,dec) # Get unit vectors pointing to the cluster
    vec_dist = (vec_unit.T * com_dists).T # Mpc
    tree = cKDTree(vec_dist)
    pairs = tree.query_pairs(sep_max, output_type='ndarray')

    dista = vec_dist[pairs[:,0]]
    distb = vec_dist[pairs[:,1]]
    com_sep = np.linalg.norm(dista - distb, axis=1)

    sort_ind = np.argsort(com_sep)
    index_bin_edges = np.searchsorted(com_sep[sort_ind],bins,side='right')
    #the value at the index is just abov the bin edge
    #last value is upper edge

    pairs = pairs[sort_ind[index_bin_edges[0]:],:].copy()
    rev_index_bin_edges = index_bin_edges-index_bin_edges[0]

    coma  = com_dists[pairs[:,0]]
    comb  = com_dists[pairs[:,1]]
    fielda = field[pairs[:,0]]
    fieldb = field[pairs[:,1]]

    costheta = np.sum(vec_unit[pairs[:,0]]*vec_unit[pairs[:,1]], axis=1)

    cij = (coma-comb)*(1+costheta)/(2*com_sep[sort_ind[index_bin_edges[0]:]])

    if flip_sign:
        fij = fielda + fieldb
    else:
        fij = fielda - fieldb

    numerator = (fij*cij)
    denominator = (cij**2)
    bin_numerator= np.zeros(nbins,dtype=float)
    bin_denominator= np.zeros(nbins,dtype=float)

    for i in range(nbins):
        bin_numerator[i]=np.sum(numerator[rev_index_bin_edges[i]:rev_index_bin_edges[i+1]])
        bin_denominator[i]=np.sum(denominator[rev_index_bin_edges[i]:rev_index_bin_edges[i+1]])

    if count:
        counter = rev_index_bin_edges[1:]-rev_index_bin_edges[0:-1]
    else:
        counter = None

    TpkSZ = -1.*bin_numerator/bin_denominator

    TpkSZbootstrap= np.zeros((nboot,len(bins)-1))

    #seed = secrets.getrandbits(128)
    #rng = PCG64(seed)
    with Pool(cpu_pool) as p:
        pw_bs_tmp = p.starmap(BS_loop, [(i, nclusts, nbins,pairs,numerator,denominator,rev_index_bin_edges,0) for i in range(nboot)])

    for i in range(nboot):
        TpkSZbootstrap[i,:] = pw_bs_tmp[i]

    return rbins, TpkSZ, TpkSZbootstrap, counter

def pkSZcalc(RA, DEC, com_dists, T, bins=None, binnumber=16, sep_min=41, sep_good=None, sep_max=300, Jackknife=True, Bootstrap=False, subsamples=150, sign=False, count=False, cpu_pool=8):
    if Jackknife:
        r, TpkSZ, TpkSZknife, counter = CalculatePairwiseCDistJackknife(RA, DEC, com_dists, T, nbins=binnumber, sep_min=sep_min, sep_good=sep_good, sep_max=sep_max, knifebins=subsamples, flip_sign=sign, count=count, bins=bins)
        TpkSZcov = np.cov(TpkSZknife.T, bias=True)*(subsamples-1)
    elif Bootstrap:
        r, TpkSZ, TpkSZknife, counter = CalculatePairwiseCDistBootstrap(RA, DEC, com_dists, T, nbins=binnumber, sep_min=sep_min, sep_good=sep_good, sep_max=sep_max, nboot=subsamples, flip_sign=sign, count=count, bins=bins)
        TpkSZcov = np.cov(TpkSZknife.T)
    else:
        print('Either Jackknife or Bootstrap must be True')
        return

    TpkSZcorr = np.corrcoef(TpkSZknife.T)
    Terror = np.sqrt(np.diagonal(TpkSZcov))
    return r, TpkSZ, TpkSZcov, Terror, TpkSZcorr, counter


def pkSZstoN(r, TpkSZ, TpkSZcov, rthreshold=45, subsamples=150, returnSN = False):
    pkSZ = TpkSZ[np.where(r>rthreshold)]
    invcov = np.zeros((pkSZ.size, pkSZ.size))
    invcov = ((subsamples - len(pkSZ) - 2)/(subsamples - 1))*np.linalg.pinv(TpkSZcov[TpkSZ.size - pkSZ.size:,TpkSZ.size - pkSZ.size:])
    StoN = np.sqrt(np.matmul(pkSZ.T, np.matmul(invcov, pkSZ)))
    print(StoN)
    if returnSN:
        return StoN
    return
