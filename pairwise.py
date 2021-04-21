import numpy as np
from astropy.io import ascii
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic as binstats
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

    h = cosmo.Hz(0)/100
    com_dists = cosmo.comovingDistance(z_max=Z)/h          # Compute comoving distance to each cluster
        
    return RA, DEC, com_dists, T, richness


def CalculatePairwiseCDistBootstrap(ra, dec, com_dists, field, sep_min=41, sep_good=None, sep_max=300, nbins=20, nboot=25, flip_sign=False, count=True,bins=None, cpu_pool=None):
    nclusts = len(ra)
    if bins is None:
        if sep_good is not None:
            assert(sep_min < sep_good < sep_max)
            delta_sep = (sep_good-sep_min)/(nbins-1)
            bins = np.arange(sep_min,sep_good+delta_sep/2,delta_sep)
            bins = np.append(bins, sep_max)
        else:
            delta_sep = (sep_max-sep_min)/nbins
            bins = np.arange(sep_min,sep_max+delta_sep/2,delta_sep)
            sep_max  = min(sep_max,bins[-1])
    else:
        nbins=len(bins)-1
        sep_max=bins[-1]
        sep_min=bins[0]
    
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
    
    rbins = zeros(nbins)
    for i in range(0,nbins):
        rbins[i] = (bins[i+1] + bins[i])/2

    TpkSZbootstrap= np.zeros((nboot,len(bins)-1))
    # now do jackknife resampling:
    #randomly reorder to avoid any location dependence on indices
    pw_bs = np.zeros((nboot,len(bins)-1))
    #randid = np.arange(nclusts)
    #np.random.shuffle(randid)
    cluster_count = np.zeros(len(ra),dtype=int)
    
    if cpu_pool is not None:
        with Pool(cpu_pool) as p:
            pw_bs_tmp = p.starmap(Bootstrap, [(i, nbins, nclusts, pairs, numerator, denominator, rev_index_bin_edges) for i in range(nboot)])
    else:
        with Pool() as p:
            pw_bs_tmp = p.starmap(Bootstrap, [(i, nbins, nclusts, pairs, numerator, denominator, rev_index_bin_edges) for i in range(nboot)])
        
    for i in range(nboot):
        TpkSZbootstrap[i,:] = pw_bs_tmp[i]
    
    
    return rbins, TpkSZ, TpkSZbootstrap, counter 

def Bootstrap(i, nbins, nclusts, pairs, numerator, denominator, rev_index_bin_edges):
    #making this 1 extra to ensure weights below is always right length
    TpkSZbootstrap= np.zeros(nbins)
    
    idx = np.random.randint(nclusts,size=nclusts+1)
    idx[-1]=nclusts-1 #always want idx to include last one for next function
    weights = np.bincount(idx)
    weights[-1]-=1 #taking off the artifical instance of nclust-1

    weight_pairs = weights[pairs[:,0]]*weights[pairs[:,1]]
    tmp_num = numerator*weight_pairs
    tmp_den = denominator*weight_pairs

    for j in range(nbins):
    #numerator_bs, _, _ = binstats(com_sep, numerator*weight_pairs, statistic='sum', bins=bins)
    #denominator_bs, _, _ = binstats(com_sep, denominator*weight_pairs, statistic='sum', bins=bins)
        #TpkSZbootstrap[i,j]=-1 * (np.sum(tmp_num[bin_inds == (j+1)]))/(np.sum(tmp_den[bin_inds == (j+1)]))
        TpkSZbootstrap[j]=-1 * (np.sum(tmp_num[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]])) / (np.sum(tmp_den[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]]))
        #below was x2 slower
        #TpkSZbootstrap[i,j]=-1 * (np.sum(tmp_num*(bin_inds == (j+1))))/(np.sum(tmp_den*(bin_inds == (j+1))))
    return TpkSZbootstrap


def CalculatePairwiseCDistJackknife(ra, dec, com_dists, field, sep_min=41, sep_good=None, sep_max=300, nbins=20, knifebins=50, flip_sign=False, count=True,bins=None, cpu_pool=None):
    nclusts = len(ra)
    if bins is None:
        if sep_good is not None:
            assert(sep_min < sep_good < sep_max)
            delta_sep = (sep_good-sep_min)/(nbins-1)
            bins = np.arange(sep_min,sep_good+delta_sep/2,delta_sep)
            bins = np.append(bins, sep_max)
        else:
            delta_sep = (sep_max-sep_min)/nbins
            bins = np.arange(sep_min,sep_max+delta_sep/2,delta_sep)
            sep_max  = min(sep_max,bins[-1])
    else:
        nbins = len(bins)-1
        sep_max=bins[-1]
        sep_min=bins[0]
    
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
    
    bin_numerator, _, _ = binstats(com_sep[sort_ind[index_bin_edges[0]:]], numerator, statistic='sum', bins=bins)
    bin_denominator, _, _ = binstats(com_sep[sort_ind[index_bin_edges[0]:]], denominator, statistic='sum', bins=bins)
    
    if count:
        counter = rev_index_bin_edges[1:]-rev_index_bin_edges[0:-1]
    else:
        counter = None

    TpkSZ = -1.*bin_numerator/bin_denominator

    rbins = zeros(nbins)
    for i in range(0,nbins):
        rbins[i] = (bins[i+1] + bins[i])/2

    TpkSZknife = np.zeros((knifebins,len(bins)-1))
    # now do jackknife resampling:
    #randomly reorder to avoid any location dependence on indices
    cuts = int(nclusts/knifebins)
    pw_bs = np.zeros((knifebins,len(bins)-1))
    randid = np.arange(nclusts)
    
    np.random.shuffle(randid)
    keep = np.zeros(len(ra),dtype=bool)
    
    
    numerator_to_drop=np.zeros(nbins,dtype=float)
    denominator_to_drop=np.zeros(nbins,dtype=float)
    for i in range(knifebins):
        keep[randid[i::knifebins]]=True
        keep_pairs = np.logical_or(keep[pairs[:,0]],keep[pairs[:,1]])
        for j in range(nbins):
            numerator_to_drop[j]=np.sum((numerator[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]])[keep_pairs[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]]])
            denominator_to_drop[j]=np.sum((denominator[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]])[keep_pairs[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]]])

        TpkSZknife[i,:]=-1 * (bin_numerator-numerator_to_drop)/(bin_denominator-denominator_to_drop)
        keep[randid[i::knifebins]]=False
    '''
    if cpu_pool is not None:
        with Pool(cpu_pool) as p:
            pw_bs_tmp = p.starmap(Jackknife, [(i, nbins, cuts, knifebins, randid, keep, pairs, numerator, denominator, rev_index_bin_edges, bin_numerator, bin_denominator) for i in range(knifebins)])
    else:
        with Pool() as p:
            pw_bs_tmp = p.starmap(Jackknife, [(i, nbins, cuts, knifebins, randid, keep, pairs, numerator, denominator, rev_index_bin_edges, bin_numerator, bin_denominator) for i in range(knifebins)])
        
    for i in range(knifebins):
        TpkSZknife[i,:] = pw_bs_tmp[i]
    '''
    
    return rbins, TpkSZ, TpkSZknife, counter         
    
def Jackknife(i, nbins, cuts, knifebins, randid, keep, pairs, numerator, denominator, rev_index_bin_edges, bin_numerator, bin_denominator):
    keep[randid[i::knifebins]]=True
    keep_pairs = np.logical_or(keep[pairs[:,0]],keep[pairs[:,1]])
    
    numerator_to_drop=np.zeros(nbins,dtype=float)
    denominator_to_drop=np.zeros(nbins,dtype=float)
    for j in range(nbins):
        numerator_to_drop[j]=np.sum((numerator[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]])[keep_pairs[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]]])
        denominator_to_drop[j]=np.sum((denominator[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]])[keep_pairs[rev_index_bin_edges[j]:rev_index_bin_edges[j+1]]])

    TpkSZknife=-1 * (bin_numerator-numerator_to_drop)/(bin_denominator-denominator_to_drop)
    keep[randid[i::knifebins]]=False
    return TpkSZknife
    
def pkSZcalc(RA, DEC, com_dists, T, bins=None, binnumber=16, sep_min=41, sep_good=None, sep_max=300, Jackknife=True, Bootstrap=False, subsamples=150, sign=False, count=False, cpu_pool=8):
    
    if Jackknife:
        r, TpkSZ, TpkSZknife, counter = CalculatePairwiseCDistJackknife(RA, DEC, com_dists, T, nbins=binnumber, sep_min=sep_min, sep_good=sep_good, sep_max=sep_max, knifebins=subsamples, flip_sign=sign, count=count, bins=None, cpu_pool=cpu_pool)
        TpkSZcov = np.cov(TpkSZknife.T, bias=True)*(subsamples-1)
    elif Bootstrap:
        r, TpkSZ, TpkSZknife, counter = CalculatePairwiseCDistBootstrap(RA, DEC, com_dists, T, nbins=binnumber, sep_min=sep_min, sep_good=sep_good, sep_max=sep_max, nboot=subsamples, flip_sign=sign, count=count, bins=None, cpu_pool=cpu_pool)
        TpkSZcov = np.cov(TpkSZknife.T)
    else:
        print('Either Jackknife or Bootstrap must be True')
        return
    
    TpkSZcorr = np.corrcoef(TpkSZknife.T)
    Terror = np.sqrt(np.diagonal(TpkSZcov))
    return r, TpkSZ, TpkSZcov, Terror, TpkSZcorr, counter


def pkSZstoN(r, TpkSZ, TpkSZcov, rthreshold=40, subsamples=150, returnSN = False):
    pkSZ = TpkSZ[np.where(r>rthreshold)]
    invcov = np.zeros((pkSZ.size, pkSZ.size))
    invcov = ((subsamples - len(pkSZ) - 2)/(subsamples - 1))*np.linalg.pinv(TpkSZcov[TpkSZ.size - pkSZ.size:,TpkSZ.size - pkSZ.size:])
    StoN = np.sqrt(np.matmul(pkSZ.T, matmul(invcov, pkSZ)))
    print(StoN)
    if returnSN:
        return StoN
    else:
        return     
