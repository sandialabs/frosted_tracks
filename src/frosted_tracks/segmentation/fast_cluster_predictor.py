# -*- coding: utf-8 -*-

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Estimate the number of clusters for TICC

This is an experimental routine that tries to formulate a good guess
for the number of clusters (labels) you should ask TICC to compute.

Written by Keith Dalbey.
"""

from fast_ticc import data_preparation
from typing import List, Tuple
from scipy.stats import chi2
import numpy as np
import numpy.typing as npt
import numba
import numba_scipy
from scipy.special import gammainc, gammaincc, gammaincinv
from collections import Counter, OrderedDict
from pprint import pprint
from time import sleep
from copy import deepcopy


def stack_training_data_multiple_series(
        all_series: List[npt.NDArray[np.float64]],
        window_size: int) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.int64]]:
    """Stack data points from multiple input series into time windows.

    NOTE: This function is not quite the same as the version in
          data_preparation.  There are some changes to minimize the
          effects of boundaries between data series.

    This method takes a list of data series with the same sensors (variables
    per point) but potentially a different number of points and stacks
    them into the windows used by TICC.

    Note that for implementation reasons, the last window_size - 1
    points are trimmed from each input data series.

    If you call this function you will probably also want to use
    label_switching_cost_template() to build your label switching
    cost array.

    Arguments:
        data (NumPy array): Input data: one data point per row, one
            data series per column.
        window_size (int): How many data points to put in each window

    Returns:
        Single stacked data array with input points from all samples
    """

     # Stack all the input data sets separately
    stacked_data = [
        data_preparation.stack_training_data(data, window_size)
        for data in all_series
    ]

    # Get the length of each one.  The stacking process trims the last
    # (window_size - 1) points off each input series.
    stacked_data_sizes = [sd.shape[0] for sd in stacked_data]
    i_dont_gobble_across = np.insert(np.cumsum(stacked_data_sizes),0,0)
    # Concatenate the individual stacks into our one big brick.
    combined_data_series = np.vstack(stacked_data)
    return (combined_data_series,i_dont_gobble_across)


#this is what we needed numba_scipy for... it extends numba to make it aware
#of some some scipy features
@numba.njit(parallel=False,nopython=True)
def chi2Cdf(dof : np.float64 , x : np.float64) -> np.float64:
    x=np.maximum(x,np.float64(0.0))
    dof=np.maximum(dof,np.float64(0.0))
    if x==0.0:
        return np.float64(0.0)
    if not (x<np.inf):
        return 1.0
    return np.float64(gammainc(dof*0.5,x*0.5))


@numba.njit(parallel=False,nopython=True)
def chi2Ccdf(dof : np.float64 , x : np.float64) -> np.float64:
    x=np.maximum(x,np.float64(0.0))
    dof=np.maximum(dof,np.float64(0.0))
    if x==0.0:
        return np.float64(1.0)
    if not (x<np.inf):
        return 0.0
    return np.float64(gammaincc(dof*0.5,x*0.5))


@numba.njit(parallel=False,nopython=True)
def chi2InvCdf(dof : np.float64, cdfValue : np.float64) -> np.float64:
    cdfValue=np.maximum(np.float64(0.0),np.minimum(cdfValue,np.float64(1.0)))
    dof=np.maximum(dof,np.float64(0.0))
    if cdfValue == 0.0:
        return np.float64(0.0)
    elif cdfValue == 1.0:
        return np.float64(np.inf)
    return np.float64(2.0*gammaincinv(dof*0.5,cdfValue))


@numba.njit(parallel=False,nopython=True)
def covMatDegreesOfFreedom(n_features : int) -> int:
    dof = int(n_features*(n_features+1)/2)
    return dof


@numba.njit(parallel=False,nopython=True)
def min_window_size(n_features : int) -> int:
    #the cov mat has n_features*(n_features+1)/2 degrees of freedom
    #but we collect n_features/pieces-of-information per time step
    #and we need 1 additional piece of information per feature to determine
    #the per window local per feature means
    ws = np.ceil(0.5*(n_features+1))+1
    return ws


@numba.njit(parallel=False,nopython=True)
def pooledCovariance(\
        CovA : npt.NDArray[np.float64], window_A_info_size : np.float64,\
        CovB : npt.NDArray[np.float64], window_B_info_size : np.float64)->\
    npt.NDArray[np.float64]:
    #basically this https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Two-sample_statistic but adjusted because our mean is a linear fit rather than a constant
    return (window_A_info_size*CovA+window_B_info_size*CovB) \
        /(window_A_info_size+window_B_info_size)


@numba.njit(parallel=False)
def calc_max_cluster_compare_dof(n_features : np.int64,\
                                 window_size : np.int64)\
    ->np.float64:
    #a large number of samples can render human insignifcant differences
    #"statistical significant," what we want is to detect differences that
    #are "human useful"/"practically significant." the idea is to cause it to
    #merge clusters with covariance matrices about which a human would say
    #"they're the same within noise."  I decided to use the
    #max_cluster_compare_dof threshold because it is also used to determine the
    #magnitude of the penalty for switching clusters between sequential windows
    #I chose max_cluster_compare_dof for use in the penalty because it is the
    #degrees of freedom in the inverse covariance matrices that TICC compares
    return np.float64((n_features*window_size)*(n_features*window_size+1))*0.5


#don't know if scipy.stats.chi2 allows for jit with nopython=true
@numba.njit(parallel=False)
def BoxMTestRightTailedPValue(\
        CovA : npt.NDArray[np.float64], window_A_info_size: np.float64,\
        CovB : npt.NDArray[np.float64], window_B_info_size: np.float64,
        max_cluster_compare_dof : np.float64 = np.inf)\
    -> Tuple[np.float64,np.float64,np.float64]:
    nA = np.minimum(window_A_info_size,max_cluster_compare_dof)
    nB = np.minimum(window_B_info_size,max_cluster_compare_dof)

    #the return value (right tailed pvalue) "hand waves" like the probability that the two covariance matrices are the same, under the assumption that the covariance matrices are for normal distributions
    #I basically did this https://stats.stackexchange.com/questions/16557/how-to-test-whether-a-covariance-matrix-has-changed-over-two-time-points
    #converted M to a chi2 stat and converted the chi2 stat into a right tailed p-value
    assert(len(CovA.shape)==2)
    assert(len(CovB.shape)==2)
    p=CovA.shape[0]
    assert(p==CovA.shape[1])
    assert(p==CovB.shape[0])
    assert(p==CovB.shape[1])
    chi2Dof=np.float64(covMatDegreesOfFreedom(p))
    CovC=pooledCovariance(CovA,nA,CovB,nB)
    #the determininants of covariance matrices should be non-negative , may have to protect against zero and/or numerical error
    #The abs is likely better/more accurate protection regarding numerical round off error causing a negative number than max(0.0)
    temp=np.abs(np.linalg.eigvalsh(CovA).flatten())
    yada=temp.max()
    if yada==0:
        yada=1.0
    temp=np.maximum(temp,yada/np.float64(2**26))
    logDetA=np.sum(np.log(temp))
    temp=np.abs(np.linalg.eigvalsh(CovB).flatten())
    yada=temp.max()
    if yada==0:
        yada=1.0
    temp=np.maximum(temp,yada/np.float64(2**26))
    logDetB=np.sum(np.log(temp))
    temp=np.abs(np.linalg.eigvalsh(CovC).flatten())
    yada=temp.max()
    if yada==0:
        yada=1.0
    temp=np.maximum(temp,yada/np.float64(2**26))
    logDetC=np.sum(np.log(temp))
    logM=0.5*(logDetA*nA+logDetB*nB-logDetC*(nA+nB))
    c1= (1.0/nA+1.0/nB-1.0/(nA+nB))*(2.0*p*p+3.0*p-1.0)/(6.0*(p+1.0))
    assert(0.0<c1)
    origChi2Stat=np.float64(np.maximum(np.float64(0.0),-2.0*(1.0-c1)*logM))
    right_tailed_pvalue = np.float64(chi2Ccdf(chi2Dof,origChi2Stat)) #this should be more accurate than 1-chi2Cdf() in the cases where it matters
    return (right_tailed_pvalue,origChi2Stat,chi2Dof)


@numba.njit(parallel=False,nopython=False)
def compute_least_squares_factor(window_size : np.int64)\
    -> np.float64:
    #least_squares_factor=np.float64(1.0-2.0/np.float64(window_size)) #local linear trend func
    least_squares_factor=np.float64(1.0-1.0/np.float64(window_size)) #local constant trend func
    assert(np.float64(0.0) < least_squares_factor and least_squares_factor<np.float64(1.0))
    return least_squares_factor


@numba.njit(parallel=False)
def gen_log_det_and_cov_mat_for_window(n_features : np.int64, window_size : np.int64,\
                            features_for_window : npt.NDArray[np.float64])\
    -> Tuple[np.float64,npt.NDArray[np.float64]]:
    assert(features_for_window.size==n_features*window_size)
    yada = np.zeros(shape=(1,n_features),dtype=np.float64)
    for i_sample in range(0,window_size):
        yeda = features_for_window[np.arange(i_sample*n_features,(i_sample+1)*n_features)]
        yada+= yeda
    yada/=np.float64(window_size)
    C = np.zeros(shape=(n_features,n_features),dtype=np.float64)
    for i_sample in range(0,window_size):
        yeda = features_for_window[np.arange(i_sample*n_features,(i_sample+1)*n_features)]-yada
        C = C + np.outer(yeda,yeda)
    C = C / window_size

    (eigvals,eigvect)=np.linalg.eigh(C)
    mineig =2.0**-np.floor(52/n_features)
    eigvals=np.maximum(np.abs(eigvals.flatten()),mineig)
    C=np.dot(eigvect,np.dot(np.diag(eigvals),eigvect.T))
    logDetC=np.sum(np.log(eigvals))
    return (logDetC,C)


#you can't numba this because it replaces None with the NDArray[np.float64]
def instantiate_list_of_cov_mats(\
        n_in_list: np.int64,\
        n_mat_edge: np.int64)\
    ->List[npt.NDArray[np.float64]]:
    list_of_cov_mats=[None]*n_in_list
    for i in range(0,n_in_list):
        list_of_cov_mats[i]=np.zeros(shape=(n_mat_edge,n_mat_edge),\
                                     dtype=np.float64)
    return list_of_cov_mats


@numba.njit(parallel=False,nopython=False)
def gen_log_det_and_cov_mat_for_all_windows(\
        stacked_training_data : npt.NDArray[np.float64],\
        n_features: np.int64,\
        window_size : np.int64,\
        n_windows : np.int64,\
        window_cov_mats : List[npt.NDArray[np.float64]])\
    ->Tuple[npt.NDArray[np.float64],List[npt.NDArray[np.float64]]]:
    log_det_of_window_cov_mats=np.zeros(n_windows,dtype=np.float64)
    for i in range(0,n_windows):
        (ldC,C)=gen_log_det_and_cov_mat_for_window(n_features, window_size,\
                                                   stacked_training_data[i])
        log_det_of_window_cov_mats[i]=ldC
        window_cov_mats[i]=C
    return (log_det_of_window_cov_mats,window_cov_mats)


@numba.njit(parallel=False,nopython=False)
def gen_initial_clusters(\
        n_windows : np.int64,\
        window_info_size : np.float64,\
        max_cluster_compare_dof : np.float64,\
        delta_log : np.float64,\
        mineig : np.float64,\
        threshold : np.float64,\
        log_det_of_window_cov_mats : npt.NDArray[np.float64],\
        window_cov_mats : List[npt.NDArray[np.float64]])\
    ->Tuple[List[np.float64],List[np.float64],\
            List[npt.NDArray[np.float64]],List[npt.NDArray[np.float64]]]:

    i_sort=np.argsort(log_det_of_window_cov_mats)
    min_log_det_cov_mat=log_det_of_window_cov_mats[i_sort[0]]
    max_log_det_cov_mat=log_det_of_window_cov_mats[i_sort[n_windows-1]]
    print("log_det_cov_mat: min:",min_log_det_cov_mat,", max:",max_log_det_cov_mat)

    per_cluster_log_det_cov_mat=[log_det_of_window_cov_mats[i_sort[0]]+0.0]
    per_cluster_sum_cov_mat=[np.copy(window_cov_mats[i_sort[0]])]
    per_cluster_mean_cov_mat=[np.copy(window_cov_mats[i_sort[0]])]
    n_windows_per_cluster=[1.0]
    n_clusters=np.int64(1)

    for i in range(1,n_windows):
        this=log_det_of_window_cov_mats[i_sort[i]]
        lower=this-delta_log
        upper=this+delta_log
        if_not_in_existing_cluster=True
        for i_cluster in range(0,n_clusters):
            if (lower <= per_cluster_log_det_cov_mat[i_cluster]) and\
                (per_cluster_log_det_cov_mat[i_cluster]<upper):
                yada = np.minimum(n_windows_per_cluster[i_cluster]*window_info_size,max_cluster_compare_dof)
                denom = yada +window_info_size
                C = 0.5*(window_cov_mats[i_sort[i]]*window_info_size+per_cluster_mean_cov_mat[i_cluster]*yada)/denom
                logDetC=np.sum(np.log(np.maximum(np.abs(np.linalg.eigvalsh(C).flatten()),mineig)))
                crit = (logDetC*denom-this*window_info_size-per_cluster_log_det_cov_mat[i_cluster]*np.minimum(n_windows_per_cluster[i_cluster]*window_info_size,max_cluster_compare_dof))/max_cluster_compare_dof
                if crit <= threshold:
                    per_cluster_sum_cov_mat[i_cluster]+=window_cov_mats[i_sort[i]]
                    n_windows_per_cluster[i_cluster]+=1.0
                    per_cluster_mean_cov_mat[i_cluster] =\
                        per_cluster_sum_cov_mat[i_cluster]/n_windows_per_cluster[i_cluster]
                    per_cluster_log_det_cov_mat[i_cluster]=\
                        np.sum(np.log(np.maximum(np.abs(np.linalg.eigvalsh(per_cluster_mean_cov_mat[i_cluster]).flatten()),mineig)))
                    if_not_in_existing_cluster = False
                    break
        if if_not_in_existing_cluster:
            per_cluster_log_det_cov_mat.append(this+0.0)
            per_cluster_sum_cov_mat.append(np.copy(window_cov_mats[i_sort[i]]))
            per_cluster_mean_cov_mat.append(np.copy(window_cov_mats[i_sort[i]]))
            n_windows_per_cluster.append(1.0)
            n_clusters+=1

    assert(n_clusters==len(n_windows_per_cluster))
    assert(n_clusters==len(per_cluster_log_det_cov_mat))
    assert(n_clusters==len(per_cluster_mean_cov_mat))
    assert(n_clusters==len(per_cluster_sum_cov_mat))
    return (n_windows_per_cluster,per_cluster_log_det_cov_mat,\
            per_cluster_mean_cov_mat,per_cluster_sum_cov_mat)


@numba.njit(parallel=False,nopython=False)
def merge_clusters_that_are_similar_enough(\
       delta_log : np.float64,\
       mineig : np.float64,\
       threshold : np.float64,\
       window_info_size : np.float64,\
       max_cluster_compare_dof : np.float64,\
       n_windows_per_cluster : List[np.float64],\
       per_cluster_log_det_cov_mat : List[np.float64],\
       per_cluster_mean_cov_mat : List[npt.NDArray[np.float64]],\
       per_cluster_sum_cov_mat : List[npt.NDArray[np.float64]])\
    ->Tuple[List[np.float64],List[np.float64],List[npt.NDArray[np.float64]],\
            List[npt.NDArray[np.float64]]]:
    n_clusters = len(per_cluster_log_det_cov_mat)
    assert(n_clusters == len(per_cluster_mean_cov_mat))
    n_clusters_prev = n_clusters+1
    while n_clusters < n_clusters_prev:
        n_clusters_prev = n_clusters +0
        i_cluster =1
        while i_cluster<n_clusters:
            this = per_cluster_log_det_cov_mat[i_cluster]
            lower = this - delta_log
            upper = this + delta_log
            if_merged_with_another_cluster = False
            yada=np.minimum(n_windows_per_cluster[i_cluster]*window_info_size,max_cluster_compare_dof)

            for j_cluster in range(0,i_cluster):
                if (lower<=per_cluster_log_det_cov_mat[j_cluster]) and\
                    (per_cluster_log_det_cov_mat[j_cluster]<=upper):
                    yeda=np.minimum(n_windows_per_cluster[j_cluster]*window_info_size,max_cluster_compare_dof)
                    yida=yada+yeda
                    C = 0.5*(per_cluster_mean_cov_mat[i_cluster]*yada+\
                             per_cluster_mean_cov_mat[j_cluster]*yeda)/yida
                    logDetC=np.sum(np.log(np.maximum(np.abs(np.linalg.eigvalsh(C).flatten()),mineig)))
                    crit = (logDetC*yida-this*yada-per_cluster_log_det_cov_mat[i_cluster]*yeda)/max_cluster_compare_dof
                    if crit <= threshold:
                        per_cluster_sum_cov_mat[j_cluster]+=per_cluster_sum_cov_mat[i_cluster]
                        n_windows_per_cluster[j_cluster]+=n_windows_per_cluster[i_cluster]
                        per_cluster_mean_cov_mat[j_cluster]=per_cluster_sum_cov_mat[j_cluster]/n_windows_per_cluster[j_cluster]
                        per_cluster_log_det_cov_mat[j_cluster]=\
                            np.sum(np.log(np.maximum(np.abs(np.linalg.eigvalsh(per_cluster_mean_cov_mat[j_cluster]).flatten()),mineig)))
                        per_cluster_sum_cov_mat=per_cluster_sum_cov_mat[0:i_cluster]+\
                            per_cluster_sum_cov_mat[i_cluster+1:]
                        per_cluster_mean_cov_mat=per_cluster_mean_cov_mat[0:i_cluster]+\
                            per_cluster_mean_cov_mat[i_cluster+1:]
                        n_windows_per_cluster=n_windows_per_cluster[0:i_cluster]+\
                            n_windows_per_cluster[i_cluster+1:]
                        per_cluster_log_det_cov_mat=per_cluster_log_det_cov_mat[0:i_cluster]+\
                            per_cluster_log_det_cov_mat[i_cluster+1:]
                        n_clusters-=1
                        assert(len(per_cluster_sum_cov_mat)==n_clusters)
                        assert(len(per_cluster_mean_cov_mat)==n_clusters)
                        assert(len(per_cluster_log_det_cov_mat)==n_clusters)
                        assert(len(n_windows_per_cluster)==n_clusters)
                        if_merged_with_another_cluster = True
                        break
            if if_merged_with_another_cluster == False:
                i_cluster+=1
    assert(n_clusters==len(n_windows_per_cluster))
    assert(n_clusters==len(per_cluster_log_det_cov_mat))
    assert(n_clusters==len(per_cluster_mean_cov_mat))
    assert(n_clusters==len(per_cluster_mean_cov_mat))
    return (n_windows_per_cluster,per_cluster_log_det_cov_mat,\
       per_cluster_mean_cov_mat,per_cluster_sum_cov_mat)


@numba.njit(parallel=False,nopython=False)
def cluster_switch_penalty_coef(\
        n_features : np.int64,\
        window_size : np.int64,\
        penalty_p_upper_limit : np.float64,\
        p_same : np.float64)\
    -> np.float64:
    a=np.float64(2.0-1.0/penalty_p_upper_limit)
    dof = np.float64(n_features*window_size*(n_features*window_size+1)*0.5) #if the penalty isn't large enough, then remove the *0.5
    return 0.5*np.float64(chi2InvCdf(dof,np.float64(1.0+penalty_p_upper_limit*(np.minimum(a,p_same)-1.0))))


@numba.njit(parallel=False,nopython=False)
def reassign_windows_to_newly_merged_clusters(\
        n_features : np.int64,\
        window_size : np.int64,\
        window_info_size : np.float64,\
        max_cluster_compare_dof : np.float64,\
        mineig : np.float64,\
        d1 : np.float64,\
        chi2Dof : np.float64,\
        penalty_p_upper_limit : np.float64,\
        log_det_of_window_cov_mats : npt.NDArray[np.float64],\
        window_cov_mats : List[npt.NDArray[np.float64]],\
        per_cluster_log_det_cov_mat : List[np.float64],\
        n_dof_per_cluster : npt.NDArray[np.float64],\
        per_cluster_sum_cov_mat : List[npt.NDArray[np.float64]])\
    ->Tuple[np.int64,np.int64,np.int64,np.int64,\
            npt.NDArray[np.int64],\
            npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    n_windows=len(log_det_of_window_cov_mats)
    assert(n_windows==len(window_cov_mats))
    n_clusters=len(per_cluster_log_det_cov_mat)
    assert(n_clusters==len(per_cluster_sum_cov_mat))

    per_window_most_likely_i_cluster=np.zeros(n_windows,dtype=np.int64)
    prob_this_window_in_same_cluster_as_next=np.zeros(n_windows-1,dtype=np.float64)
    beta=np.zeros(n_windows-1,dtype=np.float64)

    p_sum = 0.0
    prob_window_belongs_to_each_cluster=np.zeros(n_clusters,dtype=np.float64)
    this=log_det_of_window_cov_mats[0]
    p_max=0.0
    i_p_max=0
    for i_cluster in range(0,n_clusters):
        yeda=n_dof_per_cluster[i_cluster]+window_info_size
        c1=(1.0/window_info_size+1.0/n_dof_per_cluster[i_cluster]-1.0/yeda)*d1
        C = (window_cov_mats[0]*window_info_size+\
             per_cluster_sum_cov_mat[i_cluster])/yeda
        logDetC=np.sum(np.log(np.maximum(np.abs(np.linalg.eigvalsh(C).flatten()),mineig)))
        chi2stat=(1.0-c1)*(logDetC*yeda-this*window_info_size-\
                           per_cluster_log_det_cov_mat[i_cluster]*n_dof_per_cluster[i_cluster])
        p_same=chi2Ccdf(chi2Dof,chi2stat)
        if p_max < p_same:
            p_max = p_same+0.0
            i_p_max = i_cluster+0
        prob_window_belongs_to_each_cluster[i_cluster]=p_same
        p_sum+=p_same
    if not (0.0<p_sum):
        print("window: ",0,", p_sum: ",p_sum)
        assert(False)
    prob_window_belongs_to_each_cluster/=p_sum
    per_window_most_likely_i_cluster[0]=i_p_max+0

    for i in range(1,n_windows):
        prob_prev_window_belongs_to_each_cluster=\
            np.copy(prob_window_belongs_to_each_cluster)
        p_sum = 0.0
        prob_window_belongs_to_each_cluster=np.zeros(n_clusters,dtype=np.float64)
        this=log_det_of_window_cov_mats[i]
        p_max=0.0
        i_p_max=0
        for i_cluster in range(0,n_clusters):
            yeda=n_dof_per_cluster[i_cluster]+window_info_size
            c1=(1.0/window_info_size+1.0/n_dof_per_cluster[i_cluster]-1.0/yeda)*d1
            C = (window_cov_mats[i]*window_info_size+\
                 per_cluster_sum_cov_mat[i_cluster])/yeda
            logDetC=np.sum(np.log(np.maximum(np.abs(np.linalg.eigvalsh(C).flatten()),mineig)))
            chi2stat=(1.0-c1)*(logDetC*yeda-this*window_info_size-\
                               per_cluster_log_det_cov_mat[i_cluster]*n_dof_per_cluster[i_cluster])
            p_same=chi2Ccdf(chi2Dof,chi2stat)
            if p_max < p_same:
                p_max = p_same+0.0
                i_p_max = i_cluster+0
            prob_window_belongs_to_each_cluster[i_cluster]=p_same
            p_sum+=p_same
        if not (0.0<p_sum):
            print("window: ",i,", p_sum: ",p_sum)
            assert(False)
        prob_window_belongs_to_each_cluster/=p_sum
        p_same=np.dot(prob_prev_window_belongs_to_each_cluster,\
                      prob_window_belongs_to_each_cluster)
        prob_this_window_in_same_cluster_as_next[i-1]=p_same
        beta[i-1]=cluster_switch_penalty_coef(n_features,\
            window_size,penalty_p_upper_limit,p_same)
        per_window_most_likely_i_cluster[i]=i_p_max+0

    # i=n_windows
    # beta[i-1]=cluster_switch_penalty_coef(n_features,\
    #     window_size,penalty_p_upper_limit,\
    #     prob_this_window_in_same_cluster_as_next[i-1])
    return (n_features+0,window_size+0,n_windows+0,n_clusters+0,\
            per_window_most_likely_i_cluster,\
            prob_this_window_in_same_cluster_as_next,beta)


def global_design_basis_clusters(\
        stacked_training_data: npt.NDArray[np.float64],\
        n_features : np.int64,\
        window_size : np.int64,\
        penalty_p_upper_limit : np.float64)\
    ->Tuple[np.int64,np.int64,np.int64,np.int64,\
            npt.NDArray[np.int64],\
            npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    least_squares_factor = compute_least_squares_factor(window_size)

    n_windows = int(stacked_training_data.shape[0])
    n_cols = int(stacked_training_data.shape[1])
    assert(n_cols == n_features * window_size)
    window_info_size=window_size*least_squares_factor
    max_cluster_compare_dof=calc_max_cluster_compare_dof(n_features,window_size)

    window_cov_mats=instantiate_list_of_cov_mats(n_windows,n_features)

    (log_det_of_window_cov_mats,window_cov_mats)=\
     gen_log_det_and_cov_mat_for_all_windows(stacked_training_data,n_features,\
            window_size,n_windows,window_cov_mats)
    print("finished generating cov_mat and log(Det(cov_mat)) for all windows")

    pp1=np.float64(n_features+1)
    L = 2.0*np.float64(n_features*window_size*(n_features*window_size+1))/pp1-2.0+np.float64(n_features+3)/pp1**2
    threshold=np.log(2.0/np.sqrt(3.0))*(26.0+5.0/9.0)/L #if -log(M) is less than this, you can say it's a match without evaluating chi2Ccdf
    f=np.exp(threshold)
    threshold*=n_features*max_cluster_compare_dof
    alpha=2.0*f**2-1.0+2.0*f*np.sqrt(f**2-1.0)
    #print("alpha: ",alpha)
    delta_log=n_features*np.log(alpha) #-delta_log <=log(det(B)/det(A))<=delta_log is the search region/neighborhood that could match, don't check outside this range in first pass
    mineig=2**-np.floor(52/n_features)

    (n_windows_per_cluster,per_cluster_log_det_cov_mat,\
     per_cluster_mean_cov_mat,per_cluster_sum_cov_mat)=\
     gen_initial_clusters(n_windows,window_info_size,max_cluster_compare_dof,\
                          delta_log,mineig,threshold,\
                          log_det_of_window_cov_mats,window_cov_mats)
    print("finished generating initial clusters")

    (n_windows_per_cluster,per_cluster_log_det_cov_mat,\
     per_cluster_mean_cov_mat,per_cluster_sum_cov_mat)=\
     merge_clusters_that_are_similar_enough(delta_log,mineig,threshold,\
        window_info_size,max_cluster_compare_dof,\
        n_windows_per_cluster,per_cluster_log_det_cov_mat,\
        per_cluster_mean_cov_mat,per_cluster_sum_cov_mat)
    print("finished merging clusters that are similar enough; about to regen them")
    n_clusters = len(n_windows_per_cluster)

    #now that i've merged all clusters that are similar enough to each other
    #I'm going to reassign each window to the remaining cluster it best matches
    #and compute probabilities of cluster not switching from window to window
    #and from that a cluster switching penalty for TICC to use

    n_dof_per_cluster=np.zeros(n_clusters,dtype=np.float64)
    for i_cluster in range(0,n_clusters):
        n_dof_per_cluster[i_cluster]=\
            np.minimum(n_windows_per_cluster[i_cluster]*window_info_size,\
                       max_cluster_compare_dof)
        per_cluster_sum_cov_mat[i_cluster]=\
            per_cluster_mean_cov_mat[i_cluster]*n_dof_per_cluster[i_cluster]

    d1=(2.0*n_features**2+3.0*n_features-1.0)/(6.0*pp1)
    chi2Dof=np.float64(n_features*pp1*0.5)

    return reassign_windows_to_newly_merged_clusters(n_features,window_size,\
        window_info_size,max_cluster_compare_dof,mineig,d1,chi2Dof,\
        penalty_p_upper_limit,log_det_of_window_cov_mats,\
        window_cov_mats,per_cluster_log_det_cov_mat,\
        n_dof_per_cluster,per_cluster_sum_cov_mat)


def print_cluster_stats(cluster_data :\
        Tuple[np.int64,np.int64,np.int64,np.int64,\
              npt.NDArray[np.int64],\
              npt.NDArray[np.float64],npt.NDArray[np.float64]]):
    print("window_size: %d, n_windows: %d, n_clusters: %d"\
          %(cluster_data[1],cluster_data[2],cluster_data[3]))


#doesn't worry about deep vs shallow copies because it assumes this function
#is called only once at the end of the optimization loop.
def ship_best_solution(best_cluster_data :\
        Tuple[np.int64,np.int64,np.int64,np.int64,\
              npt.NDArray[np.int64],\
              npt.NDArray[np.float64],npt.NDArray[np.float64]],\
        best_i_dont_gobble_across : npt.NDArray[np.int64],\
        best_stacked_training_data : npt.NDArray[np.float64],
        penalty_p_upper_limit)\
    -> OrderedDict:
    n_features = best_cluster_data[0]
    window_size = best_cluster_data[1]
    n_windows = best_cluster_data[2]
    n_traj=len(best_i_dont_gobble_across)-1
    assert(n_windows==best_i_dont_gobble_across[n_traj])
    assert(best_i_dont_gobble_across[n_traj-1]<best_i_dont_gobble_across[n_traj])
    n_clusters = best_cluster_data[3]
    per_window_most_likely_i_cluster=best_cluster_data[4]
    prob_this_window_in_same_cluster_as_next=best_cluster_data[5]
    beta=best_cluster_data[6]
    #n_effect_time_samples_per_cluster = best_cluster_data[9]

    best_answer=OrderedDict()
    best_answer['n_features']=n_features+0
    best_answer['window_size']=window_size+0
    best_answer['n_windows']=n_windows+0
    best_answer['n_clusters']=n_clusters+0
    best_answer['per_window_most_likely_i_cluster']=\
        per_window_most_likely_i_cluster
    best_answer['prob_this_window_in_same_cluster_as_next']=\
        prob_this_window_in_same_cluster_as_next
    best_answer['beta']=np.copy(beta)
    best_answer['penalty_p_upper_limit']=penalty_p_upper_limit+0.0
    best_answer['stacked_training_data']=np.copy(best_stacked_training_data)
    best_answer['each_trajectorys_left_edge_index_into_stacked_traning_data']=\
        np.copy(best_i_dont_gobble_across) #has one extra index at end = total n_windows
    return best_answer


def print_segments(best_answer : dict, n_traj_to_print : None):
    each_trajectorys_left_edge_index_into_stacked_traning_data=\
        best_answer["each_trajectorys_left_edge_index_into_stacked_traning_data"]
    assert(each_trajectorys_left_edge_index_into_stacked_traning_data[len(each_trajectorys_left_edge_index_into_stacked_traning_data)-2]<\
           each_trajectorys_left_edge_index_into_stacked_traning_data[len(each_trajectorys_left_edge_index_into_stacked_traning_data)-1])
    n_beta=each_trajectorys_left_edge_index_into_stacked_traning_data[\
        len(each_trajectorys_left_edge_index_into_stacked_traning_data)-1]-1
    assert(1<n_beta)
    each_trajectorys_left_edge_index_into_stacked_traning_data=\
        best_answer['each_trajectorys_left_edge_index_into_stacked_traning_data']
    if type(n_traj_to_print) == type(None):
        n_traj_to_print = len(each_trajectorys_left_edge_index_into_stacked_traning_data)-1
    n_windows_to_print=\
        each_trajectorys_left_edge_index_into_stacked_traning_data[n_traj_to_print]
    n_beta_to_print=np.minimum(n_windows_to_print,n_beta)
    rounded_beta=np.round(best_answer['beta'][range(0,n_beta_to_print)]*10.0)/10.0
    per_window_most_likely_i_cluster=\
        best_answer["per_window_most_likely_i_cluster"][range(0,n_windows_to_print)]
    i_contiguous_segment_to_print_start=np.unique((np.where(np.diff(per_window_most_likely_i_cluster))[0]+1).tolist()+each_trajectorys_left_edge_index_into_stacked_traning_data[0:n_traj_to_print+1].tolist()).flatten().tolist()
    n_contiguous_segments_to_print=len(i_contiguous_segment_to_print_start)-1
    print("window_size: %d, n_windows: %d, total_n_clusters: %d"\
          %(best_answer['window_size'],best_answer['n_windows'],\
            best_answer['n_clusters']))
    i_stop=0
    i_traj=0
    print('****************************************************************************************************')
    for i_seg in range(0,n_contiguous_segments_to_print):
        i_start=i_stop+0
        i_stop=i_contiguous_segment_to_print_start[i_seg+1]+0
        n_inc=0
        while each_trajectorys_left_edge_index_into_stacked_traning_data[i_traj+1] <= i_start:
            i_traj+=1
            n_inc+=1
            assert(n_inc==1)
            assert(i_contiguous_segment_to_print_start[i_seg]==each_trajectorys_left_edge_index_into_stacked_traning_data[i_traj])
            assert(i_contiguous_segment_to_print_start[i_seg+1]<=each_trajectorys_left_edge_index_into_stacked_traning_data[i_traj+1])

        assert(n_inc<=1)
        if i_traj == n_traj_to_print:
            break;
        print("traj:",i_traj,", seg:",i_seg,", cluster:",\
            per_window_most_likely_i_cluster[i_start],",",\
            i_stop-i_start, "windows: [",i_start,",",i_stop,"), beta:",\
            rounded_beta[range(i_start,np.minimum(i_stop,n_beta_to_print))].tolist())
        sleep(0.0001)

#ToDo: make the verbosity control do something
def fast_cluster_predictor(data : np.ndarray,\
        penalty_p_lower_limit = 2**-26,\
        window_sizes : int = None,\
        verbosity : int = 0):

    penalty_p_lower_limit = np.float64(np.minimum(0.5,np.maximum(2**-45,penalty_p_lower_limit)))
    penalty_p_upper_limit = np.float64(1.0)-penalty_p_lower_limit

    n_features = data.shape[1]
    mws = int(min_window_size(n_features))
    if type(window_sizes)==type(None):
        window_sizes = np.arange(mws,mws+1,dtype=np.int64).flatten()
    else:
        window_sizes = np.unique(np.maximum(mws,np.int64(np.round(np.atleast_1d(window_sizes)*16.0)/16.0)))
    n_window_sizes = window_sizes.size

    print("******************************************************************")
    print("------------------------------------")
    print("starting search for best window_size")
    best_window_size=0
    for i_window_size in range(0,n_window_sizes):
        window_size=np.int64(window_sizes[i_window_size])
        stacked_training_data=stack_training_data(data,window_sizes[i_window_size])
        i_dont_gobble_across=[0,stacked_training_data.shape[0]]
        print("------------------------------------")
        print("finished stacking training data for window_size: ",window_size)

        cluster_data = global_design_basis_clusters(stacked_training_data,\
            n_features,window_size,penalty_p_upper_limit)
        print_cluster_stats(cluster_data)

        if i_window_size ==0:
            print("*******above is current best**********")
            best_cluster_data = deepcopy(cluster_data)
            best_i_dont_gobble_across=np.copy(i_dont_gobble_across)
            best_window_size = window_size+0
            best_stacked_training_data=np.copy(stacked_training_data)
        elif cluster_data[3]<best_cluster_data[3]:
            print("*******above is current best**********")
            best_cluster_data = deepcopy(cluster_data)
            if not (window_size == best_window_size):
                best_i_dont_gobble_across=np.copy(i_dont_gobble_across)
                best_stacked_training_data = np.copy(stacked_training_data)
                best_window_size = window_size+0

    print("------------------------------------")
    print("chose:")
    print("------------------------------------")
    print_cluster_stats(best_cluster_data)
    return ship_best_solution(best_cluster_data,best_i_dont_gobble_across,\
           best_stacked_training_data,penalty_p_upper_limit)


#ToDo: make the verbosity control do something
def fast_cluster_predictor_multiple_series(all_series: List[np.ndarray],\
        penalty_p_lower_limit = 2**-26,\
        window_sizes : int = None,\
        verbosity : int = 0):

    penalty_p_lower_limit = np.float64(np.minimum(0.5,np.maximum(2**-45,penalty_p_lower_limit)))
    penalty_p_upper_limit = np.float64(1.0)-penalty_p_lower_limit

    n_features = np.int64(all_series[0].shape[1])
    mws = int(min_window_size(n_features))
    if type(window_sizes)==type(None):
        window_sizes = np.arange(mws,mws+1,dtype=np.int64).flatten()
    else:
        window_sizes = np.unique(np.maximum(mws,np.int64(np.round(np.atleast_1d(window_sizes)*16.0)/16.0)))
    n_window_sizes = window_sizes.size

    print("******************************************************************")
    print("------------------------------------")
    print("starting search for best window_size")
    best_window_size=0
    for i_window_size in range(0,n_window_sizes):
        window_size=np.int64(window_sizes[i_window_size])
        (stacked_training_data,i_dont_gobble_across)=\
            stack_training_data_multiple_series(all_series,window_size)
        assert(i_dont_gobble_across[len(i_dont_gobble_across)-2]<i_dont_gobble_across[len(i_dont_gobble_across)-1])
        print("------------------------------------")
        print("finished stacking training data for window_size: ",window_size)

        cluster_data = global_design_basis_clusters(stacked_training_data,\
            n_features,window_size,penalty_p_upper_limit)
        print_cluster_stats(cluster_data)

        if i_window_size ==0:
            print("*******above is current best**********")
            best_cluster_data = deepcopy(cluster_data)
            best_i_dont_gobble_across=np.copy(i_dont_gobble_across)
            best_window_size = window_size+0
            best_stacked_training_data=np.copy(stacked_training_data)
        elif cluster_data[3]<best_cluster_data[3]:
            print("*******above is current best**********")
            best_cluster_data = deepcopy(cluster_data)
            if not (window_size == best_window_size):
                best_i_dont_gobble_across=np.copy(i_dont_gobble_across)
                best_stacked_training_data = np.copy(stacked_training_data)
                best_window_size = window_size+0

    print("------------------------------------")
    print("chose:")
    print("------------------------------------")
    print_cluster_stats(best_cluster_data)
    return ship_best_solution(best_cluster_data,best_i_dont_gobble_across,\
           best_stacked_training_data,penalty_p_upper_limit)