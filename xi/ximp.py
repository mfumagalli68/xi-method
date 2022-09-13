import numpy as np
import warnings
from typing import *
from utils import *
from xi.exceptions import *


def Ximp(x: np.array,
         y: np.array,
         nc: int = None,
         m: dict = None,
         obs: dict = None,
         discrete: dict = None,
         ties=False) -> Dict:
    """

    :param x: Design matrix
    :param y: Response variable
    :param nc: number of target classes, integer
    :param m: A dictionary where each key represent the position of the features and value the number of desired partitions
    for that variable. Ex : {1: 10, 2: 10}. The remaining covariates will take the default value as described in
    [INSERT PAPER REFERENCE]
    :param obs:  A dictionary where each key represent the position of the covariate and value the number of observation
    for that variable in each partition. Ex : {1: 10, 2: 10}. The remaining covariates will take the default value as described in
    [INSERT PAPER REFERENCE]
    :param discrete: A dictionary where each key represent the position of the covariate and value as 1 stating if variable
    should be treated as discrete. Ex : {1: 1, 2: 1}. The remaining covariates are assumed to not be discrete
    :param ties:  if True, we return estimators acounting the variability in the ranking of ties. The default is False.

    :return: Dictionary
    """
    '''
    Input:
    x - design matrix
    y - response
    nc - number of target classes, integer
    
    m - vector, containing number of desired partitions for each covariate .
    In particular, if m[i]>0, then it is the number of desired partitions for the ith variable,
    if m[i]=0 then the ith covariate should be treated as discrete,
    if m[i]<0 then m[i] is the number of desired observations in each partition
    ties - if True, we return estimators acounting the variability in the ranking of ties. The default is False.

    Output:
    dictionary

    '''
    # In particular, if m[i]>0, then it is the number of desired partitions for the ith variable,
    # if m[i]=0 then the ith covariate should be treated as discrete,
    # if m[i]<0 then m[i] is the number of desired observations in each partition
    n, k = x.shape

    # m might be a dictionary {'1': 10,
    #                          '2': 20 }
    #

    partition_validation(m, k)
    partition_validation(obs, k)
    partition_validation(discrete, k)
    check_args_overlap(m, obs, discrete)

    # k = x.shape[1]
    if isinstance(m, int):
        m = np.repeat(m, k)

    if nc is None:
        classlist = np.unique(y)

    else:
        classlist = np.arange(nc)

    if m is None:
        m = np.repeat(np.ceil(np.sqrt(n)).astype('int'), k)  # default partitioning: n^1/2 samples in n^1/2 partitions

    (uniquey, totalmass) = np.unique(y, return_counts=True)

    # TODO
    # totalmass = nrmd(totalmass)

    if np.any(totalmass == 0):
        # Extract what labels are zero and output in the message
        XiError('Empty class. You should check labels')

    if ties:
        replicates = 50
    else:
        replicates = 1

    B_replica = np.zeros((k, replicates))  # each column to a replica
    D_replica = np.zeros((k, replicates))
    M_replica = np.zeros((k, replicates))
    H_replica = np.zeros((k, replicates))
    K_replica = np.zeros((k, replicates))
    Q_replica = np.zeros((k, replicates))

    for replica in range(replicates):
        ix = np.argsort(x + np.random.rand(*x.shape), axis=0)  # Here I am randomizing the ranking of ties
        xs = np.sort(x, axis=0)

        for mm in range(len(m)):
            m_part = m[mm]
            if m_part == 0:
                m_part = len(np.unique(X[:, mm]))
            elif m_part < 0:
                m_part = np.ceil(n / (-m_part)).astype('int')
            indx = np.round(np.linspace(start=0, stop=n, num=m_part + 1)).astype('int')
            Bi = np.zeros((m_part, k))
            Di = np.zeros((m_part, k))
            Mi = np.zeros((m_part, k))
            Vi = np.zeros((m_part, k))
            Li = np.zeros((m_part, k))
            Ki = np.zeros((m_part, k))
            Hi = np.zeros((m_part, k))
            Qi = np.zeros((m_part, k))
            for i in range(m_part):
                z = y[ix[indx[i]:indx[i + 1], :]]
                for j in range(k):

                    condmass = np.zeros(len(uniquey))
                    for kk in range(len(uniquey)):
                        condmass[kk] = np.count_nonzero(z[:, j] == uniquey[kk])
                    # print(condmass)
                    condmass = nrmd(condmass)

                    dmass = np.subtract(condmass, totalmass)

                    Bi[i, j] = np.max(np.abs(dmass))
                    Mi[i, j] = np.max(dmass) - np.min(dmass)
                    Di[i, j] = np.sum(np.abs(dmass))
                    # Power Divergence not implemented here
                    # Kullback - Leibler
                    kl = np.multiply(condmass, np.log(np.divide(condmass, totalmass)))
                    kl[np.isnan(kl)] = 0
                    Ki[i, j] = np.sum(kl)
                    # Hellinger
                    Hi[i, j] = 1 - np.sum(np.sqrt(np.multiply(condmass, totalmass)))
                    Qi[i, j] = np.sum(np.square(dmass))

            B_replica[:, replica] = np.mean(Bi, axis=0)
            D_replica[:, replica] = np.mean(Di, axis=0)
            M_replica[:, replica] = np.mean(Mi, axis=0)
            H_replica[:, replica] = np.mean(Hi, axis=0)
            K_replica[:, replica] = np.mean(Ki, axis=0)
            Q_replica[:, replica] = np.mean(Qi, axis=0)

        B = np.mean(B_replica, axis=1)
        D = np.mean(D_replica, axis=1)
        M = np.mean(M_replica, axis=1)
        K = np.mean(K_replica, axis=1)
        H = np.mean(H_replica, axis=1)
        Q = np.mean(Q_replica, axis=1)

    return {"l1": D, "l2": Q, "Kuiper": M, "linf": B, "KullbackLeibler": K, "Hellinger": H}
