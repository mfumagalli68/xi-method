import numpy as np
import warnings
from typing import *
from utils import *

def Ximp(x,
         y,
         nc=None,
         m=None,
         ties=False):
    '''
    Input:
    x - design matrix
    y - response
    nc - number of target classes, integer
    m - vector, containing number of desired partitions for each covariate . In particular, if m[i]>0,
    then it is the number of desired partitions for the ith variable, if m[i]=0 then the ith covariate should be treated as discrete, if m[i]<0 then
        -m[i] is the number of desired observations in each partition
    ties - if True, we return estimators acounting the variability in the ranking of ties. The default is False.

    Output:
    dictionary

    '''
    n, k = x.shape

    m_validation(m,k)

    # k = x.shape[1]
    if isinstance(m, int):
        m = np.repeat(m, k)

    if k != len(m) and k != 1:
        warnings.warn("You need to input an integer number of partition for each covariate.")
    if nc is None:
        classlist = np.unique(y)

    else:
        classlist = np.arange(nc)

    if m is None:
        m = np.repeat(np.ceil(np.sqrt(n)).astype('int'), k)  # default partitioning: n^1/2 samples in n^1/2 partitions

    (uniquey, totalmass) = np.unique(y, return_counts=True)
    totalmass = nrmd(totalmass)

    if np.any(totalmass == 0):
        warnings.warn("Empty Class. Check labels.")

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
