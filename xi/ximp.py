import numpy as np
import warnings
from typing import *
from utils import *
from xi.exceptions import *
from xi import PARTITIONS
import time
from collections import defaultdict

# A class XI which will have method
# - explain_instances (?) ( better just explain maybe)
# - produce_plot
# --> produce plot should accept a parameter that will say
# what type of data we have.
# Then we need a class Mapper to map the type to the function
# responsible for the plot. Inside produce_plot we will call Mapper
# and then mapper will delegate the plot to the appropriate function.

class XI(object):

    def __init__(self,
                 m: Union[dict, int] = None,
                 obs: dict = None,
                 discrete: dict = None,
                 ties=False):

        self.m = m
        self.obs = obs
        self.discrete = discrete
        self.ties = ties

    def _compute_default_m(self,
                           n: int):

        val = np.ceil(np.sqrt(n)).astype('int')

        return val

    def _extract_partitions(self, idx):

        # Ugly af need to change

        partitions = self.obs.get(idx, None) if self.obs is not None else None
        if partitions is not None:
            return partitions, PARTITIONS.OBSERVATION

        partitions = self.m.get(idx, None) if self.m is not None else None
        if partitions is not None:
            return partitions, PARTITIONS.M

        partitions = self.discrete.get(idx, None) if self.discrete is not None else None
        if partitions is not None:
            return partitions, PARTITIONS.DISCRETE

        if partitions is None:
            return partitions, PARTITIONS.M

    def _build_zeros_like_matrix(self,
                                 row: int,
                                 col: int,
                                 names: List = None):
        out = {}
        for n in names:
            out[n] = np.zeros((row, col))

        return out

    def explain(self,
                X: np.array,
                y: np.array):

        n, k = X.shape

        partition_validation(self.m, k)
        partition_validation(self.obs, k)
        partition_validation(self.discrete, k)

        check_args_overlap(self.m,
                           self.obs,
                           self.discrete)

        # self._compute_default_m(n, k)

        (uniquey, totalmass) = np.unique(y, return_counts=True)

        totalmass = nrmd(totalmass)

        if np.any(totalmass == 0):
            XiError('Empty class. You should check labels')

        if self.ties:
            replicates = 50
        else:
            replicates = 1

        matrix_replicas = self._build_zeros_like_matrix(row=k,
                                                        col=replicates,
                                                        names=['B_replica',
                                                               'D_replica',
                                                               'M_replica',
                                                               'H_replica',
                                                               'K_replica',
                                                               'Q_replica'])

        for replica in range(replicates):
            ix = np.argsort(X + np.random.rand(*X.shape), axis=0)
            for idx in range(k):
                partitions, types = self._extract_partitions(idx)
                if partitions is None:
                    partitions = self._compute_default_m(n=n)

                if types == PARTITIONS.OBS:
                    partitions = len(np.unique(X[:, idx]))
                if types == PARTITIONS.DISCRETE:
                    partitions = np.ceil(n / (-partitions)).astype('int')

                indx = np.round(np.linspace(start=0,
                                            stop=n,
                                            num=partitions + 1)).astype('int')


if __name__=='__main__':

    X = np.random.normal(3, 7, size=100 * 1000)  # df_np[:, 1:11]
    X = X.reshape((1000, 100))
    Y = np.array(np.random.uniform(2,4,size=1000))

    start_time = time.perf_counter()

    xi = XI()
    xi.explain(X=X,y=Y)
    #P_measures = Ximp(X, Y, None, m=100, ties=False)
    end_time = time.perf_counter()
    print(end_time - start_time, "seconds")
    #print(P_measures)

# def Ximp(X: np.array,
#          y: np.array,
#          nc: int = None,
#          m: dict = None,
#          obs: dict = None,
#          discrete: dict = None,
#          ties=False) -> Dict:
#     n, k = X.shape
#     partition_validation(m, k)
#     partition_validation(obs, k)
#     partition_validation(discrete, k)
#     check_args_overlap(m, obs, discrete)
#
#     if nc is None:
#         classlist = np.unique(y)
#
#     else:
#         classlist = np.arange(nc)
#
#     if m is None:
#         m = np.repeat(np.ceil(np.sqrt(n)).astype('int'), k)
#
#     """
#
#     :param x: Design matrix
#     :param y: Response variable
#     :param nc: number of target classes, integer
#     :param m: A dictionary where each key represent the position of the features and value the number of desired partitions
#     for that variable. Ex : {1: 10, 2: 10}. The remaining covariates will take the default value as described in
#     [INSERT PAPER REFERENCE]
#     :param obs:  A dictionary where each key represent the position of the covariate and value the number of observation
#     for that variable in each partition. Ex : {1: 10, 2: 10}. The remaining covariates will take the default value as described in
#     [INSERT PAPER REFERENCE]
#     :param discrete: A dictionary where each key represent the position of the covariate and value as 1 stating if variable
#     should be treated as discrete. Ex : {1: 1, 2: 1}. The remaining covariates are assumed to not be discrete
#     :param ties:  if True, we return estimators accounting for the variability in the ranking of ties. The default is False.
#
#     :return: Dictionary
#     """
#
#     # In particular, if m[i]>0, then it is the number of desired partitions for the ith variable,
#     # if m[i]=0 then the ith covariate should be treated as discrete,
#     # if m[i]<0 then m[i] is the number of desired observations in each partition
#     n, k = X.shape
#
#     # m might be a dictionary {'1': 10,
#     #                          '2': 20 }
#     #
#
#     partition_validation(m, k)
#     partition_validation(obs, k)
#     partition_validation(discrete, k)
#     check_args_overlap(m, obs, discrete)
#
#     # k = x.shape[1]
#     if isinstance(m, int):
#         m = np.repeat(m, k)
#
#     if nc is None:
#         classlist = np.unique(y)
#
#     else:
#         classlist = np.arange(nc)
#
#     if m is None:
#         m = np.repeat(np.ceil(np.sqrt(n)).astype('int'), k)  # default partitioning: n^1/2 samples in n^1/2 partitions
#
#     (uniquey, totalmass) = np.unique(y, return_counts=True)
#
#     totalmass = nrmd(totalmass)
#
#     if np.any(totalmass == 0):
#         # Extract what labels are zero and output in the message
#         XiError('Empty class. You should check labels')
#
#     if ties:
#         replicates = 50
#     else:
#         replicates = 1
#
#     B_replica = np.zeros((k, replicates))  # each column to a replica
#     D_replica = np.zeros((k, replicates))
#     M_replica = np.zeros((k, replicates))
#     H_replica = np.zeros((k, replicates))
#     K_replica = np.zeros((k, replicates))
#     Q_replica = np.zeros((k, replicates))
#
#     for replica in range(replicates):
#         ix = np.argsort(X + np.random.rand(*X.shape), axis=0)  # Here I am randomizing the ranking of ties
#         xs = np.sort(X, axis=0)
#
#         for mm in range(len(m)):
#             m_part = m[mm]
#             if m_part == 0:
#                 m_part = len(np.unique(X[:, mm]))
#             elif m_part < 0:
#                 m_part = np.ceil(n / (-m_part)).astype('int')
#             indx = np.round(np.linspace(start=0, stop=n, num=m_part + 1)).astype('int')
#             Bi = np.zeros((m_part, k))
#             Di = np.zeros((m_part, k))
#             Mi = np.zeros((m_part, k))
#             Vi = np.zeros((m_part, k))
#             Li = np.zeros((m_part, k))
#             Ki = np.zeros((m_part, k))
#             Hi = np.zeros((m_part, k))
#             Qi = np.zeros((m_part, k))
#             for i in range(m_part):
#                 z = y[ix[indx[i]:indx[i + 1], :]]
#                 for j in range(k):
#
#                     condmass = np.zeros(len(uniquey))
#                     for kk in range(len(uniquey)):
#                         condmass[kk] = np.count_nonzero(z[:, j] == uniquey[kk])
#                     # print(condmass)
#                     condmass = nrmd(condmass)
#
#                     dmass = np.subtract(condmass, totalmass)
#
#                     Bi[i, j] = np.max(np.abs(dmass))
#                     Mi[i, j] = np.max(dmass) - np.min(dmass)
#                     Di[i, j] = np.sum(np.abs(dmass))
#                     # Power Divergence not implemented here
#                     # Kullback - Leibler
#                     kl = np.multiply(condmass, np.log(np.divide(condmass, totalmass)))
#                     kl[np.isnan(kl)] = 0
#                     Ki[i, j] = np.sum(kl)
#                     # Hellinger
#                     Hi[i, j] = 1 - np.sum(np.sqrt(np.multiply(condmass, totalmass)))
#                     Qi[i, j] = np.sum(np.square(dmass))
#
#             B_replica[:, replica] = np.mean(Bi, axis=0)
#             D_replica[:, replica] = np.mean(Di, axis=0)
#             M_replica[:, replica] = np.mean(Mi, axis=0)
#             H_replica[:, replica] = np.mean(Hi, axis=0)
#             K_replica[:, replica] = np.mean(Ki, axis=0)
#             Q_replica[:, replica] = np.mean(Qi, axis=0)
#
#         B = np.mean(B_replica, axis=1)
#         D = np.mean(D_replica, axis=1)
#         M = np.mean(M_replica, axis=1)
#         K = np.mean(K_replica, axis=1)
#         H = np.mean(H_replica, axis=1)
#         Q = np.mean(Q_replica, axis=1)
#
#     return {"l1": D, "l2": Q, "Kuiper": M, "linf": B, "KullbackLeibler": K, "Hellinger": H}
