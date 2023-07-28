import numpy as np
import warnings
from typing import *
from utils import *
from xi.exceptions import *
from xi import PARTITIONS
import time
from collections import defaultdict
import abc
import pandas as pd
from xi.measure import *


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
                 discrete: List = None,
                 ties=False):
        self.m = {} if m is None else m
        self.obs = {} if obs is None else obs
        self.discrete = [] if discrete is None else discrete
        self.ties = ties

    @abc.abstractmethod
    def explain(self, *args, **kwargs):
        pass


class XIClassifier(XI):

    def __init__(self,
                 m: Union[dict, int] = None,
                 obs: dict = None,
                 discrete: List = None,
                 ties=False):

        super(XIClassifier, self).__init__(m=m, obs=obs, discrete=discrete, ties=ties)

    def _compute_default_m(self,
                           n: int):

        val = np.ceil(np.sqrt(n)).astype('int')

        return val

    def _compute_partitions(self, col, n):

        partition = self._compute_default_m(n)

        if col in self.m.keys():
            return self.m.get(col)

        if col in self.discrete:
            partition = n

        if col in self.obs.keys():
            desired_obs = self.m.get(col)
            partition = np.ceil(n / desired_obs).astype('int')

        return partition

    def _get_missing_covariates_partition(self, full_columns) -> List:

        # Ugly af need to change

        cols = list(set().union(*self.obs, self.m))
        cols.extend(self.discrete)
        missing_col = [x for x in full_columns if x not in cols]

        return missing_col

    def explain(self,
                X: pd.DataFrame,
                y: np.array,
                separation_measure="kuiper"):

        full_columns = X.columns
        mapping_col = {k: v for k, v in zip(range(X.shape[1]), X.columns)}

        X = X.values if isinstance(X, pd.DataFrame) else X
        n, k = X.shape

        partition_validation(self.m, k)
        partition_validation(self.obs, k)
        partition_validation(self.discrete, k)
        # Validate separation_measure name

        # TODO fix it
        # check_args_overlap(self.m,
        #                    self.obs,
        #                    self.discrete)

        # self._compute_default_m(n, k)

        (uniquey, totalmass) = np.unique(y, return_counts=True)

        totalmass = nrmd(totalmass)

        if np.any(totalmass == 0):
            XiError('Empty class. You should check labels')

        if self.ties:
            replicates = 50
        else:
            replicates = 1

        # Input from user, one or multiple sep measure?
        factory = SepMeasureFactory()

        _sep_measure = f'{separation_measure}'

        factory.register_builder(_sep_measure, KuiperBuilder())
        Sep_i = factory.create(_sep_measure, row=n, col=k,replica=replicates)

        for replica in range(replicates):

            ix = np.argsort(X + np.random.rand(*X.shape), axis=0)

            # missing_cols = self._get_missing_covariates_partition(full_columns)

            for idx in range(k):
                col = mapping_col.get(idx)
                partitions = self._compute_partitions(col=col, n=n)

                indx = np.round(np.linspace(start=0,
                                            stop=n,
                                            num=partitions + 1)).astype('int')

                for i in range(partitions):
                    z = y[ix[indx[i]:indx[i + 1], :]]
                    for j in range(k):

                        condmass = np.zeros(len(uniquey))
                        for kk in range(len(uniquey)):
                            condmass[kk] = np.count_nonzero(z[:, j] == uniquey[kk])
                        # print(condmass)
                        condmass = nrmd(condmass)

                        dmass = np.subtract(condmass, totalmass)

                        Sep_i.compute(i=i, j=j, dmass=dmass, condmass=condmass, totalmass=totalmass)
                        # Mi[i, j] = np.max(dmass) - np.min(dmass)
                        # Di[i, j] = np.sum(np.abs(dmass))
                        # Power Divergence not implemented here
                        # Kullback - Leibler
                        # kl = np.multiply(condmass, np.log(np.divide(condmass, totalmass)))
                        # kl[np.isnan(kl)] = 0
                        # Ki[i, j] = np.sum(kl)
                        # Hellinger
                        # Hi[i, j] = 1 - np.sum(np.sqrt(np.multiply(condmass, totalmass)))
                        # Qi[i, j] = np.sum(np.square(dmass))

                Sep_i.avg()
                # B_replica[:, replica] = np.mean(Bi, axis=0)
                # D_replica[:, replica] = np.mean(Di, axis=0)
                # M_replica[:, replica] = np.mean(Mi, axis=0)
                # H_replica[:, replica] = np.mean(Hi, axis=0)
                # K_replica[:, replica] = np.mean(Ki, axis=0)
                # Q_replica[:, replica] = np.mean(Qi, axis=0)

            # B = np.mean(B_replica, axis=1)
            # D = np.mean(D_replica, axis=1)
            # M = np.mean(M_replica, axis=1)
            # K = np.mean(K_replica, axis=1)
            # H = np.mean(H_replica, axis=1)
            # Q = np.mean(Q_replica, axis=1)

        return None  # {"l1": D, "l2": Q, "Kuiper": M, "linf": B, "KullbackLeibler": K, "Hellinger": H}


if __name__ == '__main__':
    X = np.random.normal(3, 7, size=5 * 1000)  # df_np[:, 1:11]
    X = X.reshape((1000, 5))
    Y = np.array(np.random.uniform(2, 4, size=1000))

    start_time = time.perf_counter()

    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    xi = XIClassifier(obs={'col_4': 10}, discrete=['col_1', 'col_3'], m={'col_5': 10})
    xi.explain(X=df, y=Y)
    # P_measures = Ximp(X, Y, None, m=100, ties=False)
    end_time = time.perf_counter()
    print(end_time - start_time, "seconds")
    # print(P_measures)

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
