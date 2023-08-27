import abc
import logging
import multiprocessing
from operator import itemgetter
from scipy.stats import rv_histogram
from xi.plotting.plot import *
from xi.separation.measurement import *
from xi.utils import *
import datetime
import time
from tqdm import tqdm
from joblib import Parallel, delayed


class XI(object):

    def __init__(self,
                 m: Union[dict, int] = None,
                 obs: dict = None,
                 discrete: List = None,
                 type: AnyStr = 'classifier',
                 ties=False,
                 grid: int = 100):
        self.m = {} if m is None else m
        self.obs = {} if obs is None else obs
        self.discrete = [] if discrete is None else discrete
        self.ties = ties
        self.grid = grid
        self.type = type

    def _compute_default_m(self,
                           n: int) -> int:
        """
        Compute default partition number
        :param n:
        :return:
        """
        val = np.ceil(np.log(n)).astype('int')

        return val

    @abc.abstractmethod
    def explain(self, *args, **kwargs):
        pass


class XIClassifier(XI):

    def __init__(self,
                 m: Union[dict, int] = None,
                 obs: dict = None,
                 discrete: List = None,
                 ties=False,
                 type='classifier'):

        super(XIClassifier, self).__init__(m=m, obs=obs, discrete=discrete, ties=ties, type=type)

    def _compute_partitions(self, col: AnyStr, n: int) -> int:

        if isinstance(self.m, int):
            return self.m

        partition = self._compute_default_m(n)

        if col in self.m.keys():
            return self.m.get(col)

        if col in self.discrete:
            partition = n

        if col in self.obs.keys():
            desired_obs = self.obs.get(col)
            partition = np.ceil(n / desired_obs).astype('int')

        return partition

    @timeit
    def explain(self,
                X: pd.DataFrame,
                y: np.array,
                replicates: int = 1,
                separation_measurement: Union[AnyStr, List] = 'L1',
                multiprocess=False,
                verbose=False) -> Dict:
        """
        Provide post-hoc explanations

        :param X: Design matrix, without target variable
        :param y: target variable
        :param replicates: number of replications
        :param separation_measurement: Separation measurement.
        :param multiprocess:
        :param verbose:
        Read documentation for the implemented ones.
        You can specify one or more than one as list.

        :return: dictionary mapping separation measurement name to object containing explanations
        for each covariates
        """

        if isinstance(separation_measurement, str):
            separation_measurement = [separation_measurement]

        if isinstance(X, pd.DataFrame):
            mapping_col = {k: v for k, v in zip(range(X.shape[1]), X.columns)}
        else:
            mapping_col = {k: str(v) for k, v in zip(range(X.shape[1]), range(X.shape[1]))}

        X = X.values if isinstance(X, pd.DataFrame) else X
        n, k = X.shape

        X = np.float32(X)

        partition_validation(self.m, k)
        partition_validation(self.obs, k)
        partition_validation(self.discrete, k)

        separation_measurement_validation(measure=separation_measurement)

        check_args_overlap(self.m,
                           self.obs,
                           self.discrete)

        (uniquey, totalmass) = np.unique(y, return_counts=True)

        totalmass = nrmd(totalmass)

        if np.any(totalmass == 0):
            XiError('Empty class. You should check labels')

        if self.ties:
            replicates = 50

        # Input from user, one or multiple sep measure?
        factory = SepMeasureFactory()

        builder_names = itemgetter(*separation_measurement)(builder_mapping)
        if isinstance(builder_names, type):
            builder_names = [builder_names]

        seps = {}

        # register builder
        for _sep_measure, builder_name in zip(separation_measurement, builder_names):
            factory.register_builder(_sep_measure, builder_name())

        logging.info('Beginning computing explanations...')

        for replica in range(replicates):

            ix = np.argsort(X + np.random.rand(*X.shape), axis=0)

            for idx in tqdm(range(k)):

                col = mapping_col.get(idx)
                if verbose:
                    logging.info(f'Computing explanation value for variable {col}')

                partitions = self._compute_partitions(col=col, n=n)

                # builder registered. First iteration
                # builder will create object.
                # Second iteration it won't overwrite since it's already created.
                for _sep_measure, builder_name in zip(separation_measurement, builder_names):
                    seps[_sep_measure] = factory.create(_sep_measure,
                                                        row=partitions,
                                                        col=k,
                                                        replica=replicates,
                                                        idx_to_col=mapping_col)

                indx = np.round(np.linspace(start=0,
                                            stop=n,
                                            num=partitions + 1)).astype('int')

                def _compute_pmf(j):
                    condmass = np.zeros(len(uniquey))
                    for kk in range(len(uniquey)):
                        condmass[kk] = np.count_nonzero(z[:, j] == uniquey[kk])

                    condmass = nrmd(condmass)

                    dmass = np.subtract(condmass, totalmass)

                    return (condmass, dmass)

                for i in range(partitions):

                    z = y[ix[indx[i]:indx[i + 1], :]]

                    if multiprocess:
                        n_jobs = multiprocessing.cpu_count() - 1
                        results = Parallel(n_jobs=n_jobs) \
                            (delayed(_compute_pmf)(j) for j in range(k))
                    else:
                        results = []
                        for j in range(k):
                            results.append(_compute_pmf(j))

                    for j in range(k):
                        for _, _sep in seps.items():
                            _sep.compute(i=i, j=j, dmass=results[j][1], condmass=results[j][0], totalmass=totalmass,
                                         type=self.type)

            for _, _sep in seps.items():
                _sep.avg_replica(replica=replica)

        for _, _sep in seps.items():
            _sep.avg()

        return seps


class XIRegressor(XI):

    def __init__(self,
                 m: Union[dict, int] = 20,
                 grid: int = None,
                 ties=False,
                 type='regressor'):
        super(XIRegressor, self).__init__(m=m, grid=grid, ties=ties, type=type)

    def explain(self,
                X: pd.DataFrame,
                y: np.array,
                replicates: int,
                separation_measurement: Union[AnyStr, List],
                multiprocess=False,
                verbose=False) -> Dict:

        if isinstance(separation_measurement, str):
            separation_measurement = [separation_measurement]

        if isinstance(X, pd.DataFrame):
            mapping_col = {k: v for k, v in zip(range(X.shape[1]), X.columns)}
        else:
            mapping_col = {k: str(v) for k, v in zip(range(X.shape[1]), range(X.shape[1]))}

        X = X.values if isinstance(X, pd.DataFrame) else X
        n, k = X.shape

        X = np.float32(X)

        partition_validation(self.m, k)
        if self.m is None:
            partition = self._compute_default_m(n=n)
        else:
            partition = self.m

        separation_measurement_validation(measure=separation_measurement)

        histogram_dist = rv_histogram(np.histogram(y, bins='auto'))
        y_grid = np.linspace(np.min(y), np.max(y), self.grid)

        totalmass = [histogram_dist.pdf(point) for point in y_grid]
        totalmass = nrmd(totalmass)

        if self.ties:
            replicates = 50
        else:
            replicates = 1

        # Input from user, one or multiple sep measure?
        factory = SepMeasureFactory()

        builder_names = itemgetter(*separation_measurement)(builder_mapping)
        if isinstance(builder_names, type):
            builder_names = [builder_names]

        seps = {}

        # register builder
        for _sep_measure, builder_name in zip(separation_measurement, builder_names):
            factory.register_builder(_sep_measure, builder_name())

        if verbose:
            logging.info(f'Computing explanation value for variable {col}')
        for replica in range(replicates):

            for idx in tqdm(range(k)):

                col = mapping_col.get(idx)
                logging.info(f'Computing explanation value for variable {col}')
                # builder registered. First iteration
                # builder will create object.
                # Second iteration it won't overwrite since it's already created.
                for _sep_measure, builder_name in zip(separation_measurement, builder_names):
                    seps[_sep_measure] = factory.create(_sep_measure,
                                                        row=self.m,
                                                        col=k,
                                                        replica=replicates,
                                                        idx_to_col=mapping_col)

                indx = np.round(np.linspace(start=0,
                                            stop=n,
                                            num=self.m + 1)).astype('int')

                def _compute_pmf(j):
                    ix = np.argsort(X[:, j] + np.random.rand(*X[:, j].shape), axis=0)
                    z = y[ix[indx[i]:indx[i + 1]]]
                    dmass = rv_histogram(np.histogram(z, bins='auto'))
                    condmass = [dmass.pdf(point) for point in y_grid]
                    condmass = nrmd(condmass)

                    return (condmass, dmass)

                for i in range(partition):

                    if multiprocess:
                        n_jobs = multiprocessing.cpu_count() - 1
                        results = Parallel(n_jobs=n_jobs) \
                            (delayed(_compute_pmf)(j) for j in range(k))
                    else:
                        results = []
                        for j in range(k):
                            results.append(_compute_pmf(j))

                    for j in range(k):
                        for _, _sep in seps.items():
                            _sep.compute(i=i, j=j, dmass=results[j][1], condmass=results[j][0], totalmass=totalmass,
                                         type=self.type)

            for _, _sep in seps.items():
                _sep.avg_replica(replica=replica)

        for _, _sep in seps.items():
            _sep.avg()

        return seps

#
# if __name__ == '__main__':
#     import time
#
#     np.random.seed(2)
#
#     for idx, n_var in enumerate([6, 8, 10, 12, 14, 16]):
#         tot = n_var * n_var
#         size = 500_000
#         print('Experiment {}.\n Parameter n_variables {} size={}'.format(idx, tot, size))
#         Y = np.random.choice(np.array([1, 2, 3]), size=size)
#         X = np.random.normal(n_var, n_var, size=tot * size)  # df_np[:, 1:11]
#         X = X.reshape((size, tot))
#         start_time = time.time()
#         # xi = XIClassifier(m=50)
#         # p = xi.explain(X=X, y=Y, replicates=1, separation_measurement='Kullback-leibler', multiprocess=True)
#         # print("--- With multiprocess: %s seconds ---" % (time.time() - start_time))
#         #
#         # start_time = time.time()
#         xi = XIClassifier()
#         p = xi.explain(X=X, y=Y, replicates=1, separation_measurement='L1', multiprocess=False, verbose=True)
#         print("--- With multiprocess: %s seconds ---" % (time.time() - start_time))
#
#     # start_time = time.time()
#     # xi = XIClassifier(m=300)
#     # p = xi.explain(X=X, y=Y, replicates=1, separation_measurement='Kullback-leibler')
#     # print("--- Without multiprocess:%s seconds ---" % (time.time() - start_time))

#     # X = np.random.normal(3, 7, size=5 * 100000)  # df_np[:, 1:11]
#     # X = X.reshape((100000, 5))
#     # reading from the file
#     # df = pd.read_csv("/tests/data/winequality-red.csv", sep=";")
#     df = load_digits()
#     Y = df.target
#     df = df.data
#
#
#     # df.drop(columns='quality', inplace=True)
#
#     # Y = np.array(np.random.randint(2, 4, size=100000))
#
#     # start_time = time.perf_counter()
#
#     # df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
#
#     def compute_attempt(dmass, condmass, totalmass, **kwargs):
#         return np.sum(np.sqrt(np.multiply(condmass, totalmass)))
#
#
#     # cust = CustomSeparationMeasure(separation_measure={'test':compute_attempt})
#     # cust.register()
#
#     xi = XIClassifier(m=20)
#     P = xi.explain(X=df, y=Y, separation_measure='L1')
#     # plot(separation_measurement='L1', type='tabular', explain=P, k=10)
#     # P_measures = Ximp(X, Y, None, m=100, ties=False)
#     # end_time = time.perf_counter()
#     # print(end_time - start_time, "seconds")
#     val = P.get('L1').value
#     plot(separation_measurement='L1',
#          type='image',
#          explain=P,
#          k=10,
#          shape=(8, 8))
#     print(val)
