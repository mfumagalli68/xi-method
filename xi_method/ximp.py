import abc
import logging
from operator import itemgetter
from scipy.stats import rv_histogram
from xi_method.plotting.plot import *
from xi_method.separation.measurement import *
from xi_method.utils import *

from tqdm import tqdm


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
                verbose=False) -> Dict:
        """
        Provide post-hoc explanations

        :param X: Design matrix, without target variable
        :param y: target variable
        :param replicates: number of replications
        :param separation_measurement: Separation measurement.
        :param verbose: True, for logging. Default False.

        :return: Dictionary mapping separation measurement name to object containing explanations
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

        #X = np.float32(X)

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

                for i in range(partitions):

                    z = y[ix[indx[i]:indx[i + 1], :]]
                    for j in range(k):
                        condmass = np.zeros(len(uniquey))
                        for kk in range(len(uniquey)):
                            condmass[kk] = np.count_nonzero(z[:, j] == uniquey[kk])

                        condmass = nrmd(condmass)

                        dmass = np.subtract(condmass, totalmass)

                        for _, _sep in seps.items():
                            _sep.compute(i=i, j=j, dmass=dmass, condmass=condmass, totalmass=totalmass,
                                         type=self.type)

            for _, _sep in seps.items():
                _sep.avg_replica(replica=replica)

        for _, _sep in seps.items():
            _sep.avg()

        return seps


class XIRegressor(XI):

    def __init__(self,
                 m: Union[dict, int] = 20,
                 grid: int = 100,
                 ties=False,
                 type='regressor'):
        super(XIRegressor, self).__init__(m=m, grid=grid, ties=ties, type=type)

    def explain(self,
                X: pd.DataFrame,
                y: np.array,
                replicates: int,
                separation_measurement: Union[AnyStr, List],
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

        for replica in range(replicates):

            for idx in tqdm(range(k)):

                col = mapping_col.get(idx)
                if verbose:
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

                for i in range(partition):

                    for j in range(k):

                        ix = np.argsort(X[:, j] + np.random.rand(*X[:, j].shape), axis=0)
                        z = y[ix[indx[i]:indx[i + 1]]]
                        dmass = rv_histogram(np.histogram(z, bins='auto'))
                        condmass = [dmass.pdf(point) for point in y_grid]
                        condmass = nrmd(condmass)
                        condmass_supp = nrmd(condmass[condmass != 0])
                        totalmass_supp = nrmd(totalmass[condmass != 0])

                        for _, _sep in seps.items():
                            _sep.compute(i=i, j=j, dmass=dmass, condmass=condmass_supp, totalmass=totalmass_supp,
                                         type=self.type)

            for _, _sep in seps.items():
                _sep.avg_replica(replica=replica)

        for _, _sep in seps.items():
            _sep.avg()

        return seps
