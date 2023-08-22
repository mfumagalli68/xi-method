import abc
from operator import itemgetter

from scipy.stats import rv_histogram

from xi.exceptions import *
from xi.plotting.plot import *
from xi.separation.measurement import *
from xi.utils import *
from sklearn.datasets import load_digits


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
                 type: AnyStr = 'classifier',
                 ties=False,
                 grid: int = 100):
        self.m = {} if m is None else m
        self.obs = {} if obs is None else obs
        self.discrete = [] if discrete is None else discrete
        self.ties = ties
        self.grid = grid
        self.type = type

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

    def _compute_default_m(self,
                           n: int) -> int:
        """
        Compute default partition number
        :param n:
        :return:
        """
        val = np.ceil(np.sqrt(n)).astype('int')

        return val

    def _compute_partitions(self, col: AnyStr, n: int) -> int:

        if isinstance(self.m, int):
            return self.m

        partition = self._compute_default_m(n)

        if col in self.m.keys():
            return self.m.get(col)

        if col in self.discrete:
            partition = n

        if col in self.obs.keys():
            desired_obs = self.m.get(col)
            partition = np.ceil(n / desired_obs).astype('int')

        return partition

    def explain(self,
                X: pd.DataFrame,
                y: np.array,
                replicates: int = 1,
                separation_measure: Union[AnyStr, List] = ['kuiper', 'hellinger']) -> Dict:
        """
        Provide post-hoc explainations

        :param X: Design matrix, without target variable
        :param y: target variable
        :param replicates: number of replications
        :param separation_measure: Separation measurement.
        Read documentation for the implemented ones.
        You can specify one or more than one as list.

        :return: dictionary mapping separation measurement name to object containing explanations
        for each covariates
        """
        if isinstance(separation_measure, str):
            separation_measure = [separation_measure]

        if isinstance(X, pd.DataFrame):
            mapping_col = {k: v for k, v in zip(range(X.shape[1]), X.columns)}
        else:
            mapping_col = {k: str(v) for k, v in zip(range(X.shape[1]), range(X.shape[1]))}

        X = X.values if isinstance(X, pd.DataFrame) else X
        n, k = X.shape

        partition_validation(self.m, k)
        partition_validation(self.obs, k)
        partition_validation(self.discrete, k)

        separation_measurement_validation(measure=separation_measure)

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

        builder_names = itemgetter(*separation_measure)(builder_mapping)
        if isinstance(builder_names, type):
            builder_names = [builder_names]

        seps = {}

        # register builder
        for _sep_measure, builder_name in zip(separation_measure, builder_names):
            factory.register_builder(_sep_measure, builder_name())

        for replica in range(replicates):

            ix = np.argsort(X + np.random.rand(*X.shape), axis=0)

            for idx in range(k):

                col = mapping_col.get(idx)
                partitions = self._compute_partitions(col=col, n=n)

                # builder registered. First iteration
                # builder will create object.
                # Second iteration it won't overwrite since it's already created.
                for _sep_measure, builder_name in zip(separation_measure, builder_names):
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
                            _sep.compute(i=i, j=j, dmass=dmass, condmass=condmass, totalmass=totalmass, type=self.type)

                for _, _sep in seps.items():
                    _sep.avg_replica(replica=replica)

        for _, _sep in seps.items():
            _sep.avg()

        return seps  # {"l1": D, "l2": Q, "Kuiper": M, "linf": B, "KullbackLeibler": K, "Hellinger": H}


class XIRegressor(XI):

    def __init__(self,
                 m: Union[dict, int] = None,
                 grid: int = None,
                 ties=False,
                 type='regressor'):
        super(XIRegressor, self).__init__(m=m, grid=grid, ties=ties, type=type)

    def explain(self,
                X: pd.DataFrame,
                y: np.array,
                replicates: int,
                separation_measure: Union[AnyStr, List]) -> Dict:

        if isinstance(separation_measure, str):
            separation_measure = [separation_measure]

        mapping_col = {k: v for k, v in zip(range(X.shape[1]), X.columns)}

        X = X.values if isinstance(X, pd.DataFrame) else X
        n, k = X.shape

        partition_validation(self.m, k)

        separation_measurement_validation(measure=separation_measure)

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

        builder_names = itemgetter(*separation_measure)(builder_mapping)
        if isinstance(builder_names, type):
            builder_names = [builder_names]

        seps = {}

        # register builder
        for _sep_measure, builder_name in zip(separation_measure, builder_names):
            factory.register_builder(_sep_measure, builder_name())

        for replica in range(replicates):

            ix = np.argsort(X + np.random.rand(*X.shape), axis=0)

            for idx in range(k):
                print(idx)
                col = mapping_col.get(idx)

                # builder registered. First iteration
                # builder will create object.
                # Second iteration it won't overwrite since it's already created.
                for _sep_measure, builder_name in zip(separation_measure, builder_names):
                    seps[_sep_measure] = factory.create(_sep_measure, row=self.m, col=k, replica=replicates)

                indx = np.round(np.linspace(start=0,
                                            stop=n,
                                            num=self.m + 1)).astype('int')

                for i in range(self.m):

                    for j in range(k):
                        z = y[ix[indx[i]:indx[i + 1]]]
                        dmass = rv_histogram(np.histogram(z, bins='auto'))
                        condmass = [dmass.pdf(point) for point in y_grid]
                        condmass = nrmd(condmass)

                        for _, _sep in seps.items():
                            _sep.compute(i=i, j=j, dmass=dmass, condmass=condmass, totalmass=totalmass)

                for _, _sep in seps.items():
                    _sep.avg_replica(replica=replica)

        for _, _sep in seps.items():
            _sep.avg()

        return seps


if __name__ == '__main__':
    # X = np.random.normal(3, 7, size=5 * 100000)  # df_np[:, 1:11]
    # X = X.reshape((100000, 5))
    # reading from the file
    # df = pd.read_csv("/tests/data/winequality-red.csv", sep=";")
    df = load_digits()
    Y = df.target
    df = df.data


    # df.drop(columns='quality', inplace=True)

    # Y = np.array(np.random.randint(2, 4, size=100000))

    # start_time = time.perf_counter()

    # df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

    def compute_attempt(dmass, condmass, totalmass, **kwargs):
        return np.sum(np.sqrt(np.multiply(condmass, totalmass)))


    # cust = CustomSeparationMeasure(separation_measure={'test':compute_attempt})
    # cust.register()

    xi = XIClassifier(m=20)
    P = xi.explain(X=df, y=Y, separation_measure='L1')
    # plot(separation_measurement='L1', type='tabular', explain=P, k=10)
    # P_measures = Ximp(X, Y, None, m=100, ties=False)
    # end_time = time.perf_counter()
    # print(end_time - start_time, "seconds")
    val = P.get('L1').value
    plot(separation_measurement='L1',
         type='image',
         explain=P,
         k=10,
         shape=(8, 8))
    print(val)
