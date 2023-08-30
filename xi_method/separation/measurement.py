import numpy as np
from scipy.special import rel_entr


def nrmd(x):  # maybe this can be rewritten as a lambda function
    if np.sum(x) == 0:
        total = 1
    else:
        total = np.sum(x)
    return np.divide(x, total)


class SepMeasureFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


class SeparationMeasurement:

    def __init__(self, row, col, replica, idx_to_col):
        self._matrix = np.zeros((row, col))
        self._matrix_replica = np.zeros((col, replica))
        self.explanation = 0
        self.idx_to_col = idx_to_col

    def compute(self, i, j, **kwargs):

        type = kwargs.get('type')
        if type == 'regressor':
            self._matrix[i, j] = self._regressor(**kwargs)
        else:
            self._matrix[i, j] = self._compute(**kwargs)

    def _regressor(self, **kwargs):
        pass

    def _compute(self, **kwargs):
        pass

    def avg_replica(self, replica):
        self._matrix_replica[:, replica] = np.mean(self._matrix, axis=0)

    def avg(self):
        self.explanation = np.mean(self._matrix_replica, axis=1)

    def reset(self, row, col):
        self._matrix = np.zeros((row, col))


class L1Service(SeparationMeasurement):

    def __init__(self, row, col, replica, idx_to_col):
        super(L1Service, self).__init__(row, col, replica, idx_to_col)

    def _compute(self, dmass, **kwargs):
        return np.sum(np.abs(dmass))

    def _regressor(self, dmass, condmass, totalmass):
        return np.sum(np.abs(totalmass - condmass))


class L1Builder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica, idx_to_col):
        if not self._instance:
            self._instance = L1Service(row=row, col=col, replica=replica, idx_to_col=idx_to_col)
        else:
            self._instance.reset(row=row, col=col)

        return self._instance


class L2Service(SeparationMeasurement):

    def __init__(self, row, col, replica, idx_to_col):
        super(L2Service, self).__init__(row, col, replica, idx_to_col)

    def _compute(self, dmass, **kwargs):
        return np.sum(np.square(dmass))

    def _regressor(self, totalmass, condmass, **ignored):
        return np.sqrt(np.sum(np.square(totalmass - condmass)))


class L2Builder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica, idx_to_col):
        if not self._instance:
            self._instance = L2Service(row=row, col=col, replica=replica, idx_to_col=idx_to_col)
        else:
            self._instance.reset(row=row, col=col)
        return self._instance


class KLService(SeparationMeasurement):

    def __init__(self, row, col, replica, idx_to_col):
        super(KLService, self).__init__(row, col, replica, idx_to_col)

    def _compute(self, dmass,condmass,totalmass, **ignored):

        kl = np.multiply(condmass, np.log(np.divide(condmass, totalmass)))
        kl[np.isnan(kl)] = 0
        return np.sum(kl)

    def _regressor(self, totalmass, condmass, dmass, **ignored):
        return np.sum(rel_entr(totalmass, condmass))


class KLBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica, idx_to_col):
        if not self._instance:
            self._instance = KLService(row=row, col=col, replica=replica, idx_to_col=idx_to_col)
        else:
            self._instance.reset(row=row, col=col)
        return self._instance


class KuiperBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica, idx_to_col):
        if not self._instance:
            self._instance = KuiperService(row=row, col=col, replica=replica, idx_to_col=idx_to_col)
        else:
            self._instance.reset(row=row, col=col)
        return self._instance


class KuiperService(SeparationMeasurement):

    def __init__(self, row, col, replica, idx_to_col):
        super(KuiperService, self).__init__(row, col, replica, idx_to_col)

    def _compute(self, dmass, **ignored):
        return np.max(np.abs(dmass))

    def _regressor(self, dmass, condmass, totalmass, **ignored):
        val = max(totalmass - condmass) - min(totalmass - condmass)
        return val


class HellingerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica, idx_to_col):
        if not self._instance:
            self._instance = HellingerService(row=row, col=col, replica=replica, idx_to_col=idx_to_col)
        else:
            self._instance.reset(row=row, col=col)
        return self._instance


class HellingerService(SeparationMeasurement):

    def __init__(self, row, col, replica, idx_to_col):
        super(HellingerService, self).__init__(row, col, replica, idx_to_col)

    def _compute(self, condmass, totalmass, **ignored):
        return 1 - np.sum(np.sqrt(np.multiply(condmass, totalmass)))

    def _regressor(self, condmass, totalmass, **ignored):
        return 1 - np.sum(np.sqrt(np.multiply(condmass, totalmass)))


# class CustomSeparationMeasureBuilder:
#     def __init__(self):
#         self._instance = None
#
#     def __call__(self, row, col, replica):
#         if not self._instance:
#             self._instance = CustomSeparationMeasureService(row=row, col=col, replica=replica)
#         return self._instance
#
#
# class CustomSeparationMeasureService(Separation):
#
#     def __init__(self, row, col, replica):
#         super(CustomSeparationMeasureService, self).__init__(row, col, replica)
#
#     def compute(self, i, j, **kwargs):
#         condmass = kwargs.get('condmass', 0)
#         totalmass = kwargs.get('totalmass', 0)
#         dmass = kwargs.get('dmass', 0)
#
#         self.matrix[i, j] = self.compute_custom(**kwargs)
#
#     @staticmethod
#     def compute_custom(**kwargs):
#         pass
#
#
# class CustomSeparationMeasure:
#
#     def __init__(self, separation_measure: Dict):  # dict name -> callable
#         self.separation = separation_measure
#         self.custom = None
#
#     def register(self):
#         _, fun = list(self.separation.items())[0]
#         CustomSeparationMeasureService.compute_custom = fun
#         self._update_builder_mapping()
#
#     def _update_builder_mapping(self):
#         key, val = list(self.separation.items())[0]
#         builder_mapping.update({key: CustomSeparationMeasureBuilder})


builder_mapping = {'Kuiper': KuiperBuilder,
                   'Hellinger': HellingerBuilder,
                   'Kullback-leibler': KLBuilder,
                   'L1': L1Builder,
                   'L2': L2Builder}

# if __name__ == '__main__':
#     factory = SepMeasureFactory()
#     factory.register_builder('KUIPER', KuiperBuilder())
#     factory.create('KUIPER', row=100, col=20)
#     print(factory)
