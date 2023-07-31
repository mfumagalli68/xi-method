from typing import *
import numpy as np
import inspect


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

    def __init__(self, row, col, replica):
        self.matrix = np.zeros((row, col))
        self.matrix_replica = np.zeros((col, replica))
        self.value = 0

    def avg_replica(self, replica):
        self.matrix_replica[:, replica] = np.mean(self.matrix, axis=0)

    def avg(self):
        self.value = np.mean(self.matrix_replica, axis=1)

    def reset(self, row, col):
        self.matrix = np.zeros((row, col))


class L1Service(SeparationMeasurement):

    def __init__(self, row, col, replica):
        super(L1Service, self).__init__(row, col, replica)

    def compute(self, i, j, dmass, **kwargs):
        self.matrix[i, j] = self._compute(dmass=dmass)

    def _compute(self, dmass):
        return np.sum(np.abs(dmass))


class L1Builder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = L1Service(row=row, col=col, replica=replica)
        return self._instance


class L2Service(SeparationMeasurement):

    def __init__(self, row, col, replica):
        super(L2Service, self).__init__(row, col, replica)

    def compute(self, i, j, dmass, **kwargs):
        self.matrix[i, j] = self._compute(dmass=dmass)

    def _compute(self, dmass):
        return np.sum(np.square(dmass))


class L2Builder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = L2Service(row=row, col=col, replica=replica)
        return self._instance


class KLService(SeparationMeasurement):

    def __init__(self, row, col, replica):
        super(KLService, self).__init__(row, col, replica)

    def compute(self, i, j, dmass, **kwargs):
        self.matrix[i, j] = self._compute(dmass=dmass, **kwargs)

    def _compute(self, dmass, **kwargs):
        condmass = kwargs.get('condmass')
        totalmass = kwargs.get('totalmass')
        kl = np.multiply(condmass, np.log(np.divide(condmass, totalmass)))
        kl[np.isnan(kl)] = 0
        return np.sum(kl)


class KLBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = KLService(row=row, col=col, replica=replica)
        return self._instance


class KuiperBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = KuiperService(row=row, col=col, replica=replica)
        return self._instance


class KuiperService(SeparationMeasurement):

    def __init__(self, row, col, replica):
        super(KuiperService, self).__init__(row, col, replica)

    def compute(self, i, j, dmass, **ignored):
        self.matrix[i, j] = np.max(np.abs(dmass))

    def _compute(self, dmass, **ignored):
        return np.max(np.abs(dmass))


class HellingerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = HellingerService(row=row, col=col, replica=replica)
        return self._instance


class HellingerService(SeparationMeasurement):

    def __init__(self, row, col, replica):
        super(HellingerService, self).__init__(row, col, replica)

    def compute(self, i, j, condmass, totalmass, **ignored):
        self.matrix[i, j] = self._compute(condmass, totalmass)

    def _compute(self, condmass, totalmass, **ignored):
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

if __name__ == '__main__':
    factory = SepMeasureFactory()
    factory.register_builder('KUIPER', KuiperBuilder())
    factory.create('KUIPER', row=100, col=20)
    print(factory)
