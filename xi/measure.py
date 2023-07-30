from typing import *
import numpy as np


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


class Measurement:

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


class L1Service(Measurement):

    def __init__(self, row, col, replica):
        super(L1Service,self).__init__(row, col, replica)

    def compute(self, i, j, dmass, **kwargs):
        self.matrix[i, j] = np.sum(np.abs(dmass))


class L1Builder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = L1Service(row=row, col=col, replica=replica)
        return self._instance


class L2Service(Measurement):


    def __init__(self, row, col, replica):
        super(L2Service,self).__init__(row, col, replica)

    def compute(self, i, j, dmass, **kwargs):
        self.matrix[i, j] = np.sum(np.square(dmass))


class L2Builder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = L2Service(row=row, col=col, replica=replica)
        return self._instance


class KLService(Measurement):

    def __init__(self, row, col, replica):
        super(KLService,self).__init__(row, col, replica)

    def compute(self, i, j, dmass, **kwargs):
        condmass = kwargs.get('condmass')
        totalmass = kwargs.get('totalmass')
        kl = np.multiply(condmass, np.log(np.divide(condmass, totalmass)))
        kl[np.isnan(kl)] = 0
        self.matrix[i, j] = np.sum(kl)


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


class KuiperService(Measurement):

    def __init__(self, row, col, replica):
        super(KuiperService,self).__init__(row, col, replica)

    def compute(self, i, j, dmass, **ignored):
        self.matrix[i, j] = np.max(np.abs(dmass))


class HellingerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = HellingerService(row=row, col=col, replica=replica)
        return self._instance


class HellingerService(Measurement):

    def __init__(self, row, col, replica):
        super(HellingerService,self).__init__(row, col, replica)

    def compute(self, i, j, condmass, totalmass, **ignored):
        self.matrix[i, j] = 1 - np.sum(np.sqrt(np.multiply(condmass, totalmass)))


builder_mapping = {'kuiper': KuiperBuilder, 'hellinger': HellingerBuilder,
                   'Kullback-leibler': KLBuilder, 'l1': L1Builder,
                   'L2': L2Builder}

if __name__ == '__main__':
    factory = SepMeasureFactory()
    factory.register_builder('KUIPER', KuiperBuilder())
    factory.create('KUIPER', row=100, col=20)
    print(factory)
