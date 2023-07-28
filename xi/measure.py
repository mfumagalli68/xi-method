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


class KuiperBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col,replica):
        if not self._instance:
            self._instance = KuiperService(row=row, col=col,replica=replica)
        return self._instance


class KuiperService:

    def __init__(self, row, col, replica):
        self.matrix = np.zeros((row, col))
        self.matrix_replica = np.zeros((col, replica))
        self.value = 0

    def compute(self,i,j, dmass, **ignored):
        self.matrix[i, j] = np.max(np.abs(dmass))


    def avg_replica(self,replica):
        self.matrix_replica[:, replica] = np.mean(self.matrix, axis=0)

    def avg(self):
        self.value= np.mean(self.matrix_replica, axis=1)

    def reset(self,row,col):
        self.matrix = np.zeros((row, col))

class HellingerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, row, col, replica):
        if not self._instance:
            self._instance = HellingerService(row=row, col=col, replica=replica)
        return self._instance


class HellingerService:

    def __init__(self, row, col, replica):
        self.matrix = np.zeros((row, col))
        self.matrix_replica = np.zeros((col, replica))
        self.value = 0

    def compute(self, i, j, condmass, totalmass, **ignored):
        self.matrix[i, j] = 1 - np.sum(np.sqrt(np.multiply(condmass, totalmass)))

    def avg_replica(self,replica):
        self.matrix_replica[:, replica] = np.mean(self.matrix, axis=0)

    def avg(self):
        self.value = np.mean(self.matrix_replica, axis=1)


    def reset(self,row,col):
        self.matrix = np.zeros((row, col))

builder_mapping = {'kuiper': KuiperBuilder,'hellinger': HellingerBuilder}

if __name__ == '__main__':
    factory = SepMeasureFactory()
    factory.register_builder('KUIPER', KuiperBuilder())
    factory.create('KUIPER', row=100, col=20)
    print(factory)
