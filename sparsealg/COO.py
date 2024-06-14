import numpy as np
from sparsealg.core import smatrix, svector

class COO_matrix(smatrix):
    def __init__(self, M : int, N : int, row : np.ndarray = None, column : np.ndarray = None, values : np.ndarray = None):
        super().__init__(M, N)
        self.row = np.array([], dtype = np.int64) if row is None else np.copy(row)
        self.column = np.array([], dtype = np.int64) if column is None else np.copy(column)
        self.values = np.array([], dtype = np.int64) if values is None else np.copy(values)

    def copy(self):
        return type(self)(self.shape[0], self.shape[1], self.row, self.column, self.values)
    
    @property
    def num_nonzero(self):
        return len(self.row)
    
    def where(self, i : int, j : int) -> int:
        assert self.check_index(i, j)
        index = np.where((self.row == i)*(self.column == j))[0]
        if len(index) > 1: raise Exception("Multi defined elements are found")

        return None if len(index) == 0 else index[0]
    
    def __setitem__(self, key : tuple, v : float):
        i, j = key
        index = self.where(i, j)
        if v == 0.:
            if index is not None:
                self.row = np.delete(self.row, index)
                self.column = np.delete(self.column, index)
                self.values = np.delete(self.values, index)
        else:
            if index is None:
                self.row = np.append(self.row, i)
                self.column = np.append(self.column, j)
                self.values = np.append(self.values, v)
            else:
                self.values[index] = v
    
    def __getitem__(self, key : tuple) -> float:
        i, j = key
        index = self.where(i, j)
        return 0. if index is None else self.values[index]
    
    def slice(self, k : int, axis : int = 0) -> svector:
        assert axis <= 1
        max_size = self.shape[0] if axis == 0 else self.shape[1]
        assert k < max_size

        mask = (self.row == k) if axis == 0 else (self.column == k)
        indice = self.column[mask] if axis == 0 else self.row[mask]
        values = self.values[mask]

        dim = self.shape[1] if axis == 0 else self.shape[0]
        return svector(dim, indice, values)
    
    def __add__(self, another):
        output = self.copy()
        if type(another) != type(self):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    output[i,j] += another
        else:
            for i, j, v in zip(another.row, another.column, another.values):
                output[i,j] += v
        
        return output
    
    def __neg__(self):
        return type(self)(self.shape[0], self.shape[1], self.row, self.column, -self.values)
    
    def __mul__(self, another):
        output = self.copy()

        if type(another) != type(self):
            output.values *= another
        else:
            for i, j in zip(self.row, self.column):
                output[i, j] *= another[i, j]
        
        return output



def to_dense(coo : COO_matrix) -> np.ndarray:
    matrix = np.zeros(coo.shape)
    for i, j, v in zip(coo.row, coo.column, coo.values):
        matrix[i,j] = v
    return matrix

def to_sparse(matrix : np.ndarray) -> COO_matrix:
    shape = matrix.shape
    row = np.array([], dtype = np.int64)
    column = np.array([], dtype = np.int64)
    values = np.array([], dtype = np.float64)

    for i in range(shape[0]):
        for j in range(shape[1]):
            v = matrix[i, j]
            if v != 0.:
                row = np.append(row, i)
                column = np.append(column, j)
                values = np.append(values, v)
    
    return COO_matrix(shape[0], shape[1], row, column, values)