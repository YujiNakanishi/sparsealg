import numpy as np
import sys


"""
---svector---
非ゼロ要素の数値のみ保存されたベクトルのクラス

att:
    dim -> <int> 次元数
    indice -> <np:int64:(n, )> 非ゼロ要素のインデックス番号。nは非ゼロ要素数
    values -> <np:float64:(n, )> 非ゼロ要素の値
"""
class svector:
    def __init__(self, dim : int, indice : np.ndarray = None, values : np.ndarray = None):
        self.dim = dim
        self.indice = np.array([], dtype = np.int64) if indice is None else np.copy(indice)
        self.values = np.array([], dtype = np.float64) if values is None else np.copy(values)
    
    
    #ベクトルの次元数を返す
    def __len__(self) -> int:
        return self.dim
    
    #非ゼロ要素数を返す
    @property
    def num_nonzero(self) -> int:
        return len(self.indice)
    
    def copy(self):
        return type(self)(self.dim, self.indice, self.values)
    
    """
    --- where ---
    i番目の要素(いわゆるsvector[i])がindiceの何番目に定義されているか探査

    input:
        i -> <int> インデックス番号
    output:
        index -> <int> indiceにおけるインデックス番号
    
    example:
        >svec.indice
        [2, 0, 1]
        >svec.where(0)
        1
    """
    def where(self, i : int) -> int:
        assert i < self.dim #インデックス番号iが次元数以下か判定
        index = np.where(self.indice == i)[0]

        if len(index) > 1: raise Exception("Found multi defined elements")
        return None if len(index) == 0 else index[0]
    
    def __setitem__(self, i : int, v : float):
        index = self.where(i)
        if v == 0.:
            if index is not None:
                self.indice = np.delete(self.indice, index)
                self.values = np.delete(self.values, index)
        else:
            if index is None:
                self.indice = np.append(self.indice, i)
                self.values = np.append(self.values, v)
            else:
                self.values[index] = v
    

    def __getitem__(self, i : int) -> float:
        index = self.where(i)
        return 0. if index is None else self.values[index]
    

    def __neg__(self):
        return type(self)(self.dim, self.indice, -self.values)
    
    def __add__(self, another):
        output = self.copy()

        if type(another) != type(self):
            i = np.arange(self.dim, dtype = np.int64)
            v = another*np.ones(self.dim, dtype = np.float64)
            another = type(self)(self.dim, i, v)

        for i, v in zip(another.indice, another.values):
            output[i] += v
        
        return output
    
    def __radd__(self, another):
        return self.__add__(another)

    def __sub__(self, another):
        return self.__add__(-another)
    
    def __rsub__(self, another):
        return (-self).__add__(another)
    
    def __mul__(self, another):
        if type(another) == type(self):
            output = self.copy()
            for i, index in enumerate(output.indice):
                index_another = another.where(index)

                if index_another is None:
                    output[index] = 0.
                else:
                    output.values[i] *= another.values[index_another]
            return output
        else:
            if another == 0.:
                return type(self)(self.dim)
            else:
                output = self.copy()
                output.values *= another
                return output
    
    def __rmul__(self, another):
        return self.__mul__(another)
    
    def __truediv__(self, another):

        if type(another) == type(self):
            vec1 = self.to_dense()
            vec2 = another.to_dense()
            output = type(self)(self.dim)
            output.as_sparse(vec1/vec2)

            return output
        else:
            output = self.copy()
            output.values /= another

            return output
    
    def __rtruediv__(self, another):
        if type(another) != type(self):
            another = type(self)(self.dim, np.arange(self.dim, dtype = np.int64), another*np.ones(self.dim, dtype = np.float64))
        
        return another.__truediv__(self)
    
    def __matmul__(self, another) -> float:
        output = 0.
        for i, v in zip(self.indice, self.values):
            output += v*another[i]
        return float(output)
    
    def to_dense(self) -> np.ndarray:
        vec = np.zeros(self.dim)
        for index, value in zip(self.indice, self.values):
            vec[index] = value
        
        return vec
    
    def as_sparse(self, vec : np.ndarray):
        assert len(vec) == self.dim
        self.indice = np.array([], dtype = np.int64)
        self.values = np.array([], dtype = np.float64)

        for idx, v in enumerate(vec):
            if v == 0.: continue #ゼロ要素は無視
            self.indice = np.append(self.indice, idx)
            self.values = np.append(self.values, v)
    

"""
--- smatrix ---
祖行列のクラス
"""
class smatrix:
    def __init__(self, M : int, N : int):
        self.shape = (M, N)
    
    def __len__(self) -> int:
        return self.shape[0]
    
    @property
    def num_nonzero(self) -> int:
        raise NotImplementedError
    
    def check_index(self, i : int, j : int) -> bool:
        return (i < self.shape[0]) and (j < self.shape[1])
    
    def copy(self):
        raise NotImplementedError
    
    def to_dense(self) -> np.ndarray:
        raise NotImplementedError
    
    def as_sparse(self, matrix : np.ndarray):
        raise NotImplementedError
    
    def __setitem__(self, key : tuple, v : float):
        raise NotImplementedError
    
    def slice(self, k : int, axis : int = 0) -> svector:
        raise NotImplementedError
    
    def __getitem__(self, key : tuple) -> float:
        raise NotImplementedError
    
    def __add__(self, another):
        raise NotImplementedError
    
    def __radd__(self, another):
        return self.__add__(another)
    
    def __neg__(self):
        raise NotImplementedError
    
    def __sub__(self, another):
        return self.__add__(-another)
    
    def __rsub__(self, another):
        return (-self).__add__(another)
    
    def __mul__(self, another):
        raise NotImplementedError
    
    def __rmul__(self, another):
        return self.__mul__(another)
    
    def __matmul__(self, another):
        if type(another) == type(self):
            M = self.shape[0]; N = another.shape[1]
            output = type(self)(M, N)

            for i in range(M):
                svec1 = self.slice(i, axis = 0)
                for j in range(N):
                    svec2 = another.slice(j, axis = 1)
                    
                    output[i,j] = svec1@svec2
            
            return output
        
        else:
            output = svector(self.shape[0])
            for i in range(len(another)):
                slice_vec = self.slice(i)

                output[i] = slice_vec@another
            
            return output
        
    
    def diag(self):
        raise NotImplementedError
    def tril(self, k : int = 0):
        raise NotImplementedError
    def triu(self, k : int = 0):
        raise NotImplementedError
    
    @property
    def T(self):
        raise NotImplementedError