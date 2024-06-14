import numpy as np

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
    
    # svector[i]にvを代入
    def set(self, i : int, v : float):
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
    
    def add_at(self, i : int, v : float):
        index = self.where(i)

        if index is None:
            self.set(i, v)
        else:
            if self.values[index] == -v:
                self.set(i, 0)
            else:
                self.values[index] += v
    
    def sub_at(self, i : int, v : float):
        self.add_at(i, -v)
    
    def mul_at(self, i : int, v : float):
        if v == 0.:
            self.set(i, 0.)
        else:
            index = self.where(i)
            if index is not None:
                self.values[index] *= v
    
    def div_at(self, i : int, v : float):
        self.mul_at(i, 1./v)
    

    def __neg__(self):
        return type(self)(self.dim, self.indice, -self.values)
    
    def __add__(self, another):
        output = self.copy()

        if type(another) != type(self):
            i = np.arange(self.dim, dtype = np.int64)
            v = another*np.ones(self.dim, dtype = np.float64)
            another = type(self)(self.dim, i, v)

        for i, v in zip(another.indice, another.values):
            output.add_at(i, v)
        
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
                    output.set(index, 0.)
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