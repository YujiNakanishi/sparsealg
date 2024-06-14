import numpy as np
from sparsealg.core import svector
from typing import Union

"""
--- to_sparse ---
numpy配列をsvectorに変換

input:
    vec -> <np:float64:(dim, )> 配列
output:
    <svector>
"""
def to_sparse(vec : np.ndarray) -> svector:
    indice = np.array([], dtype = np.int64)
    values = np.array([], dtype = np.float64)

    for idx, v in enumerate(vec):
        if v == 0.: continue #ゼロ要素は無視
        indice = np.append(indice, idx)
        values = np.append(values, v)
    
    return svector(len(vec), indice, values)

"""
--- to_dense ---
svectorをnumpyに変換

input:
    svec -> <svec> 配列
output:
    <numpy>
"""
def to_dense(svec : svector) -> np.ndarray:
    vec = np.zeros(svec.dim)
    for index, value in zip(svec.indice, svec.values):
        vec[index] = value
    
    return vec


# svector同士の内積
def dot(svec1 : svector, svec2 : svector) -> float:
    output = 0.
    for i, v in zip(svec1.indice, svec1.values):
        output += v*svec2[i]
    return output


def power(svec : svector, p : Union[int, float]) -> svector:
    return svector(svec.dim, svec.indice, np.power(svec.values, p))

def sum(svec : svector) -> float:
    return np.sum(svec.values)

def abs(svec : svector) -> svector:
    return svector(svec.dim, svec.indice, np.abs(svec.values))

def norm(svec : svector, ord : int = 2) -> float:
    return np.linalg.norm(svec.values, ord)