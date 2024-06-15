import numpy as np
from sparsealg.core import svector
from sparsealg import sparse_vector as sv
import sys

def solve(A, b : svector) -> svector:
    Ad = A.to_dense()
    bd = b.to_dense()
    xd = np.linalg.solve(Ad, bd)
    x = svector(len(xd))
    x.as_sparse(xd)

    return x

def Jacob(A, b : svector, x_init : svector = None, max_itr : int = 100, tol : float = 1e-5, reltol : float = 1e-3) -> svector:
    x = svector(b.dim) if x_init is None else x_init.copy()
    b_norm = sv.norm(b)

    A_diag = A.diag()
    D = A - A_diag
    d = svector(x.dim, A_diag.row, A_diag.values)
    for _ in range(max_itr):
        r = b - D@x
        x = r/d

        residual = sv.norm(A@x - b)
        if (residual < tol) and ((residual/b_norm) < reltol): break
    
    return x


def Gauss_Seidel(A, b : svector, x_init : svector = None, max_itr : int = 100, tol : float = 1e-5, reltol : float = 1e-3) -> svector:
    x = svector(b.dim) if x_init is None else x_init.copy()
    b_norm = sv.norm(b)

    for _ in range(max_itr):
        for i in range(x.dim):
            d = x@A.slice(i) - x[i]*A[i,i]
            x[i] = (b[i] - d)/A[i,i]
        
        residual = sv.norm(A@x - b)
        if (residual < tol) and ((residual/b_norm) < reltol): break
    
    return x

def SOR(A, b : svector, x_init : svector = None, alpha : float = 1., max_itr : int = 100, tol : float = 1e-5, reltol : float = 1e-3) -> svector:
    x = svector(b.dim) if x_init is None else x_init.copy()
    b_norm = sv.norm(b)

    for _ in range(max_itr):
        for i in range(x.dim):
            d = x@A.slice(i) - x[i]*A[i,i]
            y_i = (b[i] - d)/A[i,i]
            x[i] += alpha*(y_i - x[i])
        
        residual = sv.norm(A@x - b)
        if (residual < tol) and ((residual/b_norm) < reltol): break
    
    return x

def CG(A, b : svector, x_init : svector = None, max_itr : int = 100, tol : float = 1e-5, reltol : float = 1e-3) -> svector:
    x = svector(b.dim) if x_init is None else x_init.copy()
    b_norm = sv.norm(b)

    r = b - A@x
    p = r.copy()

    for _ in range(max_itr):
        alpha = (r@p) / (p@(A@p))
        x += alpha*p

        residual = sv.norm(A@x - b)
        if (residual < tol) and ((residual/b_norm) < reltol): break
        
        r -= alpha*(A@p)
        beta = -(r@(A@p)) / (p@(A@p))
        p = r + beta*p
    
    return x


def BiCG(A, b : svector, x_init : svector = None, max_itr : int = 100, tol : float = 1e-5, reltol : float = 1e-3) -> svector:
    x = svector(b.dim) if x_init is None else x_init.copy()
    b_norm = sv.norm(b)

    r = b - A@x
    p = r.copy()

    rb = sv.rand_uniform(b.dim)
    pb = rb.copy()

    for _ in range(max_itr):
        alpha = (rb@r) / (pb@(A@p))
        x += alpha*p

        residual = sv.norm(A@x - b)
        if (residual < tol) and ((residual/b_norm) < reltol): break
        
        r_before = r.copy()
        rb_before = rb.copy()
        r -= alpha*(A@p)
        rb -= alpha*((A.T)@pb)
        
        beta = (rb@r) / (r_before@rb_before)
        p = r + beta*p
        pb = rb + beta*pb
    
    return x


def BiCGStab(A, b : svector, x_init : svector = None, max_itr : int = 100, tol : float = 1e-5, reltol : float = 1e-3) -> svector:
    x = svector(b.dim) if x_init is None else x_init.copy()
    b_norm = sv.norm(b)

    r = b - A@x
    p = r.copy()

    rb = sv.rand_uniform(b.dim)

    for _ in range(max_itr):
        alpha = (rb@r) / (rb@(A@p))
        t = r - alpha*(A@p)
        w = (t@(A@t)) / sv.norm(A@t)
        x += alpha*p + w*t

        residual = sv.norm(A@x - b)
        print(residual)
        if (residual < tol) and ((residual/b_norm) < reltol): break

        r_before = r.copy()
        r = t - w*(A@t)

        beta = (alpha/w)*(rb@r)/(rb@r_before)
        p = r + beta*(p - w*(A@p))
    
    return x