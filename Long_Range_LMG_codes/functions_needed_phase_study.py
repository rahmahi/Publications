from qutip import mesolve, basis, jmat
import numpy as np
import scipy.linalg as la
from numpy import angle, pi
from qutip import Qobj, propagator, floquet_modes
import time as time
import matplotlib.pyplot as plt

def floquet_modes_mod(H, T, args=None, parallel=False, sort=False, U=None):
    if 'opts' in args:
        options = args['opts']
    else:
        options = Options()
        options.rhs_reuse = True
        rhs_clear() 
    
    if U is None:
        U = propagator(H, T, [], args, parallel=parallel, progressbar=True, options=options)
    
    evals, evecs = la.eig(U.full())

    eargs = angle(evals)
    eargs += (eargs <= -pi) * (2 * pi) + (eargs > pi) * (-2 * pi)
    e_quasi = -eargs / T

    if sort:
        order = np.argsort(-e_quasi)
    else:
        order = list(range(len(evals)))

    new_dims = [U.dims[0], [1] * len(U.dims[0])]
    new_shape = [U.shape[0], 1]
    kets_order = [Qobj(np.matrix(evecs[:, o]).T,
                       dims=new_dims, shape=new_shape) for o in order]

    return kets_order, e_quasi[order]

def floquet_modes_phtr(H, T, args=None, parallel=False, sort=False, U=None):
    if 'opts' in args:
        options = args['opts']
    else:
        options = Options()
        options.rhs_reuse = True
        rhs_clear() 
    
    if U is None:
        U = propagator(H, T, [], args, options=options)
    
    evals, evecs = la.eig(U.full())

    eargs = angle(evals)
    eargs += (eargs <= -pi) * (2 * pi) + (eargs > pi) * (-2 * pi)
    e_quasi = -eargs / T

    if sort:
        order = np.argsort(-e_quasi)
    else:
        order = list(range(len(evals)))

    new_dims = [U.dims[0], [1] * len(U.dims[0])]
    new_shape = [U.shape[0], 1]
    kets_order = [Qobj(np.matrix(evecs[:, o]).T,
                       dims=new_dims, shape=new_shape) for o in order]

    return kets_order, e_quasi[order]


def drive_exact(t, args):
    w = args['omega']
    h = args['h']
    h0 = args['h0']
    return h * np.cos(w*t) + h0

def get_hamiltonians_exact(args):
    N = args['N']
    sx,sy,sz = jmat(N,"x"),jmat(N,"y"),jmat(N,"z")
    kn =  2.0/(N-1)                                      # kacNorm
    H0 = kn * sz **2
    H1 = 2 * sx
    return H0,H1

def floq_evolv_exact(args):
    T = 2 * np.pi/args['omega']
    H0, H1 = get_hamiltonians_exact(args)
    H = [H0,[H1,drive_exact]]
    f_states, _ = floquet_modes_mod(H, T, args=args)
    return f_states

def floq_evolv_exact_sten(args):
    w = args['omega']
    T = 2 * np.pi/w
    H0, H1 = get_hamiltonians_exact(args)
    H = [H0,[H1,drive_exact]]
    f_modes_0, f_energies = floquet_modes_phtr(H, T, args=args)
    return [w, f_modes_0, f_energies]