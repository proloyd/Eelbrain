# Author: Proloy Das <proloy@umd.edu>
# cython: boundscheck=False, wraparound=False
# distutils: language = c++
# setuptools: include_dirs = dsyevh3C/
"""
optimized statistics functions

Relies on code from 'Efficient numerical diagonalization of hermitian 3x3 matrices'
governed by GNU LESSER GENERAL PUBLIC LICENSE Version 2.1.
"""

cimport cython
from libc.math cimport sin, cos
import numpy as np
cimport numpy as cnp

ctypedef cnp.int8_t INT8
ctypedef cnp.int64_t INT64
ctypedef cnp.float64_t FLOAT64


cdef double r_TOL = 2.220446049250313e-16

cdef extern from "dsyevh3.c":
    int dsyevh3(double A[3][3], double Q[3][3], double w[3])

cdef extern from "dsytrd3.c":
    dsytrd3(double A[3][3], double Q[3][3], double d[3], double e[2])

cdef extern from "dsyevq3.c":
    int dsyevq3(double A[3][3], double Q[3][3], double w[3])

cdef extern from "dsyevc3.c":
    int dsyevc3(double A[3][3], double w[3])


def rotation_matrices(cnp.ndarray[FLOAT64, ndim=1] phis,
                      cnp.ndarray[FLOAT64, ndim=1] thetas,
                      cnp.ndarray[FLOAT64, ndim=1] xis,
                      cnp.ndarray[FLOAT64, ndim=3] out):
    cdef unsigned long i, v, case
    cdef FLOAT64 theta, phi, xi
    cdef unsigned long n_cases = phis.shape[0]
    cdef double raxis_x, raxis_y, raxis_z
    cdef double tmp1, tmp2

    cdef double c, s, t

    for case in range(n_cases):
        phi = phis[case]
        theta = thetas[case]
        xi = xis[case]

        raxis_x = sin(phi) * cos(theta)
        raxis_y = sin(phi) * sin(theta)
        raxis_z = cos(phi)
        c = cos(xi)
        s = sin(xi)
        t = 1.0 - c

        out[case, 0, 0] = c + t * raxis_x ** 2
        out[case, 1, 1] = c + t * raxis_y ** 2
        out[case, 2, 2] = c + t * raxis_z ** 2

        tmp1 = raxis_x * raxis_y * t
        tmp2 = raxis_z * s
        out[case, 1, 0] = tmp1 + tmp2
        out[case, 0, 1] = tmp1 - tmp2

        tmp1 = raxis_x * raxis_z * t
        tmp2 = raxis_y * s
        out[case, 2, 0] = tmp1 - tmp2
        out[case, 0, 2] = tmp1 + tmp2

        tmp1 = raxis_y * raxis_z * t
        tmp2 = raxis_x * s
        out[case, 2, 1] = tmp1  + tmp2
        out[case, 1, 2] = tmp1  - tmp2
    return out


@cython.cdivision(True)
def mean_norm_rotated(cnp.ndarray[FLOAT64, ndim=3] y,
                      cnp.ndarray[FLOAT64, ndim=3] rotation,
                      cnp.ndarray[FLOAT64, ndim=1] out):
    cdef unsigned long i, v, case, vi
    cdef double norm, mean

    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long n_dims = y.shape[1]
    cdef unsigned long n_tests = y.shape[2]

    for i in range(n_tests):
        norm = 0
        for v in range(n_dims):
            mean = 0
            for case in range(n_cases):
                for vi in range(n_dims):
                    mean += rotation[case, v, vi] * y[case, vi, i]
            norm += (mean / n_cases) ** 2
        out[i] = norm ** 0.5
    return out


@cython.cdivision(True)
def t2_stat(cnp.ndarray[FLOAT64, ndim=3] y,
            cnp.ndarray[FLOAT64, ndim=1] out):
    cdef unsigned long i, v, u, case
    cdef double norm, temp, max_eig, TOL

    cdef double mean[3]
    cdef double sigma[3][3]

    cdef double eig[3]
    cdef double vec[3][3]

    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long n_dims = y.shape[1]
    cdef unsigned long n_tests = y.shape[2]

    for i in range(n_tests):
        # Initialization
        norm = 0
        for v in range(n_dims):
            mean[v] = 0.0
            for u in range(n_dims):
                sigma[u][v] = 0.0
        # Computation
        for case in range(n_cases):
            for v in range(n_dims):
                temp = y[case, v, i]
                mean[v] += temp
                for u in range(n_dims):
                    sigma[u][v] += temp * y[case, u, i]
        for v in range(n_dims):
            for u in range(n_dims):
                sigma[u][v] -= mean[u] * mean[v] / n_cases
        # check non-zero variance
        for v in range(n_dims):
            if sigma[v][v] != 0:
                break
        else:
            out[i] = 0
            continue

        dsyevh3(sigma, vec, eig)
        max_eig = max(eig, 3)
        TOL = r_TOL * max_eig

        for v in range(n_dims):
            temp = 0
            for u in range(n_dims):
                temp += vec[u][v] * mean[u]
            if temp != 0.0:     # Avoid divide by zero
                if eig[v] > TOL:
                    norm += temp ** 2 / eig[v]
                else:
                    norm += temp ** 2 / TOL
        out[i] = norm ** 0.5
    return out


@cython.cdivision(True)
def t2_stat_rotated(cnp.ndarray[FLOAT64, ndim=3] y,
                    cnp.ndarray[FLOAT64, ndim=3] rotation,
                    cnp.ndarray[FLOAT64, ndim=1] out):
    cdef unsigned long i, v, u, case, vi
    cdef double norm, temp, TOL, max_eig

    cdef double mean[3], tempv[3]
    cdef double sigma[3][3]

    cdef double eig[3]
    cdef double vec[3][3]

    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long n_dims = y.shape[1]
    cdef unsigned long n_tests = y.shape[2]

    for i in range(n_tests):
        # Initialization
        norm = 0
        for v in range(n_dims):
            mean[v] = 0.0
            for u in range(n_dims):
                sigma[u][v] = 0.0
        # Computation
        for case in range(n_cases):
            # rotation
            for u in range(n_dims):
                tempv[u] = 0
                for vi in range(n_dims):
                    tempv[u] += rotation[case, u, vi] * y[case, vi, i]
                mean[u] += tempv[u]
                for v in range(u + 1):      # Only upper triangular part need to be meaningful (See dsyevh.c)
                    sigma[v][u] += tempv[u] * tempv[v]
        for u in range(n_dims):
            for v in range(u + 1):      # Only upper triangular part need to be meaningful (See dsyevh.c)
                sigma[v][u] -= mean[u] * mean[v] / n_cases
        # check non-zero variance
        for v in range(n_dims):
            if sigma[v][v] != 0:
                break
        else:
            out[i] = 0
            continue

        dsyevh3(sigma, vec, eig)
        max_eig = max(eig, 3)
        TOL = r_TOL * max_eig

        for v in range(n_dims):
            temp = 0
            for u in range(n_dims):
                temp += vec[u][v] * mean[u]
            if temp != 0.0:  # Avoid divide by zero
                if eig[v] > TOL:
                    norm += temp ** 2 / eig[v]
                else:
                    norm += temp ** 2 / TOL
        out[i] = norm ** 0.5
    return out


cdef max(double* x, int n):
    cdef int i
    cdef double max_elem = x[0]
    for i in range(1, n):
        if x[i] > max_elem:
            max_elem = x[i]
    return max_elem
