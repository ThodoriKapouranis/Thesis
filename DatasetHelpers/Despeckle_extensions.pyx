import cython
cimport cython
import numpy as np

ctypedef fused fltcpl_t:
    cython.float
    cython.double
    cython.floatcomplex
    cython.doublecomplex

ctypedef cython.doublecomplex DTYPE_t

cdef extern from "math.h":
    double sqrt(double)
    double sin(double)
    double cos(double)
    double atan2(double, double)
    double fabs(double)
    double log(double)
    double exp(double)
cdef extern from "complex.h":
    double creal(double complex)
    double cimag(double complex)
    double complex conj(double complex)
    double complex clog(double complex)

# https://github.com/birgander2/PyRAT/blob/b4236269c329d6941d01d1cd9a8689776650ff4e/pyrat/filter/Despeckle_extensions.pyx#L706
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_leesigma(float [:, :] span, fltcpl_t [:, :, :, :] array, float looks=1.0, win=(7,7)):
    cdef fltcpl_t [:, :, :, :] out = np.empty_like(array)
    cdef float diff = 2.0 * np.sqrt(1.0/looks)
    cdef int ny = array.shape[2]
    cdef int nx = array.shape[3]
    cdef int nv = array.shape[0]
    cdef int nz = array.shape[1]
    cdef int ym = win[0]/2
    cdef int xm = win[1]/2
    cdef int limit = (xm + ym)  # //4
    cdef int k, l, x, y, v, z
    if cython.float is fltcpl_t:
        foo = np.zeros((nv, nz), dtype='float32')
    elif cython.floatcomplex is fltcpl_t:
        foo = np.zeros((nv, nz), dtype='complex64')
    elif cython.double is fltcpl_t:
        foo = np.zeros((nv, nz), dtype='float64')
    elif cython.doublecomplex is fltcpl_t:
        foo = np.zeros((nv, nz), dtype='complex128')
    cdef fltcpl_t [:, :] res = foo

    cdef int n = 0
    for k in range(ym, ny-ym):
        for l in range(xm, nx-xm):
            for v in range(nv):
                for z in range(nz):
                    res[v, z] = 0.0
            n = 0
            for y in range(-ym, ym+1):
                for x in range(-xm, xm+1):
                    if span[k+y, l+x]>span[k, l]*(1.0-diff) and (span[k+y, l+x]<span[k, l]*(1.0+diff)):
                        for v in range(nv):
                            for z in range(nz):
                                res[v, z] = res[v, z] + array[v, z, k+y, l+x]
                        n += 1
            if n >= limit:
                for v in range(nv):
                    for z in range(nz):
                        out[v, z, k, l] = res[v, z] / n
            else:
               for v in range(nv):
                    for z in range(nz):
                        out[v, z, k, l] = (array[v, z, k-1, l] + array[v, z, k+1, l] + array[v, z, k, l-1] + array[v, z, k, l+1]) / 4.0
    return np.asarray(out)
    
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_leeimproved(float [:, :] span, fltcpl_t [:, :, :, :] array, bounds=(0.5, 3.0), float thres=5.0, looks=1.0, win=(9, 9), float newsig=0.5):
    cdef fltcpl_t [:, :, :, :] out = np.zeros_like(array)
    cdef float sig2 = 1.0 / looks
    cdef float sfak = 1.0 + sig2
    cdef float nsig2 = newsig
    cdef float nsfak = 1.0 + nsig2
    cdef float xtilde
    cdef int nv = array.shape[0]
    cdef int nz = array.shape[1]
    cdef int ny = array.shape[2]
    cdef int nx = array.shape[3]
    cdef int ym = win[0]/2
    cdef int xm = win[1]/2
    cdef int norm = win[0] * win[1]
    cdef int k, l, x, y, v, z
    cdef float m2arr, marr, vary, varx, kfac, i1, i2
    if cython.float is fltcpl_t:
        foo = np.zeros((nv, nz), dtype='float32')
    elif cython.floatcomplex is fltcpl_t:
        foo = np.zeros((nv, nz), dtype='complex64')
    elif cython.double is fltcpl_t:
        foo = np.zeros((nv, nz), dtype='float64')
    elif cython.doublecomplex is fltcpl_t:
        foo = np.zeros((nv, nz), dtype='complex128')
    cdef fltcpl_t [:, :] res = foo

    cdef int n = 0
    for k in range(ym, ny-ym):
        for l in range(xm, nx-xm):
            m2arr = 0.0
            marr = 0.0
            n = 0
            for y in range(-1, 2):                          # check 3x3 neighbourhood
                for x in range(-1, 2):
                    m2arr += span[k+y, l+x]**2
                    marr += span[k+y, l+x]
                    if span[k+y, l+x] > thres:
                        n += 1
            if n >= 6:                                      # keep all point targets
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        for v in range(nv):
                            for z in range(nz):
                                if span[k+y, l+x] > thres:
                                    out[v, z, k+y, l+x] = array[v, z, k+y, l+x]

            if out[0, 0, k, l] == 0.0:                      # no point target, also not prior
                m2arr /= 9.0
                marr /= 9.0
                vary = (m2arr - marr**2)
                if vary < 1e-10: vary = 1e-10
                varx = ((vary - marr ** 2 * sig2) / sfak)
                if varx < 0: varx = 0
                kfac = varx / vary

                xtilde = (span[k, l] - marr) * kfac + marr

                i1 = xtilde*bounds[0]
                i2 = xtilde*bounds[1]
                m2arr = 0.0
                marr = 0.0
                n = 0
                for v in range(nv):
                    for z in range(nz):
                        res[v, z] = 0.0

                for y in range(-ym, ym+1):
                    for x in range(-xm, xm+1):
                        if span[k+y, l+x]>i1 and span[k+y, l+x]<i2:
                            m2arr += span[k+y, l+x]**2
                            marr += span[k+y, l+x]
                            n += 1
                            for v in range(nv):
                                for z in range(nz):
                                    res[v, z] = res[v, z] + array[v, z, k+y, l+x]
                if n == 0:
                    for v in range(nv):
                        for z in range(nz):
                            out[v, z, k, l] = 0.0
                else:
                    m2arr /= n
                    marr /= n
                    vary = (m2arr - marr**2)
                    if vary < 1e-10: vary = 1e-10
                    varx = ((vary - marr ** 2 * nsig2) / nsfak)
                    if varx < 0.0: varx = 0.0
                    kfac = varx / vary
                    for v in range(nv):
                        for z in range(nz):
                            out[v, z, k, l] = (array[v, z, k, l] - res[v, z] / n) * kfac + res[v, z] / n
    return np.asarray(out)