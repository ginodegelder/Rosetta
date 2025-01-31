# -*- coding: utf-8 -*-

try:
    import numba as nb
    # jit utils functions
    # Define the jit compiling options
    # Remove and put an empty decorator for manual tests

    # jitmode = nb.jit(nopython=True, cache=True)
    # jitmode = nb.jit(nopython=True, nogil=True)
    jitmode = nb.jit(nopython=True)
    # def jitmode(func):
    #     return func
except (ImportError, NameError):
    print("Numba not available")

    def jitmode(func):
        return func

