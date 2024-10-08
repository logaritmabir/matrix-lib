import ctypes
import time
import numpy as np
import pyblas.level1 as pb1
import scipy
import scipy.linalg

def saxpy(alpha, x, y):
    """
    Perform the operation dst = alpha * x + y.

    Parameters:
    alpha (float): The scalar multiplier.
    x (np.ndarray): The first input array.
    y (np.ndarray): The second input array.

    Returns:
    np.ndarray: The result of the operation.
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Check that x and y have the same length
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Perform the axpy operation
    dst = alpha * x + y

    return dst

def daxpy(alpha, x, y):
    """
    Perform the operation dst = alpha * x + y.

    Parameters:
    alpha (float): The scalar multiplier.
    x (np.ndarray): The first input array.
    y (np.ndarray): The second input array.

    Returns:
    np.ndarray: The result of the operation.
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Check that x and y have the same length
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Perform the axpy operation
    dst = alpha * x + y

    return dst

def main():
    matrix_lib = ctypes.CDLL("./matrix-lib.dll",winmode=True)

    matrix_lib.add.argtypes = (ctypes.c_int, ctypes.c_int)
    matrix_lib.add.restype = ctypes.c_long

    matrix_lib.saxpy.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    matrix_lib.saxpy.restype = None

    matrix_lib.daxpy.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    matrix_lib.daxpy.restype = None

    x = np.random.rand(1200000)
    y = np.random.rand(1200000)
    dst = np.zeros(len(y),dtype=np.float64)

    start = time.time()
    matrix_lib.daxpy(1.0, len(dst), x.ctypes.data, y.ctypes.data, dst.ctypes.data)
    end = time.time()

    print("Calling C Func : " + f"{end-start}")

    start = time.time()
    daxpy(1, x, y)
    end = time.time()

    print("Calling Python Func : " +f"{end-start}")

    start = time.time()
    pb1.daxpy(len(dst),1,x,1,y,1)
    end = time.time()

    print("Calling Pyblas Level1 Func : " +f"{end-start}")

    start = time.time()
    scipy.linalg.blas.daxpy(x,y,a=1.0)
    end = time.time()

    print("Calling Scipy Daxpy Func : " +f"{end-start}")



if __name__ == "__main__":
    main()