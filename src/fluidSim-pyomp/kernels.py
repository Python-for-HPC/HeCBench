from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device, omp_get_team_num, omp_get_thread_num
import numpy as np
# Thread block size

@njit(inline='always')
def dot(a, b):
    return (a[0] * b[0] + a[1] * b[1]) + (a[2] * b[2] + a[3] * b[3])

# Calculates equivalent distribution

@njit(inline='always')
def ced(rho, weight, diro, u):
    u2 = (u[0] * u[0]) + (u[1] * u[1])
    eu = (diro[0] * u[0]) + (diro[1] * u[1])
    return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)

@njit(inline='always')
def newPos(p, diro):
    return p + np.round(diro).astype(int)
    """
    np = [0] * 8
    np[0] = p + int(diro[0])
    np[1] = p + int(diro[1])
    np[2] = p + int(diro[2])
    np[3] = p + int(diro[3])
    np[4] = p + int(diro[4])
    np[5] = p + int(diro[5])
    np[6] = p + int(diro[6])
    np[7] = p + int(diro[7])
    return np
    """

@njit(inline='always')
def fma8(a, b, c):
    return (a * b) + c
    """
    r = [0] * 8
    r[0] = a * b[0] + c[0]
    r[1] = a * b[1] + c[1]
    r[2] = a * b[2] + c[2]
    r[3] = a * b[3] + c[3]
    r[4] = a * b[4] + c[4]
    r[5] = a * b[5] + c[5]
    r[6] = a * b[6] + c[6]
    r[7] = a * b[7] + c[7]
    return r
    """

@njit
def lbm(width, height, if0, of0, if1234, of1234, if5678, of5678, typeo, dirX, dirY, weight, omega):
    e1234 = np.zeros(4, dtype=np.int32)
    e5678 = np.zeros(4, dtype=np.int32)
    u = np.zeros(2, dtype=np.int32) # Velocity
    x1234 = np.zeros(4, dtype=np.float64)
    y1234 = np.zeros(4, dtype=np.float64)
    x5678 = np.zeros(4, dtype=np.float64)
    y5678 = np.zeros(4, dtype=np.float64)
    temp  = np.zeros(4, dtype=np.float64)
    with openmp("""target teams distribute parallel for collapse(2) thread_limit(256) map(to: e1234, e5678, u, x1234, y1234, x5678, y5678, temp)"""):
        for idy in range(height):
            for idx in range(width):
                pos = idx + width * idy
                # Read input distributions
                f0 = if0[pos]
                f1234 = if1234[pos]
                f5678 = if5678[pos]
                # intermediate results
                e0 = 0
                rho = 0 # Density
                """
                e1234 = np.zeros(4, dtype=np.int32)
                e5678 = np.zeros(4, dtype=np.int32)
                u = np.zeros(2, dtype=np.int32) # Velocity
                """
                # Collide
                if typeo[pos]: # Boundary
                    # Swap directions
                    e1234[0] = f1234[2]
                    e1234[1] = f1234[3]
                    e1234[2] = f1234[0]
                    e1234[3] = f1234[1]
                    e5678[0] = f5678[2]
                    e5678[1] = f5678[3]
                    e5678[2] = f5678[0]
                    e5678[3] = f5678[1]
                    rho = 0
                    u[:] = 0
                else: # Fluid
                    # Compute rho and u
                    # Rho is computed by doing a reduction on f
                    temp[:] = f1234 + f5678
                    rho = f0 + temp[0] + temp[1] + temp[2] + temp[3]
                    # Compute velocity in x and y directions
                    x1234[:] = dirX[0:4]
                    y1234[:] = dirY[0:4]
                    x5678[:] = dirX[4:8]
                    y5678[:] = dirY[4:8]
                    """
                    u[0] = (dot(f1234, x1234) + dot(f5678, x5678)) / rho
                    u[1] = (dot(f1234, y1234) + dot(f5678, y5678)) / rho
                    # Compute f
                    e0 = ced(rho, weight[0], [0, 0], u)
                    e1234[0] = ced(rho, weight[1], [dirX[0], dirY[0]], u)
                    e1234[1] = ced(rho, weight[2], [dirX[1], dirY[1]], u)
                    e1234[2] = ced(rho, weight[3], [dirX[2], dirY[2]], u)
                    e1234[3] = ced(rho, weight[4], [dirX[3], dirY[3]], u)
                    e5678[0] = ced(rho, weight[5], [dirX[4], dirY[4]], u)
                    e5678[1] = ced(rho, weight[6], [dirX[5], dirY[5]], u)
                    e5678[2] = ced(rho, weight[7], [dirX[6], dirY[6]], u)
                    e5678[3] = ced(rho, weight[8], [dirX[7], dirY[7]], u)
                    e0 = (1.0 - omega) * f0 + omega * e0
                    e1234 = [(1.0 - omega) * f1234[i] + omega * e1234[i] for i in range(4)]
                    e5678 = [(1.0 - omega) * f5678[i] + omega * e5678[i] for i in range(4)]
                    """
                # Propagate
                t3 = idx > 0          # Not on Left boundary
                t1 = idx < width - 1  # Not on Right boundary
                t4 = idy > 0          # Not on Upper boundary
                t2 = idy < height - 1 # Not on lower boundary
                if t1 and t2 and t3 and t4:
                    # New positions to write (Each thread will write 8 values)
                    # Note the propagation sources (e.g. f1234) imply the OLD locations for each thread
                    pass
                    """
                    nX = newPos(idx, dirX)
                    nY = newPos(idy, dirY)
                    nPos = fma8(width, nY, nX)
                    # Write center distribution to thread's location
                    of0[pos] = e0
                    # Propagate to right cell
                    of1234[nPos[0]] = e1234[0]
                    # Propagate to Lower cell
                    of1234[nPos[1]] = e1234[1]
                    # Propagate to left cell
                    of1234[nPos[2]] = e1234[2]
                    # Propagate to Upper cell
                    of1234[nPos[3]] = e1234[3]
                    # Propagate to Lower-Right cell
                    of5678[nPos[4]] = e5678[0]
                    # Propogate to Lower-Left cell
                    of5678[nPos[5]] = e5678[1]
                    # Propagate to Upper-Left cell
                    of5678[nPos[6]] = e5678[2]
                    # Propagate to Upper-Right cell
                    of5678[nPos[7]] = e5678[3]
                    """

@njit
def fluidSim(iterations, omega, dims, h_type, u, rho, dirX, dirY, h_weight, h_if0, h_if1234, h_if5678, h_of0, h_of1234, h_of5678):
    width = dims[0]
    height = dims[1]
    temp = width * height
    dbl_size = temp
    h_of0[:] = h_if0
    h_of1234[:] = h_if1234
    h_of5678[:] = h_if5678
    with openmp("""target data map (to: h_of, h_of1234, h_of5678, h_weight, h_type) map (tofrom: h_if0, h_if1234, h_if5678)"""):
        for _ in range(iterations):
            lbm(width, height, h_if0, h_of0, h_if1234, h_of1234, h_if5678, h_of5678, h_type, dirX, dirY, h_weight, omega)
            # Swap device buffers
            temp0 = h_of0
            temp1234 = h_of1234
            temp5678 = h_of5678
            h_of0 = h_if0
            h_of1234 = h_if1234
            h_of5678 = h_if5678
            h_if0 = temp0
            h_if1234 = temp1234
            h_if5678 = temp5678
    if iterations % 2 == 0:
        h_of0[:] = h_if0
        h_of1234[:] = h_if1234
        h_of5678[:] = h_if5678
