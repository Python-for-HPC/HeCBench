from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import (
    omp_get_wtime,
    omp_set_num_threads,
    omp_get_num_threads,
    omp_get_num_devices,
    omp_is_initial_device,
    omp_get_team_num,
    omp_get_thread_num,
)
import numpy as np

# Thread block size


@njit(inline="always")
def dot(a, b):
    return (a[0] * b[0] + a[1] * b[1]) + (a[2] * b[2] + a[3] * b[3])


# Calculates equivalent distribution


@njit(inline="always")
def ced(rho, weight, dirx, diry, u_0, u_1):
    u2 = (u_0 * u_0) + (u_1 * u_1)
    eu = (dirx * u_0) + (diry * u_1)
    return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)


@njit
def lbm(
    width,
    height,
    if0,
    of0,
    if1234,
    of1234,
    if5678,
    of5678,
    typeo,
    dirX,
    dirY,
    weight,
    omega,
):
    with openmp(
        """target teams distribute parallel for collapse(2) thread_limit(256) device(1)"""
    ):
        for idy in range(height):
            for idx in range(width):
                pos = idx + width * idy
                # Read input distributions
                f0 = if0[pos]
                f1234 = if1234[pos]
                f5678 = if5678[pos]
                # intermediate results
                e0 = 0
                rho = 0  # Density
                # Collide
                if typeo[pos]:  # Boundary
                    # Swap directions
                    e1234_0 = f1234[2]
                    e1234_1 = f1234[3]
                    e1234_2 = f1234[0]
                    e1234_3 = f1234[1]
                    e5678_0 = f5678[2]
                    e5678_1 = f5678[3]
                    e5678_2 = f5678[0]
                    e5678_3 = f5678[1]
                    rho = 0
                    u_0 = 0
                    u_1 = 0
                else:  # Fluid
                    # Compute rho and u
                    # Rho is computed by doing a reduction on f
                    elem_sum_0 = f1234[0] + f5678[0]
                    elem_sum_1 = f1234[1] + f5678[1]
                    elem_sum_2 = f1234[2] + f5678[2]
                    elem_sum_3 = f1234[3] + f5678[3]
                    rho = f0 + elem_sum_0 + elem_sum_1 + elem_sum_2 + elem_sum_3
                    # Compute velocity in x and y directions
                    # x1234[:] = dirX[0:4]
                    # y1234[:] = dirY[0:4]
                    # x5678[:] = dirX[4:8]
                    # y5678[:] = dirY[4:8]
                    u_0 = (dot(f1234, dirX[0:4]) + dot(f5678, dirX[4:8])) / rho
                    u_1 = (dot(f1234, dirY[0:4]) + dot(f5678, dirY[4:8])) / rho

                    # Compute f
                    e0 = ced(rho, weight[0], 0, 0, u_0, u_1)
                    e1234_0 = ced(rho, weight[1], dirX[0], dirY[0], u_0, u_1)
                    e1234_1 = ced(rho, weight[2], dirX[1], dirY[1], u_0, u_1)
                    e1234_2 = ced(rho, weight[3], dirX[2], dirY[2], u_0, u_1)
                    e1234_3 = ced(rho, weight[4], dirX[3], dirY[3], u_0, u_1)
                    e5678_0 = ced(rho, weight[5], dirX[4], dirY[4], u_0, u_1)
                    e5678_1 = ced(rho, weight[6], dirX[5], dirY[5], u_0, u_1)
                    e5678_2 = ced(rho, weight[7], dirX[6], dirY[6], u_0, u_1)
                    e5678_3 = ced(rho, weight[8], dirX[7], dirY[7], u_0, u_1)
                    e0 = (1.0 - omega) * f0 + omega * e0
                    # e1234 = [
                    #    (1.0 - omega) * f1234[i] + omega * e1234[i] for i in range(4)
                    # ]
                    e1234_0 = (1.0 - omega) * f1234[0] + omega * e1234_0
                    e1234_1 = (1.0 - omega) * f1234[1] + omega * e1234_1
                    e1234_2 = (1.0 - omega) * f1234[2] + omega * e1234_2
                    e1234_3 = (1.0 - omega) * f1234[3] + omega * e1234_3

                    # e5678 = [
                    #    (1.0 - omega) * f5678[i] + omega * e5678[i] for i in range(4)
                    # ]
                    e5678_0 = (1.0 - omega) * f5678[0] + omega * e5678_0
                    e5678_1 = (1.0 - omega) * f5678[1] + omega * e5678_1
                    e5678_2 = (1.0 - omega) * f5678[2] + omega * e5678_2
                    e5678_3 = (1.0 - omega) * f5678[3] + omega * e5678_3
                # Propagate
                t3 = idx > 0  # Not on Left boundary
                t1 = idx < width - 1  # Not on Right boundary
                t4 = idy > 0  # Not on Upper boundary
                t2 = idy < height - 1  # Not on lower boundary
                if t1 and t2 and t3 and t4:
                    # New positions to write (Each thread will write 8 values)
                    # Note the propagation sources (e.g. f1234) imply the OLD locations for each thread
                    # nX = newPos(idx, dirX)
                    # nY = newPos(idy, dirY)
                    # nPos = fma8(width, nY, nX)
                    # Write center distribution to thread's location
                    of0[pos] = e0
                    # Propagate to right cell
                    nX = idx + dirX[0]
                    nY = idy + dirY[0]
                    nPos = width * nY + nX
                    of1234[nPos, 0] = e1234_0
                    # Propagate to Lower cell
                    nX = idx + dirX[1]
                    nY = idy + dirY[1]
                    nPos = width * nY + nX
                    of1234[nPos, 1] = e1234_1
                    ## Propagate to left cell
                    nX = idx + dirX[2]
                    nY = idy + dirY[2]
                    nPos = width * nY + nX
                    of1234[nPos, 2] = e1234_2
                    # Propagate to Upper cell
                    nX = idx + dirX[3]
                    nY = idy + dirY[3]
                    nPos = width * nY + nX
                    of1234[nPos, 3] = e1234_3
                    # Propagate to Lower-Right cell
                    nX = idx + dirX[4]
                    nY = idy + dirY[4]
                    nPos = width * nY + nX
                    of5678[nPos, 0] = e5678_0
                    # Propogate to Lower-Left cell
                    nX = idx + dirX[5]
                    nY = idy + dirY[5]
                    nPos = width * nY + nX
                    of5678[nPos, 1] = e5678_1
                    # Propagate to Upper-Left cell
                    nX = idx + dirX[6]
                    nY = idy + dirY[6]
                    nPos = width * nY + nX
                    of5678[nPos, 2] = e5678_2
                    # Propagate to Upper-Right cell
                    nX = idx + dirX[7]
                    nY = idy + dirY[7]
                    nPos = width * nY + nX
                    of5678[nPos, 3] = e5678_3


@njit
def fluidSim(
    iterations,
    omega,
    dims,
    h_type,
    dirX,
    dirY,
    h_weight,
    h_if0,
    h_if1234,
    h_if5678,
    h_of0,
    h_of1234,
    h_of5678,
):
    width = dims[0]
    height = dims[1]
    h_of0[:] = h_if0
    h_of1234[:] = h_if1234
    h_of5678[:] = h_if5678
    with openmp(
        """target data map(to: h_of0, h_of1234, h_of5678, h_weight, h_type) map(tofrom: h_if0, h_if1234, h_if5678) device(1)"""
    ):
        t1 = omp_get_wtime()
        for _ in range(iterations):
            lbm(
                width,
                height,
                h_if0,
                h_of0,
                h_if1234,
                h_of1234,
                h_if5678,
                h_of5678,
                h_type,
                dirX,
                dirY,
                h_weight,
                omega,
            )
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
    t2 = omp_get_wtime()
    print("Average kernel execution time", (t2 - t1) / iterations, "(s)\n")

    if iterations % 2 == 0:
        h_of0[:] = h_if0
        h_of1234[:] = h_if1234
        h_of5678[:] = h_if5678
