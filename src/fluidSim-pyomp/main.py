import sys
import random
import numpy as np
from numba import njit, typeof
from numba.core.event import RecordingListener, register
from numba.core import types as numba_types
from kernels import fluidSim
import time


@njit
def computefEq(rho, weight, diro, velocity):
    u2 = velocity[0] ** 2 + velocity[1] ** 2
    eu = diro[0] * velocity[0] + diro[1] * velocity[1]
    return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * u2)


# error bound
EPISON = 1e-3


@njit
def reference(
    iterations,
    omega,
    dims,
    h_type,
    rho,
    diro,
    weight,
    h_if0,
    h_if1234,
    h_if5678,
    h_of0,
    h_of1234,
    h_of5678,
):
    v_of0 = h_of0
    v_of1234 = h_of1234
    v_of5678 = h_of5678
    temp = dims[0] * dims[1]
    dbl_size = temp
    dbl4_size = (dbl_size, 4)
    v_if0 = np.zeros(dbl_size)
    v_if1234 = np.zeros(dbl4_size)
    v_if5678 = np.zeros(dbl4_size)
    v_ef0 = np.zeros(dbl_size)
    v_ef1234 = np.zeros(dbl4_size)
    v_ef5678 = np.zeros(dbl4_size)
    v_if0[:] = h_if0
    v_if1234[:] = h_if1234
    v_if5678[:] = h_if5678
    v_of0[:] = h_if0
    v_of1234[:] = h_if1234
    v_of5678[:] = h_if5678
    for i in range(iterations):
        for y in range(dims[1]):
            for x in range(dims[0]):
                pos = x + y * dims[0]
                if h_type[pos] == 1:  # Boundary
                    v_ef0[pos] = v_if0[pos]
                    temp1 = v_if1234[pos, 2]
                    temp2 = v_if1234[pos, 3]
                    temp3 = v_if1234[pos, 0]
                    temp4 = v_if1234[pos, 1]
                    v_ef1234[pos, 0] = temp1
                    v_ef1234[pos, 1] = temp2
                    v_ef1234[pos, 2] = temp3
                    v_ef1234[pos, 3] = temp4
                    temp1 = v_if5678[pos, 2]
                    temp2 = v_if5678[pos, 3]
                    temp3 = v_if5678[pos, 0]
                    temp4 = v_if5678[pos, 1]
                    v_ef5678[pos, 0] = temp1
                    v_ef5678[pos, 1] = temp2
                    v_ef5678[pos, 2] = temp3
                    v_ef5678[pos, 3] = temp4
                    rho[pos] = 0
                else:  # Fluid
                    vel = np.zeros(2)
                    den = (
                        v_if0[pos]
                        + v_if1234[pos, 0]
                        + v_if1234[pos, 1]
                        + v_if1234[pos, 2]
                        + v_if1234[pos, 3]
                        + v_if5678[pos, 0]
                        + v_if5678[pos, 1]
                        + v_if5678[pos, 2]
                        + v_if5678[pos, 3]
                    )
                    vel[0] = (
                        v_if1234[pos, 0] * diro[1][0]
                        + v_if1234[pos, 1] * diro[2][0]
                        + v_if1234[pos, 2] * diro[3][0]
                        + v_if1234[pos, 3] * diro[4][0]
                        + v_if5678[pos, 0] * diro[5][0]
                        + v_if5678[pos, 1] * diro[6][0]
                        + v_if5678[pos, 2] * diro[7][0]
                        + v_if5678[pos, 3] * diro[8][0]
                    )
                    vel[1] = (
                        v_if1234[pos, 0] * diro[1][1]
                        + v_if1234[pos, 1] * diro[2][1]
                        + v_if1234[pos, 2] * diro[3][1]
                        + v_if1234[pos, 3] * diro[4][1]
                        + v_if5678[pos, 0] * diro[5][1]
                        + v_if5678[pos, 1] * diro[6][1]
                        + v_if5678[pos, 2] * diro[7][1]
                        + v_if5678[pos, 3] * diro[8][1]
                    )
                    vel[0] /= den
                    vel[1] /= den
                    v_ef0[pos] = computefEq(den, weight[0], diro[0], vel)
                    v_ef1234[pos, 0] = computefEq(den, weight[1], diro[1], vel)
                    v_ef1234[pos, 1] = computefEq(den, weight[2], diro[2], vel)
                    v_ef1234[pos, 2] = computefEq(den, weight[3], diro[3], vel)
                    v_ef1234[pos, 3] = computefEq(den, weight[4], diro[4], vel)
                    v_ef5678[pos, 0] = computefEq(den, weight[5], diro[5], vel)
                    v_ef5678[pos, 1] = computefEq(den, weight[6], diro[6], vel)
                    v_ef5678[pos, 2] = computefEq(den, weight[7], diro[7], vel)
                    v_ef5678[pos, 3] = computefEq(den, weight[8], diro[8], vel)
                    v_ef0[pos] = (1 - omega) * v_if0[pos] + omega * v_ef0[pos]
                    v_ef1234[pos, 0] = (1 - omega) * v_if1234[
                        pos, 0
                    ] + omega * v_ef1234[pos, 0]
                    v_ef1234[pos, 1] = (1 - omega) * v_if1234[
                        pos, 1
                    ] + omega * v_ef1234[pos, 1]
                    v_ef1234[pos, 2] = (1 - omega) * v_if1234[
                        pos, 2
                    ] + omega * v_ef1234[pos, 2]
                    v_ef1234[pos, 3] = (1 - omega) * v_if1234[
                        pos, 3
                    ] + omega * v_ef1234[pos, 3]
                    v_ef5678[pos, 0] = (1 - omega) * v_if5678[
                        pos, 0
                    ] + omega * v_ef5678[pos, 0]
                    v_ef5678[pos, 1] = (1 - omega) * v_if5678[
                        pos, 1
                    ] + omega * v_ef5678[pos, 1]
                    v_ef5678[pos, 2] = (1 - omega) * v_if5678[
                        pos, 2
                    ] + omega * v_ef5678[pos, 2]
                    v_ef5678[pos, 3] = (1 - omega) * v_if5678[
                        pos, 3
                    ] + omega * v_ef5678[pos, 3]
        for y in range(1, dims[1] - 1):
            for x in range(1, dims[0] - 1):
                src_pos = x + dims[0] * y
                for k in range(9):
                    nx = x + int(diro[k][0])
                    ny = y + int(diro[k][1])
                    dst_pos = nx + dims[0] * ny
                    if k == 0:
                        v_of0[dst_pos] = v_ef0[src_pos]
                    elif k == 1:
                        v_of1234[dst_pos, 0] = v_ef1234[src_pos, 0]
                    elif k == 2:
                        v_of1234[dst_pos, 1] = v_ef1234[src_pos, 1]
                    elif k == 3:
                        v_of1234[dst_pos, 2] = v_ef1234[src_pos, 2]
                    elif k == 4:
                        v_of1234[dst_pos, 3] = v_ef1234[src_pos, 3]
                    elif k == 5:
                        v_of5678[dst_pos, 0] = v_ef5678[src_pos, 0]
                    elif k == 6:
                        v_of5678[dst_pos, 1] = v_ef5678[src_pos, 1]
                    elif k == 7:
                        v_of5678[dst_pos, 2] = v_ef5678[src_pos, 2]
                    elif k == 8:
                        v_of5678[dst_pos, 3] = v_ef5678[src_pos, 3]
        v_if0, v_of0 = v_of0, v_if0
        v_if1234, v_of1234 = v_of1234, v_if1234
        v_if5678, v_of5678 = v_of5678, v_if5678
    h_of0[:] = v_if0
    h_of1234[:] = v_if1234
    h_of5678[:] = v_if5678
    if iterations % 2:
        v_if0, v_of0 = v_of0, v_if0
        v_if1234, v_of1234 = v_of1234, v_if1234
        v_if5678, v_of5678 = v_of5678, v_if5678
    return h_of0, h_of1234, h_of5678


def verify(h_of0, h_of1234, h_of5678, v_of0, v_of1234, v_of5678):
    ok = (
        np.allclose(h_of0, v_of0, atol=EPISON)
        and np.allclose(h_of1234, v_of1234, atol=EPISON)
        and np.allclose(h_of5678, v_of5678, atol=EPISON)
    )
    print("PASS" if ok else "FAIL")


@njit
def init(dims, u, w, e, h_if0, h_if1234, h_if5678, h_type):
    # Initial velocity is nonzero for verifying host and device results
    u0 = [0.01, 0.01]
    random.seed(123)
    for y in range(dims[1]):
        for x in range(dims[0]):
            pos = x + y * dims[0]
            # Random values for verification
            den = random.randint(1, 10)
            # Initialize the velocity buffer
            u[pos] = u0
            # Initialize the frequency (i.e. the number of particles in
            # a cell going in each velocity direction)
            h_if0[pos] = computefEq(den, w[0], e[0], u0)
            h_if1234[pos] = [computefEq(den, w[i], e[i], u0) for i in range(1, 5)]
            h_if5678[pos] = [computefEq(den, w[i], e[i], u0) for i in range(5, 9)]
            # Initialize boundary cells
            if x == 0 or x == (dims[0] - 1) or y == 0 or y == (dims[1] - 1):
                h_type[pos] = True
            # Initialize fluid cells
            else:
                h_type[pos] = False

def main():
    if len(sys.argv) != 2:
        print(f"Usage {sys.argv[0]} <iterations>")
        return 1

    iterations = int(sys.argv[1])  # Simulation iterations
    lbm_width = 256  # Dimension of LBM simulation area
    lbm_height = 256
    dims = (lbm_width, lbm_height)
    temp = dims[0] * dims[1]

    # Nine velocity directions for each cell
    e = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    # Weights depend on lattice geometry
    w = np.array(
        [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ]
    )
    # Omega equals time step divided by Tau (viscosity of the fluid)
    omega = 1.2

    # host inputs
    h_if0 = np.zeros(temp)
    h_if1234 = np.zeros((temp, 4))
    h_if5678 = np.zeros((temp, 4))

    # Reference outputs
    v_of0 = np.zeros(temp)
    v_of1234 = np.zeros((temp, 4))
    v_of5678 = np.zeros((temp, 4))

    # Host outputs
    h_of0 = np.zeros(temp)
    h_of1234 = np.zeros((temp, 4))
    h_of5678 = np.zeros((temp, 4))

    # Cell Type - Boundary = 1 or Fluid = 0
    h_type = np.zeros(temp, dtype=bool)

    # Density
    rho = np.zeros(temp)

    # Velocity
    u = np.zeros((temp, 2))

    # Initial velocity is nonzero for verifying host and device results
    init(dims, u, w, e, h_if0, h_if1234, h_if5678, h_type)

    # Initialize direction vectors for each cell
    dirX = np.empty(e.shape[0] - 1, dtype=np.int64)
    dirX[:] = e[1:, 0]
    dirY = np.empty(e.shape[0] - 1, dtype=np.int64)
    dirY[:] = e[1:, 1]

    t1 = time.perf_counter()
    reference(
        iterations,
        omega,
        dims,
        h_type,
        rho,
        e,
        w,
        h_if0,
        h_if1234,
        h_if5678,
        v_of0,
        v_of1234,
        v_of5678,
    )
    t2 = time.perf_counter()
    print("Reference execution time", t2 - t1, "s")

    compile_recorder = RecordingListener()
    register("numba:compile", compile_recorder)
    # TODO: Fix eager compilation.
    t1 = time.perf_counter()

    fluidSim.compile(
       numba_types.none(
           typeof(iterations),
           typeof(omega),
           typeof(dims),
           typeof(h_type),
           typeof(dirX),
           typeof(dirY),
           typeof(w),
           typeof(h_if0),
           typeof(h_if1234),
           typeof(h_if5678),
           typeof(h_of0),
           typeof(h_of1234),
           typeof(h_of5678))
    )
    t2 = time.perf_counter()
    print("Compilation time", t2 - t1, "s")
    print("num compiles:", len(compile_recorder.buffer))

    compile_recorder = RecordingListener()
    register("numba:compile", compile_recorder)
    t1 = time.perf_counter()
    fluidSim(
        iterations,
        omega,
        dims,
        h_type,
        dirX,
        dirY,
        w,
        h_if0,
        h_if1234,
        h_if5678,
        h_of0,
        h_of1234,
        h_of5678,
    )

    t2 = time.perf_counter()
    print("num compiles:", len(compile_recorder.buffer))
    print("Total GPU execution time", t2 - t1, "s")

    verify(h_of0, h_of1234, h_of5678, v_of0, v_of1234, v_of5678)


if __name__ == "__main__":
    main()
