import sys
import numpy as np
import copy
from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device, omp_get_team_num, omp_get_thread_num
import numba

def usage():
  print("Usage:", sys.argv[0],"<rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>", file=sys.stderr)
  print("\t<rows/cols>  - number of rows/cols in the grid (positive integer)", file=sys.stderr)
  print("\t<layers>  - number of layers in the grid (positive integer)", file=sys.stderr)

  print("\t<iteration> - number of iterations", file=sys.stderr)
  print("\t<powerFile>  - name of the file containing the initial power values of each cell", file=sys.stderr)
  print("\t<tempFile>  - name of the file containing the initial temperature values of each cell", file=sys.stderr)
  print("\t<outputFile - output file", file=sys.stderr)
  sys.exit(1)

def readinput(vect, grid_rows, grid_cols, layers, filename):
    with open(filename, "r") as f:
        for i in range(grid_rows):
            for j in range(grid_cols):
                for k in range(layers):
                    line = f.readline()
                    if not line:
                        print(
                            "not enough lines in file/invalid file format", file=sys.stderr)
                        sys.exit(1)
                    vect[i*grid_cols+j+k*grid_rows*grid_cols] = np.float32(line)

def writeoutput(vect, grid_rows, grid_cols, layers, file):
  index = 0
  with open(file, "w") as f:
    for i in range(grid_rows):
        for j in range(grid_cols):
            for k in range(layers):
                print(f"{index}\t{vect[i*grid_cols+j+k*grid_rows*grid_cols]:.3f}", file=f)
                index += 1

@njit
def core(iterations, numRows, numCols, layers, cc, cw, ce, cs, cn, cb, ct, stepDivCap, tIn, tOut, pIn, sel):
    temp = np.empty(numRows * numCols * layers, dtype=np.float32)
    with openmp("target data map(to: tIn, pIn) map(alloc: tOut, temp) device(1)"):
        start = omp_get_wtime()
        for iter in range(iterations):
            with openmp("target teams distribute parallel for collapse(2) thread_limit(256) device(1)"):
                for j in range(numRows):
                    for i in range(numCols):
                        amb_temp = np.float32(80.0)

                        c = i + j * numCols
                        xy = numCols * numRows

                        W = c if (i == 0) else c - 1
                        E = c if (i == numCols-1) else c + 1
                        N = c if (j == 0) else c - numCols
                        S = c if (j == numRows-1) else c + numCols

                        temp1 = temp2 = tIn[c]
                        temp3 = tIn[c+xy]
                        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S] \
                            + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp
                        c += xy
                        W += xy
                        E += xy
                        N += xy
                        S += xy

                        for k in range(1, layers-1):
                            temp1 = temp2
                            temp2 = temp3
                            temp3 = tIn[c+xy]
                            tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S] \
                                + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp
                            c += xy
                            W += xy
                            E += xy
                            N += xy
                            S += xy

                        temp1 = temp2
                        temp2 = temp3
                        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S] \
                            + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp

            temp = tIn
            tIn = tOut
            tOut = temp

        end = omp_get_wtime()
        print("Average kernel execution time", (end-start)/iterations, "(s)")

        if (iterations%2) != 0:
            with openmp("target update from (tIn) device(1)"):
                sel[:] = tIn
        else:
            with openmp("target update from (tOut) device(1)"):
                sel[:] = tOut


@njit
def computeTempCPU(pIn, tIn, tOut, nx, ny, nz,  Cap, Rx,  Ry,  Rz, dt, amb_temp, numiter):
    stepDivCap = np.float32(dt / Cap)
    ce = cw = np.float32(stepDivCap/ Rx)
    cn = cs = np.float32(stepDivCap/ Ry)
    ct = cb = np.float32(stepDivCap/ Rz)

    cc = np.float32(1.0 - (2.0*ce + 2.0*cn + 3.0*ct))

    for i in range(numiter):
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    c = x + y * nx + z * nx * ny

                    w = c if (x == 0) else c - 1
                    e = c if (x == nx - 1) else c + 1
                    n = c if (y == 0) else c - nx
                    s = c if (y == ny - 1) else c + nx
                    b = c if (z == 0) else c - nx * ny
                    t = c if (z == nz - 1) else c + nx * ny

                    tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + \
                                tIn[t]*ct + tIn[b]*cb + (dt/Cap) * pIn[c] + ct*amb_temp
        temp = tIn
        tIn = tOut
        tOut = temp

def main():
    if len(sys.argv) != 7:
        usage()

    MAX_PD   = np.float32(3.0e6)

    # required precision in degrees
    PRECISION    = np.float32(0.001)
    SPEC_HEAT_SI = np.float32(1.75e6)
    K_SI         = np.float32(100)

    # capacitance fitting factor
    FACTOR_CHIP  = np.float32(0.5)

    t_chip      = np.float32(0.0005)
    chip_height = np.float32(0.016)
    chip_width  = np.float32(0.016)
    amb_temp    = np.float32(80.0)

    iterations = int(sys.argv[3])

    pfile            = sys.argv[4]
    tfile            = sys.argv[5]
    ofile            = sys.argv[6]
    numCols      = int(sys.argv[1])
    numRows      = int(sys.argv[1])
    layers       = int(sys.argv[2])

    # calculating parameters*/

    dx         = np.float32(chip_height/numRows)
    dy         = np.float32(chip_width/numCols)
    dz         = np.float32(t_chip/layers)

    Cap        = np.float32(FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy)
    Rx         = np.float32(dy / (2.0 * K_SI * t_chip * dx))
    Ry         = np.float32(dx / (2.0 * K_SI * t_chip * dy))
    Rz         = np.float32(dz / (K_SI * dx * dy))

    max_slope  = np.float32(MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI))
    dt         = np.float32(PRECISION / max_slope)

    stepDivCap = dt / Cap
    ce               = cw                                              = np.float32(stepDivCap/ Rx)
    cn               = cs                                              = np.float32(stepDivCap/ Ry)
    ct               = cb                                              = np.float32(stepDivCap/ Rz)
    cc               = np.float32(1.0 - (2.0*ce + 2.0*cn + 3.0*ct))

    size = numCols * numRows * layers
    tIn      = np.zeros(size, dtype=np.float32)
    pIn      = np.zeros(size, dtype=np.float32)
    tCopy = np.empty(size, dtype=np.float32)
    tOut  = np.empty(size, dtype=np.float32)
    sel = np.empty(size, dtype=np.float32) # select tIn or tOut as the output of the computation

    print("Reading input files...")
    readinput(tIn, numRows, numCols, layers, tfile)
    readinput(pIn, numRows, numCols, layers, pfile)

    tCopy = copy.deepcopy(tIn)

    compile()
    print("Running benchmark...")
    total_start = omp_get_wtime()

    core(iterations, numRows, numCols, layers, cc, cw, ce,
         cs, cn, cb, ct, stepDivCap, tIn, tOut, pIn, sel)

    total_end = omp_get_wtime()
    print("Total execution time", total_end-total_start, "(s)")

    answer = np.empty(size, dtype=np.float32)
    ref_start = omp_get_wtime()
    computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations)
    ref_end = omp_get_wtime()
    print("Reference execution time", ref_end - ref_start, "(s)")

    rmse = np.sqrt(np.mean((sel-answer)**2))
    print("Root-mean-square error:", rmse)

    print("Writing output...")
    writeoutput(tOut, numRows, numCols, layers, ofile)

def compile():
  import time
  t1 = time.perf_counter()
  core.compile("""(int64, int64, int64, int64,
      float32, float32, float32, float32, float32, float32, float32, float32,
      Array(float32, 1, 'C', False, aligned=True),
      Array(float32, 1, 'C', False, aligned=True),
      Array(float32, 1, 'C', False, aligned=True),
      Array(float32, 1, 'C', False, aligned=True))""")
  t2 = time.perf_counter()
  print("ctime", t2 - t1, "s")

if __name__ == "__main__":
    main()
