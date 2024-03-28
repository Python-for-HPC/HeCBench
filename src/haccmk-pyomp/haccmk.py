import sys
import numpy as np
import math
from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device, omp_get_team_num, omp_get_thread_num

@njit
def haccmk (
    repeat,
    n,  # global size
    ilp, # inner loop count
    fsrrmax,
    mp_rsm,
    fcoeff,
    xx,
    yy,
    zz,
    mass,
    vx2,
    vy2,
    vz2 ):
  with openmp("""target data map(to: xx, yy, zz, mass)
                  map(from: vx2[0:n], vy2[0:n], vz2[0:n]) device(1)"""):
    total_time = np.float32(0.)

    for rep in range(repeat):
      with openmp("target update to (vx2[0:n]) device(1)"):
        pass
      with openmp("target update to (vy2[0:n]) device(1)"):
        pass
      with openmp("target update to (vz2[0:n]) device(1)"):
        pass

      start = omp_get_wtime()

      with openmp("""target teams distribute parallel for
          map(alloc: vx2[0:n], vy2[0:n], vz2[0:n])
          device(1)"""):
        for i in range(n):

          ma0 = np.float32(0.269327)
          ma1 = np.float32(-0.0750978)
          ma2 = np.float32(0.0114808)
          ma3 = np.float32(-0.00109313)
          ma4 = np.float32(0.0000605491)
          ma5 = np.float32(-0.00000147177)

          xi = np.float32(0.)
          yi = np.float32(0.)
          zi = np.float32(0.)

          xxi = xx[i]
          yyi = yy[i]
          zzi = zz[i]

          for j in range(ilp):
            dxc = xx[j] - xxi;
            dyc = yy[j] - yyi;
            dzc = zz[j] - zzi;

            r2 = dxc * dxc + dyc * dyc + dzc * dzc;

            m = mass[j] if ( r2 < fsrrmax ) else np.float32(0.)

            f = r2 + mp_rsm;
            f = m * ( np.float32(1.) / (f * math.sqrt(f)) -
                (ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))))

            xi = xi + f * dxc
            yi = yi + f * dyc
            zi = zi + f * dzc

          vx2[i] += xi * fcoeff
          vy2[i] += yi * fcoeff
          vz2[i] += zi * fcoeff

      end = omp_get_wtime()
      total_time += (end - start)

    print(f"Average kernel execution time", total_time / repeat, "(s)")

@njit
def haccmk_gold(
    count1,
    xxi,
    yyi,
    zzi,
    fsrrmax2,
    mp_rsm2,
    xx1,
    yy1,
    zz1,
    mass1):
  ma0 = np.float32(0.269327)
  ma1 = np.float32(-0.0750978)
  ma2 = np.float32(0.0114808)
  ma3 = np.float32(-0.00109313)
  ma4 = np.float32(0.0000605491)
  ma5 = np.float32(-0.00000147177)

  xi = np.float32(0.)
  yi = np.float32(0.)
  zi = np.float32(0.)

  for j in range(count1):
    dxc = xx1[j] - xxi;
    dyc = yy1[j] - yyi;
    dzc = zz1[j] - zzi;

    r2 = dxc * dxc + dyc * dyc + dzc * dzc;

    m = mass1[j] if ( r2 < fsrrmax2 ) else np.float32(0.)

    f = r2 + mp_rsm2;
    f =  m * (np.float32(1.) / (f * math.sqrt(f)) - (ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))))

    xi = xi + f * dxc;
    yi = yi + f * dyc;
    zi = zi + f * dzc;

  return xi, yi, zi


def main():
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <repeat>")
    return 1

  repeat = int(sys.argv[1])

  n1 = 784;
  n2 = 15000;
  print( f"Outer loop count is set {n1}")
  print( f"Inner loop count is set {n2}")

  # Init in loop later.
  xx = np.empty(n2, dtype=np.float32)
  yy = np.empty(n2, dtype=np.float32)
  zz = np.empty(n2, dtype=np.float32)
  mass = np.empty(n2, dtype=np.float32)

  # Init to zero
  vx2 = np.zeros(n2, dtype=np.float32)
  vy2 = np.zeros(n2, dtype=np.float32)
  vz2 = np.zeros(n2, dtype=np.float32)
  vx2_hw = np.zeros(n2, dtype=np.float32)
  vy2_hw = np.zeros(n2, dtype=np.float32)
  vz2_hw = np.zeros(n2, dtype=np.float32)

  # Initial data preparation
  fcoeff = np.float32(0.23)
  fsrrmax2 = np.float32(0.5)
  mp_rsm2 = np.float32(0.03)
  dx1 = np.float32(1.0)/n2;
  dy1 = np.float32(2.0)/n2
  dz1 = np.float32(3.0)/n2
  xx[0] = np.float32(0.)
  yy[0] = np.float32(0.)
  zz[0] = np.float32(0.)
  mass[0] = np.float32(2.)

  for i in range(1, n2):
    xx[i] = xx[i-1] + dx1;
    yy[i] = yy[i-1] + dy1;
    zz[i] = zz[i-1] + dz1;
    mass[i] = i * np.float32(0.01) + xx[i];


  print("Running reference implementation for verification...")
  start = omp_get_wtime()
  for i in range(n1):
    dx2, dy2, dz2 = haccmk_gold( n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass)
    vx2[i] = vx2[i] + dx2 * fcoeff;
    vy2[i] = vy2[i] + dy2 * fcoeff;
    vz2[i] = vz2[i] + dz2 * fcoeff;
  end = omp_get_wtime()
  print("Reference execution time", end-start, "(s)")

  print("Running benchmark...")
  compile()
  haccmk(repeat, n1, n2, fsrrmax2, mp_rsm2, fcoeff, xx,
      yy, zz, mass, vx2_hw, vy2_hw, vz2_hw)

  # verify
  error = False
  eps = np.float32(1e-1)
  for i in range(n2):
    if (abs(vx2[i] - vx2_hw[i]) > eps):
      print(f"error at vx2[{i}] {vx2[i]} {vx2_hw[i]}")
      error = True;
      break;
    if (abs(vy2[i] - vy2_hw[i]) > eps):
      print(f"error at vx2[{i}] {vy2[i]} {vy2_hw[i]}")
      error = True;
      break;
    if (abs(vz2[i] - vz2_hw[i]) > eps):
      print(f"error at vx2[{i}] {vz2[i]} {vz2_hw[i]}")
      error = True;
      break;

  print(f"{'FAIL' if error else 'PASS'}")

  return 0;

def compile():
  import time
  t1 = time.perf_counter()
  haccmk.compile("""void(int64, int64, int64, float32, float32, float32, Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True),
  Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True))""")
  t2 = time.perf_counter()
  print("ctime", t2-t1, "s")

if __name__ == "__main__":
    main()
