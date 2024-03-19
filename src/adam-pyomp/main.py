import sys
import numpy as np
from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime
import numba
import math

ADAM_MODE_0 = 0
ADAM_MODE_1 = 1

@njit(fastmath=True)
def adam(
    p,
    m,
    v,
    g,
    b1,
    b2,
    eps,
    grad_scale,
    step_size,
    time_step,
    vector_size,
    mode,
    decay):
  one = np.float32(1.)
  with openmp("target teams distribute parallel for thread_limit(256) device(1)"):
    for j in range(vector_size):
      # XXX: is the original timestep from 0 correct? at t=0 m_corrected,
      # v_corrected cause division by zero.
      for t in range(1, time_step+1):
        scaled_grad = g[j]/grad_scale
        m[j] = b1*m[j] + (one-b1)*scaled_grad
        v[j] = b2*v[j] + (one-b2)*scaled_grad*scaled_grad
        m_corrected = m[j] / (one-(b1**np.float32(t)))
        v_corrected = v[j] / (one-(b2**np.float32(t)))
        if mode == ADAM_MODE_0:
          denom = math.sqrt(v_corrected + eps)
        else:  # Mode 1
          denom = math.sqrt(v_corrected) + eps
        update = (m_corrected/denom) + (decay*p[j])
        p[j] -= (step_size*update)


@njit
def reference(
    repeat,
    p,
    m,
    v,
    g,
    b1,
    b2,
    eps,
    grad_scale,
    step_size,
    time_step,
    vector_size,
    mode,
    decay):
  for i in range(repeat):
    for j in range(vector_size):
      # XXX: is the original timestep from 0 correct? at t=0 m_corrected,
      # v_corrected cause division by zero.
      for t in range(1, time_step+1):
        scaled_grad = g[j]/grad_scale
        m[j] = b1*m[j] + (1.-b1)*scaled_grad
        v[j] = b2*v[j] + (1.-b2)*scaled_grad*scaled_grad
        m_corrected = m[j] / (1.-pow(b1, t))
        v_corrected = v[j] / (1.-pow(b2, t))
        if mode == ADAM_MODE_0:
          denom = math.sqrt(v_corrected + eps)
        else:  # Mode 1
          denom = math.sqrt(v_corrected) + eps
        update = (m_corrected/denom) + (decay*p[j])
        p[j] -= (step_size*update)

@njit
def core(
    repeat,
    p,
    m,
    v,
    g,
    b1,
    b2,
    eps,
    grad_scale,
    step_size,
    time_step,
    vector_size,
    mode,
    decay):
  with openmp("""target enter data
              map(to: m, v, g, p)
              device(1)"""):
    start = omp_get_wtime()

    for i in range(repeat):
       adam(
           p, m, v, g,
           b1, b2,
           eps,
           grad_scale,
           step_size,
           time_step,
           vector_size,
           mode,
           decay)

    end = omp_get_wtime()
    print("Average kernel execution time",
          (end - start)*1000.0 / repeat, "(ms)")

  with openmp("target exit data map(from: p) device(1)"):
    pass


def main():
  if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <vector size> <number of time steps> <repeat>")
    return 1

  vector_size = int(sys.argv[1])
  time_step = int(sys.argv[2])
  repeat = int(sys.argv[3])

  np.random.seed(123)
  m = np.random.rand(vector_size).astype(np.float32)
  v = np.random.rand(vector_size).astype(np.float32)
  g = np.random.rand(vector_size).astype(np.float32)
  p = np.random.rand(vector_size).astype(np.float32)
  r = np.copy(p)
  #print("=== init")
  #print("m[0:10]", m[0:10])
  #print("v[0:10]", v[0:10])
  #print("g[0:10]", g[0:10])
  #print("p[0:10]", p[0:10])
  #print("r[0:10]", r[0:10])
  #print("=== end of init")

  # Arbitrary constants
  step_size = np.float32(1e-3)
  decay = np.float32(0.5)
  beta1 = np.float32(0.9)
  beta2 = np.float32(0.999)
  eps = np.float32(1e-10)
  grad_scale = np.float32(256.)

  mode = ADAM_MODE_0

  # Execute on device.
  compile()
  core(
        repeat,
        p, m, v, g,
        beta1, beta2,
        eps,
        grad_scale,
        step_size,
        time_step,
        vector_size,
        mode,
        decay)

  #print("=== after device")
  #print("m[0:10]", m[0:10])
  #print("v[0:10]", v[0:10])
  #print("g[0:10]", g[0:10])
  #print("p[0:10]", p[0:10])
  #print("r[0:10]", r[0:10])
  #print("=== end of after device")

  start = omp_get_wtime()
  # verify
  reference(
      repeat,
      r, m, v, g,
      beta1, beta2,
      eps,
      grad_scale,
      step_size,
      time_step,
      vector_size,
      mode,
      decay)
  end = omp_get_wtime()
  print("Average reference execution time",
        (end-start)*1000 / repeat, "(ms)")

  #print("=== after ref")
  #print("m[0:10]", m[0:10])
  #print("v[0:10]", v[0:10])
  #print("g[0:10]", g[0:10])
  #print("p[0:10]", p[0:10])
  #print("r[0:10]", r[0:10])
  #print("=== end of ref")

  ok = False if np.any((r - p) > np.float32(1e-3)) else True
  print("PASS" if ok else "FAIL")

  return 0

def compile():
  import time
  t1 = time.perf_counter()
  core.compile("""void(int64, Array(float32, 1, 'C', False, aligned=True),
  Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False,
  aligned=True), Array(float32, 1, 'C', False, aligned=True), float32, float32,
  float32, float32, float32, int64, int64, int64, float32)""")
  t2 = time.perf_counter()
  print("ctime", t2 - t1, "s")


if __name__ == "__main__":
  main()
