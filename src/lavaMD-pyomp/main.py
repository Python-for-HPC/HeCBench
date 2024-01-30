from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device, omp_get_team_num, omp_get_thread_num
import numba
import numpy as np
import sys
import math
import collections

# ----------------------------------------------------------------

DEBUG = False

@numba.njit
def DOT(A_x, A_y, A_z, B_x, B_y, B_z):
    return ((A_x)*(B_x)+(A_y)*(B_y)+(A_z)*(B_z)) # STABLE

@numba.njit
def init_box(dim_cpu_boxes1d_arg,
             box_cpu_x,
             box_cpu_y,
             box_cpu_z,
             box_cpu_number,
             box_cpu_offset, 
             box_cpu_nn,
             box_cpu_nei_x,
             box_cpu_nei_y,
             box_cpu_nei_z,
             box_cpu_nei_number,
             box_cpu_nei_offset,
             NUMBER_PAR_PER_BOX):
  nh = 0

  # home boxes in z direction
  for i in range(dim_cpu_boxes1d_arg):
    # home boxes in y direction
    for j in range(dim_cpu_boxes1d_arg):
      # home boxes in x direction
      for k in range(dim_cpu_boxes1d_arg):
        # current home box
        box_cpu_x[nh] = k
        box_cpu_y[nh] = j
        box_cpu_z[nh] = i
        box_cpu_number[nh] = nh
        box_cpu_offset[nh] = nh * NUMBER_PAR_PER_BOX

        # initialize number of neighbor boxes
        box_cpu_nn[nh] = 0

        # neighbor boxes in z direction
        for l in range(-1, 2):
          # neighbor boxes in y direction
          for m in range(-1, 2):
            # neighbor boxes in x direction
            for n in range(-1, 2):
              # check if (this neighbor exists) and (it is not the same as home box)
              if ((((i+l)>=0 and (j+m)>=0 and (k+n)>=0) and
                   ((i+l)<dim_cpu_boxes1d_arg and 
                    (j+m)<dim_cpu_boxes1d_arg and
                    (k+n)<dim_cpu_boxes1d_arg)) and
                  not (l==0 and m==0 and n==0)):

                # current neighbor box
                box_cpu_nei_x[nh, box_cpu_nn[nh]] = (k+n)
                box_cpu_nei_y[nh, box_cpu_nn[nh]] = (j+m)
                box_cpu_nei_z[nh, box_cpu_nn[nh]] = (i+l)

                box_cpu_nei_number[nh, box_cpu_nn[nh]] = ((box_cpu_nei_z[nh, box_cpu_nn[nh]] * dim_cpu_boxes1d_arg * dim_cpu_boxes1d_arg) + 
                  (box_cpu_nei_y[nh, box_cpu_nn[nh]] * dim_cpu_boxes1d_arg) + 
                  box_cpu_nei_x[nh, box_cpu_nn[nh]])
                box_cpu_nei_offset[nh, box_cpu_nn[nh]] = box_cpu_nei_number[nh, box_cpu_nn[nh]] * NUMBER_PAR_PER_BOX

                # increment neighbor box
                box_cpu_nn[nh] += 1

        # increment home box
        nh += 1

@numba.njit
def core(dim_cpu_boxes1d_arg,
         dim_cpu_number_boxes,
         box_cpu_x,
         box_cpu_y,
         box_cpu_z,
         box_cpu_number,
         box_cpu_offset, 
         box_cpu_nn,
         box_cpu_nei_x,
         box_cpu_nei_y,
         box_cpu_nei_z,
         box_cpu_nei_number,
         box_cpu_nei_offset,
         rv_cpu_v,
         rv_cpu_x,
         rv_cpu_y,
         rv_cpu_z,
         qv_cpu,
         fv_cpu_v,
         fv_cpu_x,
         fv_cpu_y,
         fv_cpu_z,
         NUMBER_PAR_PER_BOX,
         NUMBER_THREADS,
         par_cpu_alpha):
  with openmp("""target data
                   map(to: box_cpu_x,
                           box_cpu_y,
                           box_cpu_z,
                           box_cpu_number,
                           box_cpu_offset, 
                           box_cpu_nn,
                           box_cpu_nei_x,
                           box_cpu_nei_y,
                           box_cpu_nei_z,
                           box_cpu_nei_number,
                           box_cpu_nei_offset,
                           rv_cpu_v,
                           rv_cpu_x,
                           rv_cpu_y,
                           rv_cpu_z,
                           qv_cpu)
                   map(tofrom: fv_cpu_v,
                               fv_cpu_x,
                               fv_cpu_y,
                               fv_cpu_z)"""):
    kstart = omp_get_wtime()
    with openmp("target teams num_teams(dim_cpu_number_boxes) thread_limit(NUMBER_THREADS)"):
      rA_shared_v = np.empty(100, dtype=np.float32)
      rA_shared_x = np.empty(100, dtype=np.float32)
      rA_shared_y = np.empty(100, dtype=np.float32)
      rA_shared_z = np.empty(100, dtype=np.float32)
      rB_shared_v = np.empty(100, dtype=np.float32)
      rB_shared_x = np.empty(100, dtype=np.float32)
      rB_shared_y = np.empty(100, dtype=np.float32)
      rB_shared_z = np.empty(100, dtype=np.float32)
      qB_shared = np.empty(100, dtype=np.float32)

      with openmp("parallel"):
        bx = omp_get_team_num()
        tx = omp_get_thread_num()
        wtx = tx

        #  DO FOR THE NUMBER OF BOXES

        if bx < dim_cpu_number_boxes:
          # parameters
          a2 = 2 * par_cpu_alpha * par_cpu_alpha

          # nei box
          k = 0
          j = 0

          # home box - box parameters
          first_i = box_cpu_offset[bx]

          # home box - shared memory
          while wtx < NUMBER_PAR_PER_BOX:
            rA_shared_v[wtx] = rv_cpu_v[first_i+wtx]
            rA_shared_x[wtx] = rv_cpu_x[first_i+wtx]
            rA_shared_y[wtx] = rv_cpu_y[first_i+wtx]
            rA_shared_z[wtx] = rv_cpu_z[first_i+wtx]
            wtx += NUMBER_THREADS
          wtx = tx

          with openmp("barrier"):
              pass

          # loop over nei boxes of home box
          for k in range(1 + box_cpu_nn[bx]):

            #----------------------------------------50
            #  nei box - get pointer to the right box
            #----------------------------------------50

            if k == 0:
              # set first box to be processed to home box
              pointer = bx
            else:
              # remaining boxes are nei boxes
              pointer = box_cpu_nei_number[bx, k-1]

            #  Setup parameters

            # nei box - box parameters
            first_j = box_cpu_offset[pointer]

            # (enable the section below only if wanting to use shared memory)
            # nei box - shared memory
            while wtx < NUMBER_PAR_PER_BOX:
              rB_shared_v[wtx] = rv_cpu_v[first_j+wtx]
              rB_shared_x[wtx] = rv_cpu_x[first_j+wtx]
              rB_shared_y[wtx] = rv_cpu_y[first_j+wtx]
              rB_shared_z[wtx] = rv_cpu_z[first_j+wtx]
              qB_shared[wtx] = qv_cpu[first_j+wtx]
              wtx += NUMBER_THREADS
            wtx = tx

            # (enable the section below only if wanting to use shared memory)
            # synchronize threads because in next section each thread accesses data brought in by different threads here
            with openmp("barrier"):
                pass

            #  Calculation

            # loop for the number of particles in the home box
            while wtx < NUMBER_PAR_PER_BOX:

              # loop for the number of particles in the current nei box
              for j in range(NUMBER_PAR_PER_BOX):
                r2 = rA_shared_v[wtx] + rB_shared_v[j] - DOT(rA_shared_x[wtx], rA_shared_y[wtx], rA_shared_z[wtx], rB_shared_x[j], rB_shared_y[j], rB_shared_z[j]) 
                u2 = a2*r2
                vij= math.exp(-u2)
                fs = 2*vij
                d_x = rA_shared_x[wtx] - rB_shared_x[j]
                fxij=fs*d_x
                d_y = rA_shared_y[wtx] - rB_shared_y[j]
                fyij=fs*d_y
                d_z = rA_shared_z[wtx] - rB_shared_z[j]
                fzij=fs*d_z
                fv_cpu_v[first_i+wtx] += qB_shared[j]*vij
                fv_cpu_x[first_i+wtx] += qB_shared[j]*fxij
                fv_cpu_y[first_i+wtx] += qB_shared[j]*fyij
                fv_cpu_z[first_i+wtx] += qB_shared[j]*fzij

              # increment work thread index
              wtx += NUMBER_THREADS

            # reset work index
            wtx = tx

            # synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
            with openmp("barrier"):
                pass
    kend = omp_get_wtime()

def main():
  # assing default values
  dim_cpu_arch_arg = 0
  dim_cpu_cores_arg = 1
  dim_cpu_boxes1d_arg = 1

  argc = len(sys.argv)
  argv = sys.argv

  NUMBER_PAR_PER_BOX = 100 # keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

  NUMBER_THREADS = 128 # this should be roughly equal to NUMBER_PAR_PER_BOX for best performance
  print("WG size of kernel = %d", NUMBER_THREADS)

  # go through arguments
  if argc == 3:
    dim_cpu_cur_arg = 1
    while dim_cpu_cur_arg < argc:
      # check if -boxes1d
      if argv[dim_cpu_cur_arg] == "-boxes1d":
        # check if value provided
        if argc >= dim_cpu_cur_arg + 1:
          # check if value is a number
          if argv[dim_cpu_cur_arg+1].isdigit():
            dim_cpu_boxes1d_arg = int(argv[dim_cpu_cur_arg+1])
            if dim_cpu_boxes1d_arg < 0:
              print("ERROR: Wrong value to -boxes1d argument, cannot be <=0")
              sys.exit(0)
            dim_cpu_cur_arg += 1
          else:
            # value is not a number
            print("ERROR: Value to -boxes1d argument in not a number")
            sys.exit(0)
        else:
          # value not provided
          print("ERROR: Missing value to -boxes1d argument")
          sys.exit(0)
        
        dim_cpu_cur_arg += 1
      else:
        # unknown
        print("ERROR: Unknown argument")
        sys.exit(0)

    # Print configuration
    print(f"Configuration used: arch = {dim_cpu_arch_arg}, cores = {dim_cpu_cores_arg}, boxes1d = {dim_cpu_boxes1d_arg}")
  else:
    print("Provide boxes1d argument, example: -boxes1d 16")
    sys.exit(0)

  par_cpu_alpha = 0.5

  # total number of boxes
  dim_cpu_number_boxes = dim_cpu_boxes1d_arg * dim_cpu_boxes1d_arg * dim_cpu_boxes1d_arg # 8*8*8=512

  # how many particles space has in each direction
  dim_cpu_space_elem = dim_cpu_number_boxes * NUMBER_PAR_PER_BOX # 512*100=51,200

  # allocate boxes
  box_cpu_x = np.empty(dim_cpu_number_boxes, dtype=int)
  box_cpu_y = np.empty(dim_cpu_number_boxes, dtype=int)
  box_cpu_z = np.empty(dim_cpu_number_boxes, dtype=int)
  box_cpu_number = np.empty(dim_cpu_number_boxes, dtype=int)
  box_cpu_offset = np.empty(dim_cpu_number_boxes, dtype=int)
  box_cpu_nn = np.empty(dim_cpu_number_boxes, dtype=int)

  box_cpu_nei_x = np.empty((dim_cpu_number_boxes,26), dtype=int)
  box_cpu_nei_y = np.empty((dim_cpu_number_boxes,26), dtype=int)
  box_cpu_nei_z = np.empty((dim_cpu_number_boxes,26), dtype=int)
  box_cpu_nei_number = np.empty((dim_cpu_number_boxes,26), dtype=int)
  box_cpu_nei_offset = np.empty((dim_cpu_number_boxes,26), dtype=int)

  init_box(dim_cpu_boxes1d_arg,
           box_cpu_x,
           box_cpu_y,
           box_cpu_z,
           box_cpu_number,
           box_cpu_offset, 
           box_cpu_nn,
           box_cpu_nei_x,
           box_cpu_nei_y,
           box_cpu_nei_z,
           box_cpu_nei_number,
           box_cpu_nei_offset,
           NUMBER_PAR_PER_BOX)

  #====================================================================================================100
  #  PARAMETERS, DISTANCE, CHARGE AND FORCE
  #====================================================================================================100

  # random generator seed set to random value - time in this case
  np.random.seed(2)

  # input (distances)
  rv_cpu_v = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)
  rv_cpu_x = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)
  rv_cpu_y = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)
  rv_cpu_z = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)

  # input (charge)
  qv_cpu = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)

  # output (forces)
  fv_cpu_v = np.zeros(dim_cpu_space_elem)
  fv_cpu_x = np.zeros(dim_cpu_space_elem)
  fv_cpu_y = np.zeros(dim_cpu_space_elem)
  fv_cpu_z = np.zeros(dim_cpu_space_elem)

  start = omp_get_wtime()
  core(dim_cpu_boxes1d_arg,
       dim_cpu_number_boxes,
       box_cpu_x,
       box_cpu_y,
       box_cpu_z,
       box_cpu_number,
       box_cpu_offset, 
       box_cpu_nn,
       box_cpu_nei_x,
       box_cpu_nei_y,
       box_cpu_nei_z,
       box_cpu_nei_number,
       box_cpu_nei_offset,
       rv_cpu_v,
       rv_cpu_x,
       rv_cpu_y,
       rv_cpu_z,
       qv_cpu,
       fv_cpu_v,
       fv_cpu_x,
       fv_cpu_y,
       fv_cpu_z,
       NUMBER_PAR_PER_BOX,
       NUMBER_THREADS,
       par_cpu_alpha)
  end = omp_get_wtime()
  print("Device offloading time:", float(end-start) / 1000000)
  print("Kernel execution time:", float(kend-kstart) / 1000000)

  if DEBUG:
    offset = 395
    for g in range(10):
      print("g=", g, fv_cpu_v[offset+g], fv_cpu_x[offset+g], fv_cpu_y[offset+g], fv_cpu_z[offset+g])

  sys.exit(0)

if __name__ == "__main__":
  main()

