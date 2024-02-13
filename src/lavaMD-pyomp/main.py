from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device, omp_get_team_num, omp_get_thread_num, omp_get_num_teams
import numba
import numpy as np
import sys
import math

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

  # TODO this is a workaround to allocate from global memory scaled to number
  # of teams, those arrays should be in shared memory for optimization
  # NOTE: 100 = NUMBER_PAR_PER_BOX
  rA_shared_v = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)
  rA_shared_x = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)
  rA_shared_y = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)
  rA_shared_z = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)
  rB_shared_v = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)
  rB_shared_x = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)
  rB_shared_y = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)
  rB_shared_z = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)
  qB_shared = np.empty((dim_cpu_number_boxes,100), dtype=np.float32)

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

                   map(tofrom:
                           fv_cpu_v,
                           fv_cpu_x,
                           fv_cpu_y,
                           fv_cpu_z)

                    map(alloc:
                          rA_shared_v,
                          rA_shared_x,
                          rA_shared_y,
                          rA_shared_z,
                          rB_shared_v,
                          rB_shared_x,
                          rB_shared_y,
                          rB_shared_z,
                          qB_shared)

                           device(1)"""):
      kstart = omp_get_wtime()
      with openmp("target teams num_teams(dim_cpu_number_boxes) thread_limit(NUMBER_THREADS) device(1)"):
        # TODO: allocate using numba cuda interface when in CUDA compilation,
        # needs fallback for host code.
        #rA_shared_v = numba.cuda.shared.array(100, dtype=np.float32)
        #rA_shared_x = numba.cuda.shared.array(100, dtype=np.float32)
        #rA_shared_y = numba.cuda.shared.array(100, dtype=np.float32)
        #rA_shared_z = numba.cuda.shared.array(100, dtype=np.float32)
        #rB_shared_v = numba.cuda.shared.array(100, dtype=np.float32)
        #rB_shared_x = numba.cuda.shared.array(100, dtype=np.float32)
        #rB_shared_y = numba.cuda.shared.array(100, dtype=np.float32)
        #rB_shared_z = numba.cuda.shared.array(100, dtype=np.float32)
        #qB_shared = numba.cuda.shared.array(100, dtype=np.float32)

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
              rA_shared_v[bx,wtx] = rv_cpu_v[first_i+wtx]
              rA_shared_x[bx,wtx] = rv_cpu_x[first_i+wtx]
              rA_shared_y[bx,wtx] = rv_cpu_y[first_i+wtx]
              rA_shared_z[bx,wtx] = rv_cpu_z[first_i+wtx]
              wtx += NUMBER_THREADS
            wtx = tx

            with openmp("barrier"):
                pass
            # loop over nei boxes of home box
            #if tx == 0:
            #  print("line 189", "bx", bx, "tx", tx, "range", box_cpu_nn[bx])
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
                rB_shared_v[bx,wtx] = rv_cpu_v[first_j+wtx]
                rB_shared_x[bx,wtx] = rv_cpu_x[first_j+wtx]
                rB_shared_y[bx,wtx] = rv_cpu_y[first_j+wtx]
                rB_shared_z[bx,wtx] = rv_cpu_z[first_j+wtx]
                qB_shared[bx,wtx] = qv_cpu[first_j+wtx]
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
                  r2 = rA_shared_v[bx,wtx] + rB_shared_v[bx,j] - DOT(rA_shared_x[bx,wtx], rA_shared_y[bx,wtx], rA_shared_z[bx,wtx], rB_shared_x[bx,j], rB_shared_y[bx,j], rB_shared_z[bx,j])
                  u2 = a2*r2
                  u2 = -1.0*u2
                  vij= math.exp(u2)
                  fs = 2*vij
                  d_x = rA_shared_x[bx,wtx] - rB_shared_x[bx,j]
                  fxij=fs*d_x
                  d_y = rA_shared_y[bx,wtx] - rB_shared_y[bx,j]
                  fyij=fs*d_y
                  d_z = rA_shared_z[bx,wtx] - rB_shared_z[bx,j]
                  fzij=fs*d_z
                  fv_cpu_v[first_i+wtx] += qB_shared[bx,j]*vij
                  fv_cpu_x[first_i+wtx] += qB_shared[bx,j]*fxij
                  fv_cpu_y[first_i+wtx] += qB_shared[bx,j]*fyij
                  fv_cpu_z[first_i+wtx] += qB_shared[bx,j]*fzij

                # increment work thread index
                wtx += NUMBER_THREADS

              # reset work index
              wtx = tx

              # synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
              with openmp("barrier"):
                  pass

      kend = omp_get_wtime()
      print("Kernel execution time: ", float(kend-kstart), "s")


def main():
  # assing default values
  dim_cpu_arch_arg = 0
  dim_cpu_cores_arg = 1
  dim_cpu_boxes1d_arg = 1

  argc = len(sys.argv)
  argv = sys.argv

  NUMBER_PAR_PER_BOX = 100 # keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

  NUMBER_THREADS = 128 # this should be roughly equal to NUMBER_PAR_PER_BOX for best performance
  print("WG size of kernel =", NUMBER_THREADS)

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
  #rv_cpu_v = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)
  #rv_cpu_x = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)
  #rv_cpu_y = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)
  #rv_cpu_z = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)
  rv_cpu_v = np.full(dim_cpu_space_elem, 0.1, dtype=np.float32)
  rv_cpu_x = np.full(dim_cpu_space_elem, 0.2, dtype=np.float32)
  rv_cpu_y = np.full(dim_cpu_space_elem, 0.3, dtype=np.float32)
  rv_cpu_z = np.full(dim_cpu_space_elem, 0.4, dtype=np.float32)

  # input (charge)
  #qv_cpu = np.random.uniform(0.1, 1.0, size=dim_cpu_space_elem)
  qv_cpu = np.full(dim_cpu_space_elem, 0.5, dtype=np.float32)

  # output (forces)
  fv_cpu_v = np.zeros(dim_cpu_space_elem, dtype=np.float32)
  fv_cpu_x = np.zeros(dim_cpu_space_elem, dtype=np.float32)
  fv_cpu_y = np.zeros(dim_cpu_space_elem, dtype=np.float32)
  fv_cpu_z = np.zeros(dim_cpu_space_elem, dtype=np.float32)

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
  print("Device offloading time: ", float(end-start), "s")

  if DEBUG:
    offset = 395
    for g in range(10):
      print(f"g= {g} {fv_cpu_v[offset+g]:.6f} {fv_cpu_x[offset+g]:.6f} {fv_cpu_y[offset+g]:.6f} {fv_cpu_z[offset+g]:.6f}")

  sys.exit(0)

if __name__ == "__main__":
  main()

