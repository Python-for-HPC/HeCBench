from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import (
    omp_get_wtime,
    omp_get_num_threads,
    omp_get_thread_num,
    omp_get_team_num,
)
import math
import numpy as np
from functools import reduce
from kernel import fasten_main
import argparse

DEFAULT_ITERS = np.int32(8)
DEFAULT_NPOSES = np.int32(65536)
REF_NPOSES = np.int32(65536)
DEFAULT_PPWI = np.int32(1)
DEFAULT_WGSIZE = np.int32(4)
NUM_TD_PER_THREAD = np.int32(DEFAULT_PPWI)
DATA_DIR = "../../../miniBUDE/data/bm1"
FILE_LIGAND = "/ligand.in"
FILE_PROTEIN = "/protein.in"
FILE_FORCEFIELD = "/forcefield.in"
FILE_POSES = "/poses.in"
FILE_REF_ENERGIES = "/ref_energies.out"

def compile():
  import time
  t1 = time.perf_counter()
  runKernelInternal.compile(""" (Array(float32, 2, 'C', False, aligned=True),
  Array(int32, 1, 'C', False, aligned=True), Array(float32, 2, 'C', False, aligned=True),
  Array(int32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True),
  Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True),
  Array(float32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True),
  Array(float32, 1, 'C', False, aligned=True), Array(float32, 2, 'C', False, aligned=True),
  Array(int32, 1, 'C', False, aligned=True), Array(float32, 1, 'C', False, aligned=True),
  int32, int32, int32, int32, int32, int32) """)
  t2 = time.perf_counter()
  print("ctime", t2 - t1, "s")

class Params:
    def __init__(self):
        self.natlig = np.int32(0)
        self.natpro = np.int32(0)
        self.ntypes = np.int32(0)
        self.nposes = np.int32(0)
        self.protein_xyz = None
        self.protein_type = None
        self.ligand_xyz = None
        self.ligand_type = None
        self.forcefield_hbtype = None
        self.forcefield_rhe = None
        self.poses = None
        self.iterations = np.int32(0)
        self.wgSize = np.int32(0)
        self.deckDir = ""

    def __str__(self):
        return (
            "natlig:      "
            + str(self.natlig)
            + "\n"
            + "natpro:      "
            + str(self.natpro)
            + "\n"
            + "ntypes:      "
            + str(self.ntypes)
            + "\n"
            + "nposes:      "
            + str(self.nposes)
            + "\n"
            + "iterations:  "
            + str(self.iterations)
            + "\n"
            + "posesPerWI:  "
            + str(NUM_TD_PER_THREAD)
            + "\n"
            + "wgSize:      "
            + str(self.wgSize)
            + "\n"
        )


def elapsedMillis(start, end):
    return (end - start) * 1000


def printTimings(params, millis):
    ms = millis / params.iterations
    runtime = ms * 1e-3
    ops_per_wg = (
        NUM_TD_PER_THREAD * 27
        + params.natlig
        * (3 + NUM_TD_PER_THREAD * 18 + params.natpro * (11 + NUM_TD_PER_THREAD * 30))
        + NUM_TD_PER_THREAD
    )
    total_ops = ops_per_wg * (float(params.nposes) / NUM_TD_PER_THREAD)
    flops = total_ops / runtime
    gflops = flops / 1e9
    interactions = float(params.nposes) * float(params.natlig) * float(params.natpro)
    interactions_per_sec = interactions / runtime
    print(f"- Total kernel time: {millis:.3f} ms")
    print(f"- Average kernel time: {ms:.3f} ms")
    print(f"- Interactions/s: {interactions_per_sec / 1e9:.3f} billion")
    print(f"- GFLOP/s:        {gflops:.3f}")


def readAtom(path):
    with open(path, "rb") as binary_file:
        # Initialize empty lists for floats and integers
        float_list = []
        int_list = []

        while True:
            # Read the first three floats
            float_data = np.fromfile(binary_file, dtype=np.float32, count=3)
            if not float_data.size:
                break  # Reached end of file

            # Read the integer
            int_data = np.fromfile(binary_file, dtype=np.int32, count=1)

            # Append to respective lists
            float_list.append(float_data)
            int_list.append(int_data)

    # Convert lists to numpy arrays
    float_array = np.array(float_list, dtype=np.float32)
    int_array = np.squeeze(np.array(int_list, dtype=np.int32))

    return float_array, int_array


def readFF(path):
    with open(path, "rb") as binary_file:
        # Initialize empty lists for floats and integers
        float_list = []
        int_list = []

        while True:
            # Read the integer
            int_data = np.fromfile(binary_file, dtype=np.int32, count=1)
            if not int_data.size:
                break  # Reached end of file

            # Read the first three floats
            float_data = np.fromfile(binary_file, dtype=np.float32, count=3)

            # Append to respective lists
            int_list.append(int_data)
            float_list.append(float_data)

    # Convert lists to numpy arrays
    float_array = np.array(float_list, dtype=np.float32)
    int_array = np.squeeze(np.array(int_list, np.int32))

    return float_array, int_array


def readPoses(path):
    with open(path, "rb") as binary_file:
        float_array = np.fromfile(binary_file, dtype=np.float32)
    # print("float_array:", float_array)
    falen = float_array.shape[0]
    assert falen % 6 == 0
    falen //= 6

    # float_array = float_array.reshape((-1, 6))
    new_float_array = np.empty((falen, 6), dtype=float_array.dtype)
    for i in range(6):
        new_float_array[:, i] = float_array[i * falen : (i + 1) * falen]
        # new_float_array[:, i] = float_array[i::6]
    # print("new_float_array:", new_float_array)
    return new_float_array


def loadParameters():
    params = Params()
    params.iterations = DEFAULT_ITERS
    params.nposes = DEFAULT_NPOSES
    params.wgSize = DEFAULT_WGSIZE
    params.deckDir = DATA_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iterations",
        help=f"Repeat kernel I times (default: {DEFAULT_ITERS})",
        type=int,
        default=DEFAULT_ITERS,
    )
    parser.add_argument(
        "-n",
        "--numposes",
        help=f"Compute energies for N poses (default: {DEFAULT_NPOSES}",
        type=int,
        default=DEFAULT_NPOSES,
    )
    parser.add_argument(
        "-w",
        "--wgsize",
        help=f"Run with work-group size WGSIZE using nd_range, set to 0 for plain range (default {DEFAULT_WGSIZE}",
        type=int,
        default=DEFAULT_WGSIZE,
    )
    parser.add_argument(
        "--deck",
        help=f"Use the DECK directory as input deck (default: {DATA_DIR}",
        default=DATA_DIR,
    )
    args = parser.parse_args()
    params.iterations = args.iterations
    params.numposes = args.numposes
    params.wgSize = args.wgsize
    params.deckDir = args.deck

    params.ligand_xyz, params.ligand_type = readAtom(params.deckDir + FILE_LIGAND)
    params.natlig = len(params.ligand_type)
    params.protein_xyz, params.protein_type = readAtom(params.deckDir + FILE_PROTEIN)
    params.natpro = len(params.protein_type)
    params.forcefield_rhe, params.forcefield_hbtype = readFF(
        params.deckDir + FILE_FORCEFIELD
    )
    params.ntypes = len(params.forcefield_hbtype)
    params.poses = readPoses(params.deckDir + FILE_POSES)
    if params.poses.shape[0] != params.nposes:
        raise ValueError("Bad poses: " + str(len(poses)))
    # for i in range(6):
    #    params.poses[i] = poses[i * params.nposes : (i + 1) * params.nposes]
    return params


@njit
def runKernelInternal(
    protein_xyz,
    protein_type,
    ligand_xyz,
    ligand_type,
    transforms_0,
    transforms_1,
    transforms_2,
    transforms_3,
    transforms_4,
    transforms_5,
    forcefield_rhe,
    forcefield_hbtype,
    results,
    nposes,
    wgSize,
    ntypes,
    natlig,
    natpro,
    iterations,
):
    with openmp(
        """target data
        map(to: protein_xyz, protein_type, ligand_xyz, ligand_type,
        transforms_0, transforms_1, transforms_2, transforms_3, transforms_4,
        transforms_5, forcefield_rhe, forcefield_hbtype)
        map(from: results) device(1)"""
    ):
        globalo = math.ceil((nposes / NUM_TD_PER_THREAD))
        teams = math.ceil(globalo / wgSize)
        block = wgSize

        # +1 iteration for warmup.
        for i in range(iterations+1):
          # 0-th interation is warmup, timestamp on iteration 1.
          if i == 1:
            kernelStart = omp_get_wtime()

          # NOTE: We need to set launch bounds because minibude requires 256
          # computation threads for correct calculation. An extra complication is
          # that the OpenMP target runtime sets aside 1 warp (32 threads) for
          # sequential execution in GENERIC execution mode (not SIMD), so we must set
          # launch bounds to 256+32 to have 256 COMPUTATION threads that are
          # necessary for minibude.
          with openmp(
              """target teams num_teams(teams) thread_limit(block) device(1) ompx_attribute(launch_bounds(288))"""
          ):
              local_forcefield_rhe = np.empty((64, 3), dtype=np.float32)
              local_forcefield_hbtype = np.empty(64, dtype=np.int32)

              with openmp("parallel"):
                  # Constants cast to float32 for single-precision arithmetic.
                  ZERO = np.float32(0.0)
                  QUARTER = np.float32(0.25)
                  HALF = np.float32(0.5)
                  ONE = np.float32(1.0)
                  NEGONE = np.float32(-1.0)
                  TWO = np.float32(2.0)
                  FOUR = np.float32(4.0)
                  CNSTNT = np.float32(45.0)
                  HARDNESS = np.float32(38.0)
                  NPNDIST = np.float32(5.5)
                  FLT_MAX = np.float32(3.402823466e38)

                  lid = omp_get_thread_num()
                  gid = omp_get_team_num()
                  lrange = omp_get_num_threads()

                  etot = np.empty(1, dtype=np.float32)
                  lpos = np.empty((1, 3), dtype=np.float32)
                  transform = np.empty((1, 3, 4), dtype=np.float32)

                  ix = np.int32(gid * lrange * NUM_TD_PER_THREAD + lid)
                  ix = ix if (ix < nposes) else np.int32(nposes - NUM_TD_PER_THREAD)

                  for i in range(lid, ntypes, lrange):
                      # local_forcefield_rhe[i, :] = forcefield_rhe[i, :] # See numbaWithOpenmp issue #23.
                      local_forcefield_rhe[i, 0] = forcefield_rhe[i, 0]
                      local_forcefield_rhe[i, 1] = forcefield_rhe[i, 1]
                      local_forcefield_rhe[i, 2] = forcefield_rhe[i, 2]
                      local_forcefield_hbtype[i] = forcefield_hbtype[i]

                  for i in range(NUM_TD_PER_THREAD):
                      index = ix + i * lrange
                      sx = math.sin(transforms_0[index])
                      cx = math.cos(transforms_0[index])
                      sy = math.sin(transforms_1[index])
                      cy = math.cos(transforms_1[index])
                      sz = math.sin(transforms_2[index])
                      cz = math.cos(transforms_2[index])
                      transform[i, 0, 0] = cy * cz
                      transform[i, 0, 1] = sx * sy * cz - cx * sz
                      transform[i, 0, 2] = cx * sy * cz + sx * sz
                      transform[i, 0, 3] = transforms_3[index]
                      transform[i, 1, 0] = cy * sz
                      transform[i, 1, 1] = sx * sy * sz + cx * cz
                      transform[i, 1, 2] = cx * sy * sz - sx * cz
                      transform[i, 1, 3] = transforms_4[index]
                      transform[i, 2, 0] = sy * NEGONE
                      transform[i, 2, 1] = sx * cy
                      transform[i, 2, 2] = cx * cy
                      transform[i, 2, 3] = transforms_5[index]
                      etot[i] = ZERO

                  with openmp("""barrier"""):
                      pass

                  for il in range(natlig):
                      lfindex = ligand_type[il]
                      l_params_rhe = local_forcefield_rhe[lfindex]
                      l_params_hbtype = local_forcefield_hbtype[lfindex]
                      lhphb_ltz = l_params_rhe[1] < ZERO
                      lhphb_gtz = l_params_rhe[1] > ZERO
                      linitpos = ligand_xyz[il]

                      for i in range(NUM_TD_PER_THREAD):
                          lpos[i, 0] = (
                              transform[i, 0, 3]
                              + linitpos[0] * transform[i, 0, 0]
                              + linitpos[1] * transform[i, 0, 1]
                              + linitpos[2] * transform[i, 0, 2]
                          )
                          lpos[i, 1] = (
                              transform[i, 1, 3]
                              + linitpos[0] * transform[i, 1, 0]
                              + linitpos[1] * transform[i, 1, 1]
                              + linitpos[2] * transform[i, 1, 2]
                          )
                          lpos[i, 2] = (
                              transform[i, 2, 3]
                              + linitpos[0] * transform[i, 2, 0]
                              + linitpos[1] * transform[i, 2, 1]
                              + linitpos[2] * transform[i, 2, 2]
                          )

                      for ip in range(natpro):
                          p_atom_xyz = protein_xyz[ip]
                          p_atom_type = protein_type[ip]
                          p_params_rhe = local_forcefield_rhe[p_atom_type]
                          p_params_hbtype = local_forcefield_hbtype[p_atom_type]
                          radij = p_params_rhe[0] + l_params_rhe[0]
                          r_radij = ONE / radij
                          elcdst = (
                              FOUR if p_params_hbtype == 70 and l_params_hbtype == 70 else TWO
                          )
                          elcdst1 = (
                              QUARTER
                              if p_params_hbtype == 70 and l_params_hbtype == 70
                              else HALF
                          )
                          type_E = p_params_hbtype == 69 or l_params_hbtype == 69
                          phphb_ltz = p_params_rhe[1] < ZERO
                          phphb_gtz = p_params_rhe[1] > ZERO
                          phphb_nz = p_params_rhe[1] != ZERO
                          p_hphb = (
                              (p_params_rhe[1] * NEGONE)
                              if phphb_ltz and lhphb_gtz
                              else (p_params_rhe[1] * ONE)
                          )
                          l_hphb = (
                              (l_params_rhe[1] * NEGONE)
                              if phphb_gtz and lhphb_ltz
                              else (l_params_rhe[1] * ONE)
                          )
                          distdslv = (
                              (np.float32(NPNDIST) if lhphb_ltz else ONE)
                              if phphb_ltz
                              else (ONE if lhphb_ltz else (FLT_MAX * NEGONE))
                          )
                          r_distdslv = ONE / distdslv
                          chrg_init = l_params_rhe[2] * p_params_rhe[2]
                          dslv_init = p_hphb + l_hphb

                          for i in range(NUM_TD_PER_THREAD):
                              x = lpos[i, 0] - p_atom_xyz[0]
                              y = lpos[i, 1] - p_atom_xyz[1]
                              z = lpos[i, 2] - p_atom_xyz[2]
                              distij = math.sqrt(x * x + y * y + z * z)
                              distbb = distij - radij
                              zone1 = distbb < ZERO
                              etot[i] += (ONE - (distij * r_radij)) * (
                                  TWO * HARDNESS if zone1 else ZERO
                              )
                              chrg_e = chrg_init * (
                                  (ONE if zone1 else (ONE - distbb * elcdst1))
                                  * (ONE if distbb < elcdst else ZERO)
                              )
                              neg_chrg_e = abs(chrg_e) * NEGONE
                              chrg_e = neg_chrg_e if type_E else chrg_e
                              etot[i] += chrg_e * CNSTNT
                              coeff = ONE - (distbb * r_distdslv)
                              dslv_e = dslv_init * (
                                  ONE if ((distbb < distdslv) and phphb_nz) else ZERO
                              )
                              dslv_e *= ONE if zone1 else coeff
                              etot[i] += dslv_e

                  td_base = gid * lrange * NUM_TD_PER_THREAD + lid
                  if td_base < nposes:
                      for i in range(NUM_TD_PER_THREAD):
                          results[td_base + i * lrange] = etot[i] * HALF
        kernelEnd = omp_get_wtime()
    return kernelStart, kernelEnd


def runKernel(params):
    energies = np.empty(params.nposes, dtype=np.float32)
    protein_xyz = params.protein_xyz
    protein_type = params.protein_type
    ligand_xyz = params.ligand_xyz
    ligand_type = params.ligand_type

    transforms_0 = np.empty(params.poses.shape[0], dtype=np.float32)
    transforms_1 = np.empty(params.poses.shape[0], dtype=np.float32)
    transforms_2 = np.empty(params.poses.shape[0], dtype=np.float32)
    transforms_3 = np.empty(params.poses.shape[0], dtype=np.float32)
    transforms_4 = np.empty(params.poses.shape[0], dtype=np.float32)
    transforms_5 = np.empty(params.poses.shape[0], dtype=np.float32)
    transforms_0[:] = params.poses[:, 0]
    transforms_1[:] = params.poses[:, 1]
    transforms_2[:] = params.poses[:, 2]
    transforms_3[:] = params.poses[:, 3]
    transforms_4[:] = params.poses[:, 4]
    transforms_5[:] = params.poses[:, 5]
    forcefield_rhe = params.forcefield_rhe
    forcefield_hbtype = params.forcefield_hbtype
    nposes = params.nposes

    compile()
    kernelStart, kernelEnd = runKernelInternal(
        protein_xyz,
        protein_type,
        ligand_xyz,
        ligand_type,
        transforms_0,
        transforms_1,
        transforms_2,
        transforms_3,
        transforms_4,
        transforms_5,
        forcefield_rhe,
        forcefield_hbtype,
        energies,
        np.int32(nposes),
        np.int32(params.wgSize),
        np.int32(params.ntypes),
        np.int32(params.natlig),
        np.int32(params.natpro),
        np.int32(params.iterations),
    )

    printTimings(params, elapsedMillis(kernelStart, kernelEnd))
    return energies


def main():
  params = loadParameters()
  print("Poses     : " + str(params.nposes))
  print("Iterations: " + str(params.iterations))
  print("Ligands   : " + str(params.natlig))
  print("Proteins  : " + str(params.natpro))
  print("Deck      : " + params.deckDir)
  print("Types     : " + str(params.ntypes))
  print("WG        : " + str(params.wgSize))
  energies = runKernel(params)

  # Validate energies
  if params.nposes > REF_NPOSES:
      print("Only validating the first {REF_NPOSES} poses.")
  nRefPoses = REF_NPOSES

  maxdiff = np.float32(0.0)
  with open(params.deckDir + FILE_REF_ENERGIES) as f:
      for i in range(nRefPoses):
          e = np.fromfile(f, dtype=np.float32, count=1, sep="\n")
          e = np.float32(e[0])

          if abs(e - energies[i]) >= np.float32(0.01):
              print(f"ERROR e {e} energies[{i}] {energies[i]}")
              break
          if abs(e) < np.float32(1.0) and abs(energies[i]) < np.float32(1.0):
              continue

          diff = abs(e - energies[i]) / e
          if diff > maxdiff:
              maxdiff = diff

  print(f"Largest difference was {100*maxdiff:.3f}%.")


if __name__ == "__main__":
  main()
