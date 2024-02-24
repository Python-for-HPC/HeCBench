from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device
import numba
import math
import numpy as np
import time
import sys
import os
import operator
from functools import reduce
from kernel import fasten_main

DEFAULT_ITERS = 8
DEFAULT_NPOSES = 65536
DEFAULT_PPWI = 1
DEFAULT_WGSIZE = 4
NUM_TD_PER_THREAD = DEFAULT_PPWI
DATA_DIR = "../../../miniBUDE/data/bm1"
FILE_LIGAND = "/ligand.in"
FILE_PROTEIN = "/protein.in"
FILE_FORCEFIELD = "/forcefield.in"
FILE_POSES = "/poses.in"
FILE_REF_ENERGIES = "/ref_energies.out"
FLT_MAX = 3.402823466e+38

class Params:
    def __init__(self):
        self.natlig = 0
        self.natpro = 0
        self.ntypes = 0
        self.nposes = 0
        self.protein_xyz = None
        self.protein_type = None
        self.ligand_xyz = None
        self.ligand_type = None
        self.forcefield_hbtype = None
        self.forcefield_rhe = None
        self.poses = None
        self.iterations = 0
        self.wgSize = 0
        self.deckDir = ""

    def __str__(self):
        return "natlig:      " + str(self.natlig) + "\n" + \
               "natpro:      " + str(self.natpro) + "\n" + \
               "ntypes:      " + str(self.ntypes) + "\n" + \
               "nposes:      " + str(self.nposes) + "\n" + \
               "iterations:  " + str(self.iterations) + "\n" + \
               "posesPerWI:  " + str(NUM_TD_PER_THREAD) + "\n" + \
               "wgSize:      " + str(self.wgSize) + "\n"

def elapsedMillis(start, end):
    return (end - start) * 1000

def printTimings(params, millis):
    ms = millis / params.iterations
    runtime = ms * 1e-3
    ops_per_wg = NUM_TD_PER_THREAD * 27 + params.natlig * (3 + NUM_TD_PER_THREAD * 18 + params.natpro * (11 + NUM_TD_PER_THREAD * 30)) + NUM_TD_PER_THREAD
    total_ops = ops_per_wg * (float(params.nposes) / NUM_TD_PER_THREAD)
    flops = total_ops / runtime
    gflops = flops / 1e9
    interactions = float(params.nposes) * float(params.natlig) * float(params.natpro)
    interactions_per_sec = interactions / runtime
    print("- Total kernel time:    " + str(millis) + " ms")
    print("- Average kernel time:   " + str(ms) + " ms")
    print("- Interactions/s: " + str(interactions_per_sec / 1e9) + " billion")
    print("- GFLOP/s:        " + str(gflops))

def readAtom(path):
    with open(path, 'rb') as binary_file:
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
    float_array = np.array(float_list)
    int_array = np.squeeze(np.array(int_list))

    return float_array, int_array

def readFF(path):
    with open(path, 'rb') as binary_file:
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
    float_array = np.array(float_list)
    int_array = np.squeeze(np.array(int_list))

    return float_array, int_array

def readPoses(path):
    with open(path, 'rb') as binary_file:
        float_array = np.fromfile(binary_file, dtype=np.float32)
    float_array = float_array.reshape((-1, 6))
    return float_array

def loadParameters(args):
    params = Params()
    params.iterations = DEFAULT_ITERS
    params.nposes = DEFAULT_NPOSES
    params.wgSize = DEFAULT_WGSIZE
    params.deckDir = DATA_DIR

    def readParam(current, arg, matches, handle):
        if len(matches) == 0:
            return False
        if arg in matches:
            if current + 1 < len(args):
                current += 1
                handle(args[current])
            else:
                print("[" + ", ".join(matches) + "] specified but no value was given")
                sys.exit(EXIT_FAILURE)
            return True
        return False

    def bindInt(param, dest, name):
        try:
            parsed = int(param)
            if parsed < 0:
                print("positive integer required for <" + name + ">: `" + str(parsed) + "`")
                sys.exit(EXIT_FAILURE)
            dest = parsed
        except:
            print("malformed value, integer required for <" + name + ">: `" + param + "`")
            sys.exit(EXIT_FAILURE)

    for i in range(len(args)):
        arg = args[i]
        if readParam(i, arg, ["--iterations", "-i"], lambda param: bindInt(param, params.iterations, "iterations")):
            continue
        if readParam(i, arg, ["--numposes", "-n"], lambda param: bindInt(param, params.nposes, "numposes")):
            continue
        if readParam(i, arg, ["--wgsize", "-w"], lambda param: bindInt(param, params.wgSize, "wgsize")):
            continue
        if readParam(i, arg, ["--deck"], lambda param: setattr(params, "deckDir", param)):
            continue
        if arg == "--help" or arg == "-h":
            print("\n")
            print("Usage: ./main [OPTIONS]\n\n" +
                  "Options:\n" +
                  "  -h  --help               Print this message\n" +
                  "  -i  --iterations I       Repeat kernel I times (default: " + str(DEFAULT_ITERS) + ")\n" +
                  "  -n  --numposes   N       Compute energies for N poses (default: " + str(DEFAULT_NPOSES) + ")\n" +
                  "  -w  --wgsize     WGSIZE  Run with work-group size WGSIZE using nd_range, set to 0 for plain range (default: " + str(DEFAULT_WGSIZE) + ")\n" +
                  "      --deck       DECK    Use the DECK directory as input deck (default: " + DATA_DIR + ")")
            sys.exit(EXIT_SUCCESS)
        print("Unrecognized argument '" + arg + "' (try '--help')")
        sys.exit(EXIT_FAILURE)

    params.ligand_xyz, params.ligand_type = readAtom(params.deckDir + FILE_LIGAND)
    params.natlig = len(params.ligand_type)
    params.protein_xyz, params.protein_type = readAtom(params.deckDir + FILE_PROTEIN)
    params.natpro = len(params.protein_type)
    params.forcefield_rhe, params.forcefield_hbtype = readFF(params.deckDir + FILE_FORCEFIELD)
    params.ntypes = len(params.forcefield_hbtype)
    params.poses = readPoses(params.deckDir + FILE_POSES)
    if params.poses.shape[0] != params.nposes:
        raise ValueError("Bad poses: " + str(len(poses)))
    #for i in range(6):
    #    params.poses[i] = poses[i * params.nposes : (i + 1) * params.nposes]
    return params

def runKernelInternal(protein_xyz,
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
                      FLT_MAX):

    with openmp("""target data map(to: protein_xyz, protein_type, transforms_0, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5, forcefield_rhe, forcefield_hbtype) map(from: results, kernelStart, kernelEnd)"""):
        globalo = math.ceil((nposes / NUM_TD_PER_THREAD))
        teams = math.ceil(globalo / wgSize)
        block = wgSize

        fasten_main(teams,
                    block,
                    ntypes,
                    nposes,
                    natlig,
                    natpro,
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
                    NUM_TD_PER_THREAD,
                    FLT_MAX)

        kernelStart = omp_get_wtime()
        for i in range(iterations):
            fasten_main(teams,
                        block,
                        ntypes,
                        nposes,
                        natlig,
                        natpro,
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
                        NUM_TD_PER_THREAD,
                        FLT_MAX)
        kernelEnd = omp_get_wtime()
    return kernelStart, kernelEnd

def runKernel(params):
    energies = np.zeros(params.nposes)
    protein_xyz = params.protein_xyz
    protein_type = params.protein_type
    ligand_xyz = params.ligand_xyz
    ligand_type = params.ligand_type
    transforms_0 = params.poses[:, 0]
    transforms_1 = params.poses[:, 1]
    transforms_2 = params.poses[:, 2]
    transforms_3 = params.poses[:, 3]
    transforms_4 = params.poses[:, 4]
    transforms_5 = params.poses[:, 5]
    forcefield_rhe = params.forcefield_rhe
    forcefield_hbtype = params.forcefield_hbtype
    nposes = params.nposes

    kernelStart, kernelEnd = runKernelInternal(protein_xyz,
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
                                               nposes,
                                               params.wgSize,
                                               params.ntypes,
                                               params.natlig,
                                               params.natpro,
                                               params.iterations,
                                               FLT_MAX)

    printTimings(params, elapsedMillis(kernelStart, kernelEnd))
    return energies

args = sys.argv[1:]
params = loadParameters(args)
print("Poses     : " + str(params.nposes))
print("Iterations: " + str(params.iterations))
print("Ligands   : " + str(params.natlig))
print("Proteins  : " + str(params.natpro))
print("Deck      : " + params.deckDir)
print("Types     : " + str(params.ntypes))
print("WG        : " + str(params.wgSize))
energies = runKernel(params)
