from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import (
    omp_get_wtime,
    omp_set_num_threads,
    omp_get_num_threads,
    omp_get_num_devices,
    omp_is_initial_device,
)
import numba
import math
import numpy as np
import time
import sys
import os
import operator
from functools import reduce
from kernel import fasten_main
import argparse

DEFAULT_ITERS = 8
DEFAULT_NPOSES = 65536
REF_NPOSES = 65536
DEFAULT_PPWI = 1
DEFAULT_WGSIZE = 4
NUM_TD_PER_THREAD = DEFAULT_PPWI
DATA_DIR = "../../../miniBUDE/data/bm1"
FILE_LIGAND = "/ligand.in"
FILE_PROTEIN = "/protein.in"
FILE_FORCEFIELD = "/forcefield.in"
FILE_POSES = "/poses.in"
FILE_REF_ENERGIES = "/ref_energies.out"
FLT_MAX = 3.402823466e38


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
    float_array = np.array(float_list)
    int_array = np.squeeze(np.array(int_list))

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
    float_array = np.array(float_list)
    int_array = np.squeeze(np.array(int_list))

    return float_array, int_array


def readPoses(path):
    with open(path, "rb") as binary_file:
        float_array = np.fromfile(binary_file, dtype=np.float32)
    float_array = float_array.reshape((-1, 6))
    return float_array


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
    FLT_MAX,
):

    with openmp(
        """target data
        map(to: protein_xyz, protein_type, transforms_0, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5, forcefield_rhe, forcefield_hbtype)
        map(from: results) device(1)"""
    ):
        globalo = math.ceil((nposes / NUM_TD_PER_THREAD))
        teams = math.ceil(globalo / wgSize)
        block = wgSize

        fasten_main(
            teams,
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
            FLT_MAX,
        )

        kernelStart = omp_get_wtime()
        for i in range(iterations):
            fasten_main(
                teams,
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
                FLT_MAX,
            )
        kernelEnd = omp_get_wtime()
    print("kernelStart", kernelStart)
    print("kernelEnd", kernelEnd)
    return kernelStart, kernelEnd


def runKernel(params):
    energies = np.zeros(params.nposes)
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
        nposes,
        params.wgSize,
        params.ntypes,
        params.natlig,
        params.natpro,
        params.iterations,
        FLT_MAX,
    )

    printTimings(params, elapsedMillis(kernelStart, kernelEnd))
    return energies


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
    print("Only validing the first {REF_NPOSES} poses.")
nRefPoses = REF_NPOSES

maxdiff = np.float32(0.0)
with open(params.deckDir + FILE_REF_ENERGIES) as f:
    for i in range(nRefPoses):
        e = np.fromfile(f, dtype=np.float32, count=1, sep="\n")
        e = e[0]

        print(f"e {e} energies[{i}] {energies[i]}")
        input("k")
        if abs(e) < 0.1 and abs(energies[i]) < 0.1:
            continue

        diff = abs(e - energies[i]) / e
        if diff > maxdiff:
            maxdiff = diff

print(f"Largest difference was {100*maxdiff:.3f}%.")
