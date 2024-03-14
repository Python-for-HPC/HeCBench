#
#   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#

import sys
import numpy as np
from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime


#
#  Reference CPU implementation of FloydWarshall PathFinding
#  for performance comparison
#  @param pathDistanceMatrix Distance between nodes of a graph
#  @param intermediate node between two nodes of a graph
#  @param number of nodes in the graph
#
@njit
def floydWarshallCPUReference(pathDistanceMatrix, pathMatrix, numNodes):
    #
    # pathDistanceMatrix is the adjacency matrix(square) with
    # the dimension equal to the number of nodes in the graph.
    #
    width = numNodes

    #
    # for each intermediate node k in the graph find the shortest distance between
    # the nodes i and j and update as
    #
    # ShortestPath(i,j,k) = min(ShortestPath(i,j,k-1), ShortestPath(i,k,k-1) + ShortestPath(k,j,k-1))
    #

    for k in range(numNodes):
        for y in range(numNodes):
            yXwidth = y * numNodes
            for x in range(numNodes):
                distanceYtoX = pathDistanceMatrix[yXwidth + x]
                distanceYtoK = pathDistanceMatrix[yXwidth + k]
                distanceKtoX = pathDistanceMatrix[k * width + x]

                indirectDistance = distanceYtoK + distanceKtoX

                if (indirectDistance < distanceYtoX):
                    pathDistanceMatrix[yXwidth + x] = indirectDistance
                    pathMatrix[yXwidth + x] = k


#
#  The floyd Warshall algorithm is a multipass algorithm
#  that calculates the shortest path between each pair of
#  nodes represented by pathDistanceBuffer.
#
#  In each pass a node k is introduced and the pathDistanceBuffer
#  which has the shortest distance between each pair of nodes
#  considering the (k-1) nodes (that are introduced in the previous
#  passes) is updated such that
#
#  ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) + ShortestPath(k,y,k-1))
#  where x and y are the pair of nodes between which the shortest distance
#  is being calculated.
#
#  pathBuffer stores the intermediate nodes through which the shortest
#  path goes for each pair of nodes.
#
#  numNodes is the number of nodes in the graph.
#
#  for more detailed explaination of the algorithm kindly refer to the document
#  provided with the sample
#


@njit
def core(blockSize, numNodes, numIterations, pathDistanceMatrix, pathMatrix):

    numPasses = numNodes
    blockSize = blockSize * blockSize

    with openmp(
            "target data map(alloc: pathDistanceMatrix, pathMatrix) device(1)"
    ):
        total_time = 0

        for n in range(numIterations):

            # The floyd Warshall algorithm is a multipass algorithm
            # that calculates the shortest path between each pair of
            # nodes represented by pathDistanceBuffer.
            #
            # In each pass a node k is introduced and the pathDistanceBuffer
            # which has the shortest distance between each pair of nodes
            # considering the (k-1) nodes (that are introduced in the previous
            # passes) is updated such that
            #
            # ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) + ShortestPath(k,y,k-1))
            # where x and y are the pair of nodes between which the shortest distance
            # is being calculated.
            #
            # pathBuffer stores the intermediate nodes through which the shortest
            # path goes for each pair of nodes.
            #

            with openmp("target update to (pathDistanceMatrix) device(1)"):
                pass

            start = omp_get_wtime()

            for k in range(numPasses):
                with openmp("""target teams distribute parallel for collapse(2)
                    thread_limit (blockSize) device(1)"""):
                    for y in range(numNodes):
                        for x in range(numNodes):
                            distanceYtoX = pathDistanceMatrix[y * numNodes + x]
                            distanceYtoK = pathDistanceMatrix[y * numNodes + k]
                            distanceKtoX = pathDistanceMatrix[k * numNodes + x]
                            indirectDistance = distanceYtoK + distanceKtoX

                            if (indirectDistance < distanceYtoX):
                                pathDistanceMatrix[y * numNodes +
                                                   x] = indirectDistance
                                pathMatrix[y * numNodes + x] = k

            end = omp_get_wtime()
            total_time += (end - start)

        print("Average kernel execution time", total_time / numIterations,
              "(s)")

        with openmp("target update from (pathDistanceMatrix) device(1)"):
            pass


def main():
    if (len(sys.argv) != 4):
        print("Usage:", sys.argv[0],
              "<number of nodes> <iterations> <block size>")
        return 1

    # There are three required command-line arguments
    numNodes = int(sys.argv[1])
    numIterations = int(sys.argv[2])
    blockSize = int(sys.argv[3])

    # allocate and init memory used by host
    matrixSize = numNodes * numNodes
    pathDistanceMatrix = np.empty(matrixSize, dtype=np.uint32)
    pathMatrix = np.empty(matrixSize, dtype=np.uint32)

    # input must be initialized; otherwise host and device results may be different
    np.random.seed(2)
    MAXDISTANCE = 200
    pathDistanceMatrix = np.uint32(
        np.random.random_sample(matrixSize) * (MAXDISTANCE + 1))
    # fill the diagonal with zeros?
    for i in range(numNodes):
        iXWidth = i * numNodes
        pathDistanceMatrix[iXWidth + i] = 0

    # pathMatrix is the intermediate node from which the path passes
    # pathMatrix(i,j) = k means the shortest path from i to j
    # passes through an intermediate node k
    # Initialized such that pathMatrix(i,j) = i

    for i in range(numNodes):
        for j in range(i):
            pathMatrix[i * numNodes + j] = i
            pathMatrix[j * numNodes + i] = j
        pathMatrix[i * numNodes + i] = i

    verificationPathDistanceMatrix = np.empty(matrixSize, dtype=np.uint32)
    verificationPathMatrix = np.empty(matrixSize, dtype=np.uint32)

    verificationPathDistanceMatrix[:] = pathDistanceMatrix
    verificationPathMatrix[:] = pathMatrix

    compile()
    core(blockSize, numNodes, numIterations, pathDistanceMatrix, pathMatrix)

    # verify
    floydWarshallCPUReference.compile("void(uint32[::1], uint32[::1], int64)")
    start = omp_get_wtime()
    floydWarshallCPUReference(verificationPathDistanceMatrix,
                              verificationPathMatrix, numNodes)
    end = omp_get_wtime()
    print("Reference execution time", (end - start), "(s)")
    np.testing.assert_array_equal(pathDistanceMatrix,
                                  verificationPathDistanceMatrix, "FAIL")
    print("PASS\n")

    return 0


def compile():
    import time
    t1 = time.perf_counter()
    core.compile("void(int64, int64, int64, uint32[::1], uint32[::1])")
    t2 = time.perf_counter()
    print("ctime", t2 - t1, "s")


if __name__ == "__main__":
    main()
