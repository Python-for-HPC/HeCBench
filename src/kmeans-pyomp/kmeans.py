from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device
import numba
import numpy as np
import sys
import math
import collections
import argparse

BLOCK_SIZE = 256
FLT_MAX = 3.40282347e+38
#min_rmse_ref = FLT_MAX

@njit
def cluster(npoints, nfeatures, features, min_nclusters, max_nclusters, threshold, isRMSE, nloops, flt_max, block_size):
    index = 0
    membership = np.empty(npoints, dtype=np.int32)
    membership_OCL = np.empty(npoints, dtype=np.int32)
    feature_swap = np.empty(npoints * nfeatures, dtype=np.float32)
    #flat_features = np.ravel(features)    
    clusters = np.empty((max_nclusters, nfeatures), dtype=np.float32)
    initial = np.arange(npoints)
    new_centers_len = np.empty(max_nclusters, dtype=np.int32)
    #new_centers = np.empty((nfeatures, max_nclusters), dtype=np.float32)
    new_centers = np.empty((max_nclusters, nfeatures), dtype=np.float32)
    #np.set_printoptions(suppress=True, precision=2)

    #print(features)
    with openmp("""target data map(to: features,
                                       initial)
                               map(alloc: feature_swap,
                                          membership,
                                          membership_OCL,
                                          clusters,
                                          new_centers_len,
                                          new_centers)"""):
        for nclusters in range(min_nclusters, max_nclusters + 1):
            if nclusters > npoints:
                break
            c = 0

            new_centers_len[:nclusters] = 0

            #with openmp("""target teams distribute parallel for thread_limit(block_size) nowait"""):
            #with openmp("""target teams distribute parallel for thread_limit(block_size)"""):
            with openmp("""target teams distribute parallel for thread_limit(256)"""):
                for tid in range(npoints):
                    for i in range(nfeatures):
                        idx = i * npoints + tid
                        #print("idx:", idx, idx // nfeatures, idx % nfeatures)
                        feature_swap[idx] = features[idx // nfeatures, idx % nfeatures]
                        #feature_swap[tid * nfeatures + i] = features[tid, i]
            #print("feature_swap\n", feature_swap)
                           
            #print("initial\n", initial)
            initial_points = npoints

            for lp in range(nloops):
                n = 0
            
                for i in range(nclusters):
                    if initial_points < 0:
                        break
                    for j in range(nfeatures):
                        clusters[i,j] = features[initial[n], j]
                    initial[n], initial[initial_points - 1] = initial[initial_points - 1], initial[n]
                    initial_points -= 1
                    n += 1

                #print("clusters and initial", lp, initial_points, n, "\n", clusters[0:nclusters, :])
                #print(initial)

                for i in range(npoints):
                    membership[i] = -1

                loop = 0
                delta = threshold + 1
                while loop == 0 or ((delta > threshold) and loop < 500):
                    delta = 0.0
                    with openmp("""target update to (clusters)"""):
                        pass
                    with openmp("""target teams distribute parallel for thread_limit(256)"""):
                    #with openmp("""target teams distribute parallel for thread_limit(block_size)"""):
                        for point_id in range(npoints):
                            min_dist = flt_max
                            for i in range(nclusters):
                                dist = 0
                                ans = 0
                                for l in range(nfeatures):
                                    val = (feature_swap[point_id * nfeatures + l] - clusters[i,l]) 
                                    ans += val * val
                                    #print("ans:", ans, point_id, i, l)
                                dist = ans
                                if dist < min_dist:
                                    min_dist = dist
                                    index = i
                            membership_OCL[point_id] = index
                    with openmp("""target update from (membership_OCL)"""):
                        pass
                    #print("do_loop", loop)
                    #print(membership_OCL)

                    for i in range(npoints):
                        cluster_id = membership_OCL[i]
                        new_centers_len[cluster_id] += 1
                        if cluster_id != membership[i]:
                            delta += 1
                            membership[i] = membership_OCL[i]
                        for j in range(nfeatures):
                            new_centers[cluster_id, j] += features[i, j]
                            #print("aaa:", cluster_id, j, features[i, j])

                    #print("membership", loop)
                    #print(membership)
                    #print("new_centers")
                    #print(new_centers[0:nclusters, :])
                    #print("new_centers_len", loop)
                    #print(new_centers_len[:nclusters])

                    for i in range(nclusters):
                        for j in range(nfeatures):
                            if new_centers_len[i] > 0:
                                clusters[i, j] = new_centers[i, j] / new_centers_len[i]
                            new_centers[i, j] = 0.0
                        new_centers_len[i] = 0

                    c += 1
                    loop += 1

                if isRMSE:
                    pass # NOT IMPLEMENTED

    return index, None, None, None

def setup():
    parser = argparse.ArgumentParser(prog="kmeans")
    parser.add_argument('filename')
    parser.add_argument('-m', type=int, default=5, help="maximum number of clusters allowed, default 5")
    parser.add_argument('-n', type=int, default=5, help="minimum number of clusters allowed, default 5")
    parser.add_argument('-t', type=float, default=0.001, help="threshold value")
    parser.add_argument('-l', type=int, default=1, help="iteration for each number of clusters")
    parser.add_argument('-b', type=bool, default=False, help="input file in binary format")
    parser.add_argument('-r', action='store_true', default=False, help="calculate RMSE")
    parser.add_argument('-o', action='store_true', default=False, help="output cluster center coordinates")
    cargs = parser.parse_args()

    npoints = 0
    nfeatures = 0

    if cargs.b:
        print("NOT IMPLEMENTED")
        """
		int infile;
		if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
			fprintf(stderr, "Error: no such file (%s)\n", filename);
			exit(1);
		}
		read(infile, &npoints,   sizeof(int));
		read(infile, &nfeatures, sizeof(int));        

		/* allocate space for features[][] and read attributes of all objects */
		buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
		features    = (float**)malloc(npoints*          sizeof(float*));
		features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
		for (i=1; i<npoints; i++)
			features[i] = features[i-1] + nfeatures;

		read(infile, buf, npoints*nfeatures*sizeof(float));

		close(infile);
        """
    else:
        with open(cargs.filename, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break      
                if len(line.split()) > 1:
                    npoints+=1
        index = 0
        with open(cargs.filename, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break      
                lsplit = line.split()
                if len(lsplit) > 1:
                    if nfeatures == 0:
                        nfeatures = len(lsplit)
                        features = np.empty((npoints, nfeatures), dtype=np.float32)
                    features[index] = np.array(lsplit)
                    index += 1

    #print("npoints:", npoints)
    #print("nfeatures:", nfeatures)

    min_nclusters = cargs.n
    max_nclusters = cargs.m
    threshold = cargs.t
    isRMSE = cargs.r
    nloops = cargs.l
    isOutput = cargs.o

    if npoints < min_nclusters:
        print("Error: min_nclusters > npoints")
        sys.exit(0)

    np.random.seed(7)

    start = omp_get_wtime()
    index, best_nclusters, cluster_centres, rmse = cluster(npoints, nfeatures, features, min_nclusters, max_nclusters, threshold, isRMSE, nloops, FLT_MAX, BLOCK_SIZE)
    end = omp_get_wtime()
    core_time = (end - start) * 1000
    print(f"Kmeans core timing: {core_time} ms")

    if min_nclusters == max_nclusters and isOutput:
        for i in range(max_nclusters):
            print(i, end="")
            for j in range(nfeatures):
                print("{:0.2f}".format(cluster_cnetres[i,j]), end="")
            print("\n")

    print(f"Number of Iterations: {nloops}")

    if min_nclusters != max_nclusters:
        print(f"Best number of clusters is {best_nclusters}");			
    else:
        if nloops != 1:
            if isRMSE:
                print(f"Number of trials to approach the best RMSE of {rmse} is ", index + 1)
        else:
            print(f"RMSE: {rmse}")

if __name__ == "__main__":
    print(f"WG size of kernel_swap = {BLOCK_SIZE}")
    setup()
