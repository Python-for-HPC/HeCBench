import argparse
import yaml
import subprocess
from pathlib import Path


def compile(benchmark):
    cwd = Path.cwd() / f"src/{benchmark['name']}-omp"
    subprocess.run("make -f Makefile.lassen", shell=True, cwd=cwd, check=True)


def run(rep, version, argkey, benchmark):
    exe = ("python " if version == "pyomp" else "") + str(
        Path.cwd() / f"src/{benchmark['name']}-{version}/{benchmark['exe'][version]}"
    )

    cwd = Path.cwd() / f"results/pyomp/{benchmark['name']}"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Run to get execution times only for kernels, data transfer.
    logfile = f"nvprof-{version}-{benchmark['name']}-{argkey}-{rep}.csv"
    cmd = (
        f"nvprof --print-gpu-trace --normalized-time-unit us --csv --log-file {logfile} "
        + exe
        + " "
        + benchmark["inputs"][argkey]
    )
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)

    # Run to get detailed profiler information.
    logfile = f"nvprof-metrics-{version}-{benchmark['name']}-{argkey}-{rep}.csv"
    cmd = (
        f"nvprof --print-gpu-trace --metrics all --normalized-time-unit us --csv --log-file {logfile} "
        + exe
        + " "
        + benchmark["inputs"][argkey]
    )
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Measure the performance of pyomp benchmark programs."
    )
    parser.add_argument("-i", "--input", help="input configuration file", required=True)
    parser.add_argument(
        "-r", "--repeats", help="experiment repeats", type=int, required=True
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for benchmark in config["benchmarks"]:
        assert len(benchmark["inputs"]) == 1, "Expected single input"
        assert (
            list(benchmark["inputs"])[0] == "default"
        ), "Expected only default hecbench input"

        for rep in range(args.repeats):
            # Run C openmp
            compile(benchmark)
            run(rep, "omp", "default", benchmark)
            # Run PyOMP
            run(rep, "pyomp", "default", benchmark)


if __name__ == "__main__":
    main()
