import argparse
import yaml
import subprocess
from pathlib import Path


def compile(benchmark):
    cwd = Path.cwd() / f"src/{benchmark['name']}-omp"
    print(f"\N{gear} => Compile {cwd}...")
    subprocess.run("make -f Makefile.lassen", shell=True, cwd=cwd, check=True)


def run(rep, version, argkey, benchmark, metrics):
    exe = ("python " if version == "pyomp" else "") + str(
        Path.cwd() / f"src/{benchmark['name']}-{version}/{benchmark['exe'][version]}"
    )

    cwd = Path.cwd() / f"results/pyomp/{benchmark['name']}"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Run through nvrpof, collect detailed metrics if enabled.
    logfile = (
        f"nvprof-metrics-{version}-{benchmark['name']}-{argkey}-{rep}.csv"
        if metrics
        else f"nvprof-{version}-{benchmark['name']}-{argkey}-{rep}.csv"
    )

    cmd = (
        f"nvprof --print-gpu-trace --normalized-time-unit us --csv --log-file {logfile} "
        + ("--metrics all " if metrics else "")
        + exe
        + " "
        + benchmark["inputs"][argkey]
    )

    print(f"\N{rocket} => Run cmd {cmd}")
    check = cwd / logfile
    if check.exists():
        print(f"\u26A0\uFE0F WARNING: logfile {logfile} exists, will not re-run")
        return

    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Measure the performance of pyomp benchmark programs."
    )
    parser.add_argument("-i", "--input", help="input configuration file", required=True)
    parser.add_argument(
        "-r", "--repeats", help="experiment repeats", type=int, required=True
    )
    parser.add_argument(
        "-b", "--benchmarks", help="list of benchmarks to run", nargs="+"
    )
    parser.add_argument(
        "-m",
        "--metrics",
        help="enable collecting detailed nvprof metrics",
        action="store_true",
    )
    args = parser.parse_args()

    print("=== Run description ===")
    print("Input", args.input)
    print("Repeats", args.repeats)
    print("Benchmarks", args.benchmarks)
    print("Metrics", args.metrics)
    print("=======================")

    with open(args.input, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    benchmarks_to_run = (
        [b for b in config["benchmarks"] if b["name"] in args.benchmarks]
        if args.benchmarks
        else config["benchmarks"]
    )
    assert len(benchmarks_to_run) >= 1, "Expected at least 1 benchmark to run"

    for benchmark in benchmarks_to_run:
        assert len(benchmark["inputs"]) == 1, "Expected single input"
        assert (
            list(benchmark["inputs"])[0] == "default"
        ), "Expected only default hecbench input"

        for rep in range(args.repeats):
            # Run C openmp
            compile(benchmark)
            run(rep, "omp", "default", benchmark, args.metrics)
            # Run PyOMP
            run(rep, "pyomp", "default", benchmark, args.metrics)

    print("\U0001f389 DONE!")


if __name__ == "__main__":
    main()
