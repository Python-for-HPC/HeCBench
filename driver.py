import argparse
import yaml
import subprocess
from pathlib import Path
import time
import csv


def c_compile(repeats, benchmark, arch):
    cwd = Path.cwd() / f"src/{benchmark['name']}-omp"

    def clean():
        subprocess.run("make -f Makefile.clang clean",
                       shell=True,
                       cwd=cwd,
                       check=True)

    def build():
        subprocess.run(f"make -f Makefile.clang ARCH={arch}",
                       shell=True, cwd=cwd, check=True)
    # warmup
    print(f"\N{fire} Warmup build {benchmark['name']} C version...")
    clean()
    build()

    ctimes = []
    for rep in range(repeats):
        print(f"\N{gear} => Compile {benchmark['name']} C version {cwd}...")
        clean()
        t1 = time.perf_counter()
        build()
        t2 = time.perf_counter()
        ctimes.append(["omp", t2 - t1])

    return ctimes


def py_compile(repeats, benchmark):
    cwd = Path.cwd() / f"src/{benchmark['name']}-pyomp"
    exe = benchmark['exe']['pyomp'].split(".")[0]

    def build():
        p = subprocess.run(f"python -c 'import {exe};{exe}.compile()'",
                           capture_output=True,
                           shell=True,
                           cwd=cwd,
                           check=True)
        return p

    # warmup
    print(f"\N{fire} Warmup build {benchmark['name']} pyomp...")
    build()

    ctimes = []
    for rep in range(repeats):
        print(f"\N{gear} => Compile {benchmark['name']} pyomp {cwd}...")
        p = build()
        stdout = str(p.stdout)
        ctime = float(stdout.split()[1])
        ctimes.append(["pyomp", ctime])

    return ctimes


def run(repeats, profiler, outdir, version, argkey, benchmark, metrics, force):
    exe = ("python " if version == "pyomp" else "") + str(
        Path.cwd() /
        f"src/{benchmark['name']}-{version}/{benchmark['exe'][version]}")

    cwd = Path(f"{outdir}") / benchmark['name']

    def run_internal(cmd):
        subprocess.run(cmd, shell=True, cwd=cwd, check=True)

    # NOTE: We assume driver runs always at the HeCBench root path.
    input_str = benchmark["inputs"][argkey].replace(
        "$ROOT", str(Path.cwd()))

    warmup_has_ran = False

    for rep in range(repeats):
        if profiler == "nvprof":
            tracefile = (
                f"nvprof-metrics-{version}-{benchmark['name']}-{argkey}-{rep}.csv"
                if metrics else
                f"nvprof-{version}-{benchmark['name']}-{argkey}-{rep}.csv")
            check = cwd / tracefile
        elif profiler == "nsys":
            tracefile = (
                f"nsys-{version}-{benchmark['name']}-{argkey}-{rep}")
            check = cwd / f"{tracefile}.nsys-rep"
        else:
            raise Exception(f"Invalid profiler {profiler}")

        if not force:
            if check.exists():
                print(
                    f"\u26A0\uFE0F WARNING: tracefile {tracefile} exists, will not re-run"
                )
                continue

        cmd = exe + " " + input_str
        # Warmup
        if not warmup_has_ran:
            print(f"\N{fire} Warmup run {version}...")
            run_internal(cmd)
            warmup_has_ran = True

        if profiler == "nvprof":
            profile_cmd = f"nvprof --print-gpu-trace --normalized-time-unit us --csv --log-file {tracefile}"
            cmd = (profile_cmd
                   + ("--metrics all " if metrics else "") + " " + cmd)
        elif profiler == "nsys":
            profile_cmd = f"nsys profile --trace=cuda -o {tracefile}"
            cmd = profile_cmd + " " + cmd
        else:
            raise Exception(f"Invalid profiler {profiler}")

        # Run through profiler, collect detailed metrics if enabled.
        print(f"\N{rocket} => Run cmd {cmd}")
        run_internal(cmd)

        if profiler == "nsys":
            cmd = f"nsys stats --report cuda_gpu_trace -f csv:dur=us:mem=B --force-export=true -o {tracefile} {tracefile}.nsys-rep"
            print(f"=> Run create stats {cmd}")
            run_internal(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Measure the performance of pyomp benchmark programs.")
    parser.add_argument("-i",
                        "--input",
                        help="input configuration file",
                        required=True)
    parser.add_argument("-a",
                        "--arch",
                        help="GPU arch for C compilation",
                        required=True)
    parser.add_argument("-o",
                        "--output-dir",
                        help="output directory",
                        required=True)
    parser.add_argument("-p",
                        "--profiler",
                        help="profiler nvprof|nsys",
                        choices=["nvprof", "nsys"],
                        required=True)
    parser.add_argument("-r",
                        "--repeats",
                        help="experiment repeats",
                        type=int,
                        required=True)
    parser.add_argument("-b",
                        "--benchmarks",
                        help="list of benchmarks to run",
                        nargs="+")
    parser.add_argument("-f",
                        "--force",
                        help="force re-running",
                        action="store_true")
    parser.add_argument(
        "-m",
        "--metrics",
        help="enable collecting detailed nvprof metrics",
        action="store_true",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    benchmarks_to_run = ([
        b for b in config["benchmarks"] if b["name"] in args.benchmarks
    ] if args.benchmarks else config["benchmarks"])
    assert len(benchmarks_to_run) >= 1, "Expected at least 1 benchmark to run"

    print("=== Run description ===")
    print("Input", args.input)
    print("Repeats", args.repeats)
    print("Benchmarks", [b["name"] for b in benchmarks_to_run])
    print("Metrics", args.metrics)
    print("Force", args.force)
    print("=======================")

    for benchmark in benchmarks_to_run:
        assert len(benchmark["inputs"]) == 1, "Expected single input"
        assert (list(benchmark["inputs"])[0] == "default"
                ), "Expected only default hecbench input"

        cwd = Path(f"{args.output_dir}") / benchmark["name"]
        cwd.mkdir(parents=True, exist_ok=True)

        ctimes = []
        # Run C openmp
        ctimes.extend(c_compile(args.repeats, benchmark, args.arch))
        run(args.repeats, args.profiler, args.output_dir, "omp", "default",
            benchmark, args.metrics, args.force)
        # Run PyOMP
        ctimes.extend(py_compile(args.repeats, benchmark))
        run(args.repeats, args.profiler, args.output_dir, "pyomp", "default",
            benchmark, args.metrics, args.force)

        ctimes_outfn = Path(
            f"{args.output_dir}/{benchmark['name']}") / f"ctimes-{benchmark['name']}.csv"
        with open(ctimes_outfn, "w") as f:
            cw = csv.writer(f, delimiter=",")
            cw.writerow(["Version", "Ctime"])
            for row in ctimes:
                cw.writerow(row)

    print("\U0001f389 DONE!")


if __name__ == "__main__":
    main()
