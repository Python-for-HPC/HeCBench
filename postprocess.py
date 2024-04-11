import argparse
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from functools import cache
import numpy as np
from scipy import stats


common_plot_kwargs = {
    "color": ["xkcd:light blue", "xkcd:light orange"],
    "capsize": 4,
}


def create_dataframe_nsys(bench_name, result_dir, version):
    # Set all types to str because of awkwardness of nvprof CSV output.
    dtypes_dict = {
        "Start (ns)": str,
        "Duration (us)": str,
        "CorrId": str,
        "GrdX": str,
        "GrdY": str,
        "GrdZ": str,
        "BlkX": str,
        "BlkY": str,
        "BlkZ": str,
        "Reg/Trd": str,
        "StcSMem (B)": str,
        "DymSMem (B)": str,
        "Bytes (B)": str,
        "Throughput (MBps)": str,
        "SrcMemKd": str,
        "DstMemKd": str,
        "Device": str,
        "Ctx": str,
        "Strm": str,
        "Name": str,
    }

    files = frozenset(result_dir.glob(f"nsys-{version}-*.csv"))
    cat_df = pd.DataFrame()
    for i, file in enumerate(files):
        # Split by '-' then by '_' because of the cuda_gpu_trace suffix.
        # print(f"Reading file {i+1} / {len(files)}", end="\r")
        print(f"\r[{version:>5}] Reading file {i+1:3} / {len(files):3}", end="")
        rep = file.stem.split("-")[-1].split("_")[0]
        df = pd.read_csv(file, dtype=dtypes_dict)
        df["Version"] = version
        df["Rep"] = int(rep) + 1
        cat_df = pd.concat((cat_df, df))
    print()

    # Normalize kernel names.
    uniq_names = set(cat_df.Name.unique())
    mem_set = {"[CUDA memcpy Device-to-Host]", "[CUDA memcpy Host-to-Device]"}
    kernels = uniq_names - mem_set
    assert len(kernels) == 1, "Postprocessing expects 1 kernel!"
    kernels_norm = [bench_name]
    remap = dict(zip(kernels, kernels_norm))
    remap["[CUDA memcpy Device-to-Host]"] = "Memcpy DtoH"
    remap["[CUDA memcpy Host-to-Device]"] = "Memcpy HtoD"

    cat_df.rename(
        columns={
            "Duration (us)": "Duration",
            "Bytes (B)": "Size",
            "Reg/Trd": "Registers Per Thread",
        },
        inplace=True,
    )
    cat_df.replace({"Name": remap}, inplace=True)
    # Convert B to MB.
    cat_df = cat_df.astype({"Size": float})
    cat_df.Size = cat_df.Size / (1024.0 * 1024.0)
    return cat_df


def create_dataframe_nvprof(bench_name, result_dir, version):
    # Set all types to str because of awkwardness of nvprof CSV output.
    dtypes_dict = {
        "Start": str,
        "Duration": str,
        "Grid X": str,
        "Grid Y": str,
        "Grid Z": str,
        "Block X": str,
        "Block Y": str,
        "Block Z": str,
        "Registers Per Thread": str,
        "Static SMem": str,
        "Dynamic SMem": str,
        "Size": str,
        "Throughput": str,
        "SrcMemType": str,
        "DstMemType": str,
        "Device": str,
        "Context": str,
        "Stream": str,
        "Name": str,
        "Correlation_ID": str,
    }

    files = result_dir.glob(f"nvprof-{version}-*.csv")
    cat_df = pd.DataFrame()
    for file in files:
        rep = file.stem.split("-")[-1]
        df = pd.read_csv(file, skiprows=3, dtype=dtypes_dict)
        # Read the mem size units as it is auto-decided by nvprof and normalize to MB.
        unit_mem_size = df[0:1].Size[0]
        # Skip the first row after the header which contains units.
        df = df[1:]
        df = df.astype({"Size": float})
        if unit_mem_size == "B":
            df.size = df.Size / (1024.0 * 1024.0)
        elif unit_mem_size == "KB":
            df.Size = df.Size / 1024.0
        elif unit_mem_size == "MB":
            df.Size = df.Size * 1.0
        elif unit_mem_size == "GB":
            df.Size = df.Size * 1024.0
        else:
            raise Exception(f"Unhandle unit_mem_size {unit_mem_size}")
        df["Version"] = version
        df["Rep"] = int(rep) + 1
        cat_df = pd.concat((cat_df, df))

    # Normalize kernel names.
    kernels = [x for x in cat_df.Name.unique() if "__omp_offload" in x]
    kernels_norm = (
        [bench_name]
        if len(kernels) == 1
        else [f"omp_kernel_{i+1}" for i in range(len(kernels))]
    )
    remap = dict(zip(kernels, kernels_norm))
    remap["[CUDA memcpy DtoH]"] = "Memcpy DtoH"
    remap["[CUDA memcpy HtoD]"] = "Memcpy HtoD"
    cat_df.replace({"Name": remap}, inplace=True)
    return cat_df


@cache
def get_dataframe(profiler, result_dir, bench_name):
    print(f"Create the dataframe for {bench_name}...")
    cat_df = pd.DataFrame()
    if profiler == "nvprof":
        df = create_dataframe_nvprof(bench_name, result_dir / bench_name, "omp")
        cat_df = pd.concat((cat_df, df))
        df = create_dataframe_nvprof(bench_name, result_dir / bench_name, "pyomp")
    elif profiler == "nsys":
        df = create_dataframe_nsys(bench_name, result_dir / bench_name, "omp")
        cat_df = pd.concat((cat_df, df))
        df = create_dataframe_nsys(bench_name, result_dir / bench_name, "pyomp")
    else:
        raise Exception(f"Invalid profiler {profiler}")

    cat_df = pd.concat((cat_df, df))
    return cat_df


def plot_exetimes(output_dir, benchmark, df, legend):
    print("Plot exetimes...")
    # Drop rows of memcpy data tranfers (HtoD or DtoH).
    df = df[~df.Name.str.contains("Memcpy")]
    df = df[["Name", "Duration", "Version", "Rep"]].astype({"Duration": float})

    n_omp = df[df.Version == "omp"].shape[0]
    n_pyomp = df[df.Version == "omp"].shape[0]
    assert n_omp == n_pyomp, "Expected same size data for omp, pyomp"
    n = n_omp

    # Compute mean, std across Reps and unstack to get Versions as columns.
    res_mean = df.drop("Rep", axis=1).groupby(["Name", "Version"]).mean().unstack()
    res_std = df.drop("Rep", axis=1).groupby(["Name", "Version"]).std().unstack()
    res_mean.columns = res_mean.columns.droplevel(0)
    res_std.columns = res_std.columns.droplevel(0)

    unit = "us"
    if (res_mean.omp > 1000.0).any():
        unit = "ms"
        res_mean = res_mean / 1000.0
        res_std = res_std / 1000.0

    ci = 0.95
    yerr = res_std / np.sqrt(n) * stats.t.ppf(1 - (1 - ci) / 2.0, n - 1)

    ax = res_mean.plot.bar(legend=legend, yerr=yerr, width=0.4, **common_plot_kwargs)
    # ax.set_title(benchmark["name"])
    ax.set_title(rf"\emph{{{benchmark['name']}}}", pad=18.0)
    ax.set_ylabel(f"Time ({unit})", fontsize=26)
    ax.set_xlabel("")
    # labels = ax.get_xticklabels()
    # ax.set_xticklabels(labels, rotation=0)
    ax.set_xticklabels([])
    if legend:
        ax.legend(handlelength=2.0, title="", ncols=2)

    for container in ax.containers[1::2]:
        ax.bar_label(container, fmt="%.1f", label_type="center")
    plt.tight_layout()
    plt.savefig(output_dir / f"exetime-{benchmark['name']}.pdf")
    plt.close()


def plot_memcpy_times(output_dir, benchmark, df, legend):
    print("Plot memcpy times...")
    # Keep only rows of memcpy data tranfers (HtoD or DtoH).
    df = df[df.Name.str.contains("Memcpy")]
    df = df[["Name", "Duration", "Version", "Rep"]].astype({"Duration": float})
    df.Duration = df.Duration / 1000.0
    # Compute the sum of memory transfer time per Rep, unstack to get Reps as columns.
    res_sum = df.groupby(["Name", "Version", "Rep"]).sum().unstack()
    # Compute the mean, std across Reps (row-wise).
    res_mean = res_sum.mean(axis=1).unstack()
    res_std = res_sum.std(axis=1).unstack()

    nreps = len(df[df.Version == "omp"].Rep.unique())
    assert nreps == len(
        df[df.Version == "pyomp"].Rep.unique()
    ), "Expected same nreps for both omp and pyomp"

    ci = 0.95
    yerr = res_std / np.sqrt(nreps) * stats.t.ppf(1 - (1 - ci) / 2.0, nreps - 1)

    ax = res_mean.plot.bar(legend=legend, yerr=yerr, width=0.8, **common_plot_kwargs)

    ax.set_title(rf"\emph{{{benchmark['name']}}}", pad=18.0)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Time (ms)\n$log_2$", fontsize=26)
    ax.set_xlabel("")
    labels = [label.get_text().replace("Memcpy ", "") for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=0)
    if legend:
        ax.legend(handlelength=2.0, title="", ncols=1)
    for container in ax.containers[1::2]:
        # ax.bar_label(container, fmt="%.1f", label_type="center")
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.savefig(output_dir / f"memcpy-times-{benchmark['name']}.pdf")
    plt.close()


def plot_memcpy_size(output_dir, benchmark, df, legend):
    print("Plot memcpy size...")
    # Keep only rows of memcpy data tranfers (HtoD or DtoH).
    df = df[df.Name.str.contains("Memcpy")]
    # breakpoint()
    df = df[["Name", "Size", "Version", "Rep"]].astype({"Size": float})
    # Compute the sum of memory data sizes per Rep, unstack to get Reps as columns.
    res_sum = df.groupby(["Name", "Version", "Rep"]).sum().unstack()
    # Verify all Reps have the same number of data.
    assert np.all(
        [len(np.unique(row)) == 1 for row in res_sum.values]
    ), f"Expected same value across Rep columns {res_sum.values}"
    # Mean here just collapses the identical data in the Rep columns.
    res = res_sum.mean(axis=1).unstack()

    ax = res.plot.bar(legend=legend, width=0.8, **common_plot_kwargs)

    ax.set_title(rf"\emph{{{benchmark['name']}}}", pad=18.0)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Size (MB)\n$log_2$", fontsize=26)
    ax.set_xlabel("")
    labels = [label.get_text().replace("Memcpy ", "") for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=0)
    if legend:
        ax.legend(handlelength=2.0, title="", ncols=1)
    for container in ax.containers:

        # Remove single trailing zero from float label.
        def format(x):
            f = f"{x:.2f}"
            if f[-1] == "0":
                f = f[:-1]
            return f

        # ax.bar_label(container, fmt="%.2f")
        ax.bar_label(container, fmt=format)
    plt.tight_layout()
    plt.savefig(output_dir / f"memcpy-size-{benchmark['name']}.pdf")
    plt.close()


def plot_ctimes(output_dir, benchmark, df, legend):
    print("Plot ctimes ...")
    res = df.groupby("Version").mean()
    res_std = df.groupby("Version").std()

    n_omp = len(df[df.Version == "omp"])
    n_pyomp = len(df[df.Version == "pyomp"])
    assert n_omp == n_pyomp, "Expected same sample size for omp, pyomp"
    n = n_omp
    ci = 0.95
    yerr = res_std / np.sqrt(n) * stats.t.ppf(1 - (1 - ci) / 2.0, n - 1)

    # Plot transpose to make Version a column (differently colored bars).
    ax = res.T.plot.bar(legend=legend, width=0.4, yerr=yerr.T, **common_plot_kwargs)
    # ax.set_title(benchmark["name"])
    ax.set_title(rf"\emph{{{benchmark['name']}}}", pad=18.0)
    ax.set_ylabel("Time (s)", fontsize=26)
    # ax.set_xticklabels([benchmark["name"]], rotation=0)
    ax.set_xticklabels([])
    if legend:
        ax.legend(handlelength=2.0, title="", ncols=1)
    # Add label to value bar containers (odd=value, even=error).
    for container in ax.containers[1::2]:
        ax.bar_label(container, fmt="%.1f", label_type="center")
    plt.tight_layout()
    plt.savefig(output_dir / f"ctime-{benchmark['name']}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess measurement results for pyomp benchmarks."
    )
    parser.add_argument("-i", "--input", help="input configuration file", required=True)
    parser.add_argument(
        "-r", "--result-dir", help="path to the results directory", required=True
    )
    parser.add_argument(
        "-o", "--output-dir", help="path to output directory", required=True
    )
    parser.add_argument(
        "-p",
        "--profiler",
        help="profiler nvprof|nsys",
        choices=["nvprof", "nsys"],
        required=True,
    )
    parser.add_argument(
        "-b", "--benchmarks", help="list of benchmarks to process", nargs="+"
    )
    parser.add_argument(
        "--plot-exetimes", help="plot execution times", action="store_true"
    )
    parser.add_argument(
        "--plot-memcpy-times", help="plot memcpy times", action="store_true"
    )
    parser.add_argument(
        "--plot-memcpy-size", help="plot memcpy size", action="store_true"
    )
    parser.add_argument(
        "--plot-ctimes", help="plot compilation times", action="store_true"
    )
    parser.add_argument(
        "--latex", help="use latex fonts", action="store_true"
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    benchmarks_to_process = (
        [b for b in config["benchmarks"] if b["name"] in args.benchmarks]
        if args.benchmarks
        else config["benchmarks"]
    )
    assert len(benchmarks_to_process) >= 1, "Expected at least 1 benchmark to process"

    # Matplotlib setup.
    plt.rcParams.update({"font.size": 22})
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["figure.figsize"] = (6, 3.7)
    if args.latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "ptm"

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)

    df_dict = dict(
        [
            (
                benchmark["name"],
                get_dataframe(args.profiler, result_dir, benchmark["name"]),
            )
            for benchmark in benchmarks_to_process
        ]
    )

    legend = True
    for benchmark in benchmarks_to_process:
        print(f"\N{rocket} => Processing {benchmark['name']}...")

        # Plot execution times.
        if args.plot_exetimes:
            plot_exetimes(output_dir, benchmark, df_dict[benchmark["name"]], legend)

        if args.plot_memcpy_times:
            plot_memcpy_times(output_dir, benchmark, df_dict[benchmark["name"]], legend)

        if args.plot_memcpy_size:
            plot_memcpy_size(output_dir, benchmark, df_dict[benchmark["name"]], legend)

        # Plot compilation times.
        if args.plot_ctimes:
            data = (
                result_dir / f"{benchmark['name']}" / f"ctimes-{benchmark['name']}.csv"
            )
            df = pd.read_csv(data)
            plot_ctimes(output_dir, benchmark, df, legend)

        # Print legend only on the first plot.
        legend = False

    print("\U0001f389 DONE!")


if __name__ == "__main__":
    main()
