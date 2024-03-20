import argparse
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from functools import cache
from IPython import embed

def create_dataframe(bench_name, result_dir, version):
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
        # Skip the first row after the header which contains units.
        df = df[1:]
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
def get_dataframe(result_dir, bench_name):
    print("Create the dataframe...")
    cat_df = pd.DataFrame()
    df = create_dataframe(
        bench_name, result_dir / bench_name, "omp")
    cat_df = pd.concat((cat_df, df))
    df = create_dataframe(
        bench_name, result_dir / bench_name, "pyomp"
    )
    cat_df = pd.concat((cat_df, df))
    return cat_df

def plot_exetimes(output_dir, benchmark, df, legend):
    print("Plot exetimes...")
    # Drop rows of memcpy data tranfers (HtoD or DtoH).
    df = df[~df.Name.str.contains("Memcpy")]
    df = df[["Name", "Duration", "Version", "Rep"]].astype({"Duration": float})
    res = df.drop("Rep", axis=1).groupby(["Name", "Version"]).mean().unstack()
    res.columns = res.columns.droplevel(0)
    ax = res.plot.bar(width=0.6, legend=legend)
    ax.set_title(benchmark["name"])
    #ax.set_yscale("log", base=2)
    #ax.yaxis.set_major_formatter(ScalarFormatter())
    #ax.set_ylabel("Time (us)\n$log_2$")
    ax.set_ylabel("Time (us)")
    ax.set_xlabel("")
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=0)
    #ax.get_legend().set_title("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.savefig(output_dir / f"exetime-{benchmark["name"]}.pdf")
    plt.close()

def plot_memcpy_times(output_dir, benchmark, df, legend):
    print("Plot memcpy times...")
    # Keep only rows of memcpy data tranfers (HtoD or DtoH).
    df = df[df.Name.str.contains("Memcpy")]
    df = df[["Name", "Duration", "Version", "Rep"]].astype({"Duration": float})
    # Convert to ms
    df.Duration = df.Duration / 1000.0
    res = df.drop("Rep", axis=1).groupby(["Name", "Version"]).sum().unstack()
    res.columns = res.columns.droplevel(0)
    ax = res.plot.bar(width=0.6, legend=legend)
    ax.set_title(benchmark["name"])
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Time (ms)\n$log_2$")
    ax.set_xlabel("")
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=0)
    #ax.get_legend().set_title("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.savefig(output_dir / f"memcpy-times-{benchmark['name']}.pdf")
    plt.close()

def plot_memcpy_size(output_dir, benchmark, df, legend):
    print("Plot memcpy size...")
    # Keep only rows of memcpy data tranfers (HtoD or DtoH).
    df = df[df.Name.str.contains("Memcpy")]
    df = df[["Name", "Size", "Version", "Rep"]].astype({"Size": float})
    res = df.drop("Rep", axis=1).groupby(["Name", "Version"]).sum().unstack()
    res.columns = res.columns.droplevel(0)
    ax = res.plot.bar(width=0.6, legend=legend)
    ax.set_title(benchmark["name"])
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Size (MB)\n$log_2$")
    ax.set_xlabel("")
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=0)
    #ax.get_legend().set_title("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", rotation=-30)
    plt.tight_layout()
    plt.savefig(output_dir / f"memcpy-size-{benchmark['name']}.pdf")
    plt.close()

def plot_reg_usage(output_dir, benchmark, df, legend):
    print("Plot register usage...")
    df = df[["Name", "Registers Per Thread", "Version", "Rep"]
            ].dropna().astype({"Registers Per Thread": int})
    res = df.drop("Rep", axis=1).groupby(["Name", "Version"]).mean().unstack()
    res.columns = res.columns.droplevel(0)
    ax = res.plot.bar(width=0.6, legend=legend)
    ax.set_title(benchmark["name"])
    ax.set_xlabel("")
    ax.set_ylabel("Number of registers")
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=0)
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(output_dir / f"registers-{benchmark['name']}.pdf")
    plt.close()

def plot_ctimes(output_dir, benchmark, df, legend):
    print("Plot ctimes ...")
    res = df.groupby("Version").mean()
    #res.T.apply(print)
    # Plot transpose to make Version a column (differently colored bars).
    ax = res.T.plot.bar(width=0.6, legend=legend)
    ax.set_title(benchmark["name"])
    ax.set_ylabel("Time (s)")
    ax.set_xticklabels([benchmark["name"]], rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.savefig(output_dir / f"ctime-{benchmark['name']}.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess measurement results for pyomp benchmarks."
    )
    parser.add_argument("-i", "--input", help="input configuration file", required=True)
    parser.add_argument("-r", "--result-dir",
                        help="path to the results directory", required=True)
    parser.add_argument("-o", "--output-dir",
                        help="path to output directory", required=True)
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
        "--plot-reg-usage", help="plot register usage", action="store_true"
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
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["figure.figsize"] = (8,4)
    #ax.spines[['right', 'top']].set_visible(False)

    legend = True
    for benchmark in benchmarks_to_process:
        result_dir = Path(args.result_dir)
        output_dir = Path(args.output_dir)
        print(f"\N{rocket} => Processing {benchmark['name']}...")

        # Plot execution times.
        if args.plot_exetimes:
            plot_exetimes(output_dir, benchmark,
                          get_dataframe(result_dir, benchmark["name"]), legend)

        if args.plot_memcpy_times:
            plot_memcpy_times(output_dir, benchmark, get_dataframe(
                result_dir, benchmark["name"]), legend)

        if args.plot_memcpy_size:
            plot_memcpy_size(output_dir, benchmark, get_dataframe(
                result_dir, benchmark["name"]), legend)

        # Plot register usage.
        if args.plot_reg_usage:
            plot_reg_usage(output_dir, benchmark, get_dataframe(
                result_dir, benchmark["name"]), legend)

        # Plot compilation times.
        if args.plot_ctimes:
            data = result_dir / f"{benchmark['name']}" / f"ctimes-{benchmark['name']}.csv"
            df = pd.read_csv(data)
            plot_ctimes(output_dir, benchmark, df, legend)

        # Print legend only on the first plot.
        legend = False



    print("\U0001f389 DONE!")


if __name__ == "__main__":
    main()
