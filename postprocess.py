import argparse
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
        df = df[["Name", "Duration"]].astype({"Duration": float})
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


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess measurement results for pyomp benchmarks."
    )
    parser.add_argument("-i", "--input", help="input configuration file", required=True)
    parser.add_argument(
        "-b", "--benchmarks", help="list of benchmarks to process", nargs="+"
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

    for benchmark in benchmarks_to_process:
        cat_df = pd.DataFrame()
        print(f"\N{rocket} => Processing {benchmark['name']}...")
        result_dir = Path.cwd() / f"results/pyomp"

        df = create_dataframe(benchmark["name"], result_dir / benchmark["name"], "omp")
        cat_df = pd.concat((cat_df, df))
        df = create_dataframe(
            benchmark["name"], result_dir / benchmark["name"], "pyomp"
        )
        cat_df = pd.concat((cat_df, df))

        res = cat_df.drop("Rep", axis=1).groupby(["Name", "Version"]).mean().unstack()
        res.columns = res.columns.droplevel(0)
        ax = res.plot.bar(width=0.6)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("")
        ax.set_ylabel("Time (us)\n$log_2$")
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=0)
        ax.get_legend().set_title("")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")
        plt.tight_layout()
        plt.savefig(result_dir / f"time-{benchmark['name']}.pdf")

    print("\U0001f389 DONE!")


if __name__ == "__main__":
    main()
