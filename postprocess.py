import argparse
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess measurement results for pyomp benchmarks."
    )
    parser.add_argument("-i", "--input", help="input configuration file", required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    database = pd.DataFrame()

    def create_dataframes(result_dir, version):
        files = result_dir.glob(f"nvprof-{version}-*.csv")
        for file in files:
            rep = file.stem.split("-")[-1]
            df = pd.read_csv(file, skiprows=3)
            # Skip the first row after the header which contains units of metrics.
            df = df[1:]
            df = df[["Name", "Duration"]].astype({"Duration": float})
            df["Version"] = version
            df["Rep"] = int(rep)

        # Normalize kernel names.
        kernels = [x for x in df.Name.unique() if "__omp_offload" in x]
        kernels_norm = (
            [benchmark["name"]]
            if len(kernels) == 1
            else [f"omp_kernel_{i+1}" for i in range(len(kernels))]
        )
        remap = dict(zip(kernels, kernels_norm))
        remap["[CUDA memcpy DtoH]"] = "Memcpy DtoH"
        remap["[CUDA memcpy HtoD]"] = "Memcpy HtoD"
        df.replace({"Name": remap}, inplace=True)
        return df

    for benchmark in config["benchmarks"]:
        result_dir = Path.cwd() / f"results/pyomp/{benchmark['name']}"

        df = create_dataframes(result_dir, "omp")
        database = pd.concat((database, df))
        df = create_dataframes(result_dir, "pyomp")
        database = pd.concat((database, df))

        res = database.drop("Rep", axis=1).groupby(["Name", "Version"]).mean().unstack()
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
        outdir = Path.cwd() / f"results/pyomp"
        plt.savefig(outdir / f"time-{benchmark['name']}.pdf")


if __name__ == "__main__":
    main()
