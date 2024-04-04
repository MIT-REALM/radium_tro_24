import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = "results/formation2d_grad_norm_netconn/5.csv"

display_names = {
    "rmh": "RMH (gradient-free)",
    "mala": "MALA (gradient-based)",
}

if __name__ == "__main__":
    results = pd.read_csv(path)
    # Set index to steps
    results.set_index("Step", inplace=True)
    # Remape names by replacing strings of the form "Group: X-..." with just "X"
    remapper = (
        lambda s: (
            s.split("-")[0].split(": ")[-1]
            + f"{'_min' if 'MIN' in s else ''}"
            + f"{'_max' if 'MAX' in s else ''}"
        )
        if "predict" in s
        else (
            "dr" + f"{'_min' if 'MIN' in s else ''}" + f"{'_max' if 'MAX' in s else ''}"
        )
    )
    results.rename(columns=remapper, inplace=True)

    # Get a list of unique names
    names = filter(lambda s: "min" not in s, results.columns)
    names = filter(lambda s: "max" not in s, names)

    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
    fig = plt.figure(figsize=(9, 4), constrained_layout=True)

    for name, display_name in display_names.items():
        # Get the data for this name
        mean = results[name]
        min = results[name + "_min"]
        max = results[name + "_max"]

        # Plot the data
        h = plt.plot(mean.index, mean, label=display_name)

        try:
            plt.fill_between(mean.index, min, max, color=h[0].get_color(), alpha=0.2)
        except:
            import pdb

            pdb.set_trace()

    # plt.yscale("log")
    plt.ylabel("Avg. Cost (test set)")
    plt.xlabel("# Sampling Rounds")
    # plt.ylim([0, 15])
    plt.legend(loc="upper right")
    plt.savefig("formation_5.png")
