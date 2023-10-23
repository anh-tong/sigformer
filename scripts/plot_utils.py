import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_histogram(arrs, names, bins=30, stat="probability", **fig_kwargs):

    dfs = [
        pd.DataFrame.from_dict({"value": value, "name": name})
        for value, name in zip(arrs, names)
    ]
    df = pd.concat(axis=0, ignore_index=True, objs=dfs)

    fig, ax = plt.subplots(**fig_kwargs)
    sns.histplot(
        data=df,
        x="value",
        hue="name",
        ax=ax,
        multiple="dodge",
        stat=stat,
        bins=bins,
    )

    return fig
