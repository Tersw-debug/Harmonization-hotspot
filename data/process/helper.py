import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(
    correlation,
    title="Correlation matrix"
):

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_all_correlation_results(
    correlation_results: dict,
    methods: tuple = ("pearson", "spearman"),
    periods: list | None = None,
    sensors: list | None = None,
    platforms: list | None = None,
):

    plotted = 0

    for (sensor, platform), period_results in correlation_results.items():

        # Optional sensor filtering
        if sensors is not None and sensor not in sensors:
            continue

        # Optional platform filtering
        if platforms is not None and platform not in platforms:
            continue

        for period, metrics in period_results.items():

            # Optional period filtering
            if periods is not None and period not in periods:
                continue

            for method in methods:

                correlation = metrics.get(method)

                # Skip unavailable or empty results
                if (
                    correlation is None
                    or not isinstance(correlation, pd.DataFrame)
                    or correlation.empty
                ):
                    continue

                title = (
                    f"{sensor} | {platform} | "
                    f"{period} | {method.capitalize()} Correlation"
                )

                plot_correlation_matrix(
                    correlation,
                    title=title
                )

                plotted += 1

                # Release figure memory after displaying
                plt.close("all")

    print(f"Successfully plotted {plotted} correlation matrices.")

    return plotted
