import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os


def anova_and_tukey_stats(data: pd.DataFrame,
                          values: str,
                          groups: str,
                          alpha: float = 0.05):
    """
    Gets ANOVA, Tukey's HSD, and plottable tukey groups statistics for boxplots
    :param data: Dataframe for Exp data
    :param values: Values to run ANOVA on
    :param groups: Which parameter to group by
    :param alpha: Alpha, usually 0.05
    :return: ANOVA results, Tukey's HSD results, and a dictionary with plottable Tukey significance groups statistics for boxplots
    """
    anova = ols(f"{values} ~ C({groups})", data=data).fit()
    anova_results = sm.stats.anova_lm(anova, typ=2)
    tukey_results = pairwise_tukeyhsd(
        endog=data[values],
        groups=data[groups],
        alpha=alpha
    )

    results = pd.DataFrame(tukey_results.summary().data[1:], columns=tukey_results.summary().data[0])
    groups = tukey_results.groupsunique
    significant_pairs = results[results["reject"] == False][["group1", "group2"]]

    # Initialize groups
    group_labels = {group: set([chr(65 + i)]) for i, group in enumerate(groups)}

    # Merge non-significant groups
    for _, (group1, group2) in significant_pairs.iterrows():
        group_labels[group1] = group_labels[group1].union(group_labels[group2])
        group_labels[group2] = group_labels[group1]

    # Assign final group letters
    tukey_groups = {group: "".join(sorted(group_labels[group])) for group in groups}
    return anova_results, tukey_results, tukey_groups


def plot_single_histogram(data: pd.DataFrame,
                          column: str,
                          units: str = None,
                          savefig: bool = False,
                          filepath: str = None,
                          dpi: int = 300):
    """
    Single histogram for one experiment
    :param data: Dataframe of the experiment
    :param column: Column to plot
    :param units: If specified, will appear on the plot
    :param savefig: If specified, will save the figure
    :param filepath: The path to save the plot
    :param dpi: The dpi of the plot
    :return: Builds a single histogram plot
    """

    # Create a formatted string for the label
    if units:
        column_formatted = f"{column.replace('_', ' ').title()}, {units}"
    else:
        column_formatted = column.replace('_', ' ').title()

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.hist(data[column], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of the {column_formatted}")
    plt.xlabel(f"{column_formatted}")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if savefig:
        filepath = os.path.join(filepath, f"histogram_{column}.png")
        plt.savefig(filepath, dpi=dpi)
    else:
        plt.show()


def plot_single_boxplot(data: pd.DataFrame,
                        sign_gr: dict,
                        x: str,
                        y: str,
                        hue: str,
                        x_units: str = None,
                        y_units: str = None,
                        hue_units: str = None,
                        savefig: bool = False,
                        filepath: str = None,
                        dpi: int = 300):
    """
    Single boxplot for one experiment
    :param data: Dataframe of the experiment
    :param sign_gr: Dictionary that holds the results of a Tukey's HSD generated by get_tukey_significance_groups()
    :param x: The parameter by which the boxes will be plotted
    :param y: The plotted parameter
    :param hue: The parameter that colors the dots in the scatter. Usually, x and hue are interchangeable
    :param x_units: The units of the x-axis
    :param y_units: The units of the y-axis
    :param hue_units: The units of the coloring parameter
    :param savefig: If specified, will save the figure
    :param filepath: The path to save the plot
    :param dpi: The dpi of the plot
    :return: Builds a boxplot plot
    """

    # Create a formatted string for the labels
    if y_units:
        y_formatted = f"{y.replace('_', ' ').title()}, {y_units}"
    else:
        y_formatted = y.replace('_', ' ').title()
    if x_units:
        x_formatted = f"{x.replace('_', ' ').title()}, {x_units}"
    else:
        x_formatted = x.replace('_', ' ').title()
    if hue_units:
        hue_formatted = f"{hue.replace('_', ' ').title()}, {hue_units}"
    else:
        hue_formatted = hue.replace('_', ' ').title()

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        color="white",
        showfliers=False
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette='Set1',
        dodge=True,
        size=8,
        alpha=0.7
    )

    # Add Tukey group annotations
    ylim = plt.gca().get_ylim()  # Get y-axis limits
    y_offset = ylim[1] - (ylim[1] - ylim[0]) * 0.035  # Adjust position as a percentage of the axis range
    for i, value in enumerate(sign_gr.keys()):
        plt.text(
            i,
            y_offset,
            sign_gr[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )

    plt.title(f"Boxplot of {y_formatted} by {x_formatted}")
    plt.xlabel(f"{x_formatted}")
    plt.ylabel(f"{y_formatted}")
    plt.legend(title=f"{hue_formatted}", loc='upper right', bbox_to_anchor=(1, 0.95))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if savefig:
        filepath = os.path.join(filepath, f"boxplot_{y}_{x}.png")
        plt.savefig(filepath, dpi=dpi)
    else:
        plt.show()


def plot_paired_histograms(data1: pd.DataFrame,
                           data2: pd.DataFrame,
                           column: str,
                           units: str = None,
                           savefig: bool = False,
                           filepath: str = None,
                           dpi: int = 300):
    """
    Paired histograms for 2 experiments
    :param data1: Dataframe of Exp. #1
    :param data2: Dataframe of Exp. #2
    :param column: Column to plot
    :param units: If specified, will appear on the plot
    :param savefig: If specified, will save the figure
    :param filepath: The path to save the plot
    :param dpi: The dpi of the plot
    :return: Builds a paired histogram plot
    """

    # Create a str for the labels
    if units:
        column_formatted = f"{column.replace('_', ' ').title()}, {units}"
    else:
        column_formatted = column.replace('_', ' ').title()
    plt.figure(figsize=(12, 6))

    # First experiment
    plt.subplot(1, 2, 1)
    plt.hist(data1[column], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of the {column_formatted} (Exp. #1)")
    plt.xlabel(f"{column_formatted}")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Second experiment
    plt.subplot(1, 2, 2)
    plt.hist(data2[column], bins=20, color='salmon', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of the {column_formatted} (Exp. #2)")
    if units:
        plt.xlabel(f"{column_formatted}")
    else:
        plt.xlabel(f"{column_formatted}")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    if savefig:
        filepath = os.path.join(filepath, f"histogram_{column}.png")
        plt.savefig(filepath, dpi=dpi)
    else:
        plt.show()


def plot_boxplot_pairs(data1: pd.DataFrame,
                       data2: pd.DataFrame,
                       sign_gr1: dict,
                       sign_gr2: dict,
                       x: str,
                       y: str,
                       hue: str,
                       x_units: str = None,
                       y_units: str = None,
                       hue_units: str = None,
                       savefig: bool = False,
                       filepath: str = None,
                       dpi: int = 300):
    """
    Paired boxplots for 2 experiments
    :param data1: Dataframe of Exp. #1
    :param data2: Dataframe of Exp. #2
    :param sign_gr1: Dictionary that holds the results of a Tukey's HSD for Exp. #1 generated by get_tukey_significance_groups()
    :param sign_gr2: Dictionary that holds the results of a Tukey's HSD for Exp. #2 generated by get_tukey_significance_groups()
    :param x: The parameter by which the boxes will be plotted
    :param y: The plotted parameter
    :param hue: The parameter that colors the dots in the scatter. Usually, x and hue are interchangeable
    :param x_units: The units of the x-axis
    :param y_units: The units of the y-axis
    :param hue_units: The units of the coloring parameter:
    :param savefig: If specified, will save the figure
    :param filepath: The path to save the plot
    :param dpi: The dpi of the plot
    :return: Builds a boxplot plot
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    if y_units:
        y_formatted = f"{y.replace('_', ' ').title()}, {y_units}"
    else:
        y_formatted = y.replace('_', ' ').title()
    if x_units:
        x_formatted = f"{x.replace('_', ' ').title()}, {x_units}"
    else:
        x_formatted = x.replace('_', ' ').title()
    if hue_units:
        hue_formatted = f"{hue.replace('_', ' ').title()}, {y_units}"
    else:
        hue_formatted = hue.replace('_', ' ').title()

    # Boxplot 1: Experiment 1
    sns.boxplot(
        ax=axes[0],
        data=data1,
        x=x,
        y=y,
        color="white",
        showfliers=False
    )
    sns.stripplot(
        ax=axes[0],
        data=data1,
        x=x,
        y=y,
        hue=hue,
        palette='Set1',
        dodge=True,
        size=8,
        alpha=0.7
    )

    # Add Tukey group annotations for Experiment 1
    ylim1 = axes[0].get_ylim()  # Get y-axis limits for the second subplot
    y_offset1 = ylim1[1] - (ylim1[1] - ylim1[0]) * 0.035  # Adjust position as a percentage of the axis range
    for i, value in enumerate(sign_gr1.keys()):
        axes[0].text(
            i,
            y_offset1,
            sign_gr1[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )


    axes[0].set_title(f"Boxplot of {y_formatted} by {x_formatted} (Exp. #1)")
    axes[0].set_xlabel(f"{x_formatted}")
    axes[0].set_ylabel(f"{y_formatted}")
    axes[0].legend(title=f"{hue_formatted}",
                   loc='upper right',
                   bbox_to_anchor=(1, 0.95))
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Boxplot 2: Experiment 2
    sns.boxplot(
        ax=axes[1],
        data=data2,
        x=x,
        y=y,
        color="white",
        showfliers=False
    )
    sns.stripplot(
        ax=axes[1],
        data=data2,
        x=x,
        y=y,
        hue=hue,
        palette='Set1',
        dodge=True,
        size=8,
        alpha=0.7
    )

    # Add Tukey group annotations for Experiment 2
    ylim2 = axes[1].get_ylim()  # Get y-axis limits for the second subplot
    y_offset2 = ylim2[1] - (ylim2[1] - ylim2[0]) * 0.035  # Adjust position as a percentage of the axis range

    for i, value in enumerate(sign_gr2.keys()):
        axes[1].text(
            i,
            y_offset2,
            sign_gr2[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )

    axes[1].set_title(f"Boxplot of {y_formatted} by {x_formatted} (Exp. #2)")
    axes[1].set_xlabel(f"{x_formatted}")
    axes[1].set_ylabel("")
    axes[1].legend(title=f"{hue_formatted}",
                   loc='upper right',
                   bbox_to_anchor=(1, 0.95))
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and display the plots
    plt.tight_layout()

    if savefig:
        filepath = os.path.join(filepath, f"boxplot_{y}_{x}.png")
        plt.savefig(filepath, dpi=dpi)
    else:
        plt.show()


def plot_boxplot_coupled(data: pd.DataFrame,
                         sign_gr1: dict,
                         sign_gr2: dict,
                         x1: str,
                         x2: str,
                         y: str,
                         x1_units: str = None,
                         x2_units: str = None,
                         y_units: str = None,
                         savefig: bool = False,
                         filepath: str = None,
                         dpi: int = 300):
    """
    Paired boxplots for two grouping options from one experiment
    :rtype: object
    :param data1: Dataframe of Exp. #1
    :param data2: Dataframe of Exp. #2
    :param sign_gr1: Dictionary that holds the results of a Tukey's HSD for grouping #1 generated by get_tukey_significance_groups()
    :param sign_gr2: Dictionary that holds the results of a Tukey's HSD for grouping #2 generated by get_tukey_significance_groups()
    :param x: The parameter by which the boxes will be plotted
    :param y: The plotted parameter
    :param hue: The parameter that colors the dots in the scatter. Usually, x and hue are interchangeable
    :param x1_units: The units of the x-axis #1
    :param x2_units: The units of the x-axis #2
    :param y_units: The units of the y-axis
    :param hue_units: The units of the coloring parameter:
    :param savefig: If specified, will save the figure
    :param filepath: The path to save the plot
    :param dpi: The dpi of the plot
    :return: Builds a boxplot plot
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    if y_units:
        y_formatted = f"{y.replace('_', ' ').title()}, {y_units}"
    else:
        y_formatted = y.replace('_', ' ').title()
    if x1_units:
        x1_formatted = f"{x1.replace('_', ' ').title()}, {x1_units}"
    else:
        x1_formatted = x1.replace('_', ' ').title()
    if x2_units:
        x2_formatted = f"{x2.replace('_', ' ').title()}, {x2_units}"
    else:
        x2_formatted = x2.replace('_', ' ').title()

    # Boxplot 1: Group 1
    sns.boxplot(
        ax=axes[0],
        data=data,
        x=x1,
        y=y,
        color="white",
        showfliers=False
    )
    sns.stripplot(
        ax=axes[0],
        data=data,
        x=x1,
        y=y,
        hue=x2,
        palette='Set1',
        dodge=True,
        size=8,
        alpha=0.7
    )

    # Add Tukey group annotations for Experiment 1
    ylim1 = axes[0].get_ylim()  # Get y-axis limits for the second subplot
    y_offset1 = ylim1[1] - (ylim1[1] - ylim1[0]) * 0.035  # Adjust position as a percentage of the axis range
    for i, value in enumerate(sign_gr1.keys()):
        axes[0].text(
            i,
            y_offset1,
            sign_gr1[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )

    axes[0].set_title(f"Boxplot of {y_formatted} by {x1_formatted}")
    axes[0].set_xlabel(f"{x1_formatted}")
    axes[0].set_ylabel(f"{y_formatted}")
    axes[0].legend(title=f"{x2_formatted}",
                   loc='upper right',
                   bbox_to_anchor=(1, 0.95))
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Boxplot 2: Group 2
    sns.boxplot(
        ax=axes[1],
        data=data,
        x=x2,
        y=y,
        color="white",
        showfliers=False
    )
    sns.stripplot(
        ax=axes[1],
        data=data,
        x=x2,
        y=y,
        hue=x1,
        palette='Set1',
        dodge=True,
        size=8,
        alpha=0.7
    )

    # Add Tukey group annotations for Experiment 2
    ylim2 = axes[1].get_ylim()  # Get y-axis limits for the second subplot
    y_offset2 = ylim2[1] - (ylim2[1] - ylim2[0]) * 0.035  # Adjust position as a percentage of the axis range

    for i, value in enumerate(sign_gr2.keys()):
        axes[1].text(
            i,
            y_offset2,
            sign_gr2[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )

    axes[1].set_title(f"Boxplot of {y_formatted} by {x2_formatted}")
    axes[1].set_xlabel(f"{x2_formatted}")
    axes[1].set_ylabel("")
    axes[1].legend(title=f"{x1_formatted}",
                   loc='upper right',
                   bbox_to_anchor=(1, 0.95))
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and display the plots
    plt.tight_layout()

    if savefig:
        filepath = os.path.join(filepath, f"boxplot_{y}_{x1}_{x2}.png")
        plt.savefig(filepath, dpi=dpi)
    else:
        plt.show()
