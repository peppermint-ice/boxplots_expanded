import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind
import os


def anova_and_tukey_stats(data: pd.DataFrame,
                          values: str,
                          groups: str,
                          alpha: float = 0.05):
    """
    Gets ANOVA, Tukey's HSD, and plottable Tukey group statistics for boxplots.
    :param data: DataFrame for experimental data.
    :param values: Column name with values to run ANOVA on.
    :param groups: Column name with categorical groups.
    :param alpha: Significance level (default: 0.05).
    :return: ANOVA results, Tukey's HSD results, and a dictionary with formatted Tukey significance groups.
    """
    # Perform ANOVA
    anova = ols(f"{values} ~ C({groups})", data=data).fit()
    anova_results = sm.stats.anova_lm(anova, typ=2)

    # Perform Tukey's HSD
    tukey_results = pairwise_tukeyhsd(endog=data[values], groups=data[groups], alpha=alpha)
    results = pd.DataFrame(tukey_results.summary().data[1:], columns=tukey_results.summary().data[0])

    # Extract unique groups
    unique_groups = sorted(tukey_results.groupsunique)

    # Dictionary to store letter assignments
    group_letters = {group: set() for group in unique_groups}

    # Start letter assignment
    assigned_letters = []  # List of sets containing grouped treatments
    letter = ord('a')  # ASCII value for 'a'

    for group in unique_groups:
        possible_letters = set()
        for existing_group in assigned_letters:
            # Check if the group is NOT significantly different from any member of an existing group
            non_significant = results[
                ((results["group1"] == group) & (results["group2"].isin(existing_group))) |
                ((results["group2"] == group) & (results["group1"].isin(existing_group)))
                ]["reject"].eq(False).all()

            if non_significant:
                possible_letters.update(group_letters[next(iter(existing_group))])
                existing_group.add(group)

        # If no matching group was found, assign a new letter
        if not possible_letters:
            possible_letters.add(chr(letter))
            assigned_letters.append({group})
            letter += 1

        group_letters[group] = possible_letters

    # Convert sets to sorted, comma-separated strings
    tukey_groups = {group: ", ".join(sorted(group_letters[group])) for group in unique_groups}

    return anova_results, tukey_results, tukey_groups


def anova_and_tukey_stats2(df: pd.DataFrame,
                          y: str,
                          x: str,
                          alpha: float = 0.05):
    """
    Gets one-way ANOVA, Tukey's HSD, and plottable Tukey group statistics for boxplots.

    :param df: DataFrame containing the experimental data.
    :param y: Column name with numeric values to analyze.
    :param x: Column name with categorical groups.
    :param alpha: Significance level for statistical tests (default: 0.05).
    :return: ANOVA results, Tukey's HSD results, and a dictionary with formatted Tukey significance groups.
    """
    # Check if there are at least 2 or more unique values
    if x not in df.columns:
        print(f'Category "{x}" not found, ignoring')
        return
    else:
        unique_values = df[x].unique()
        if len(unique_values) < 2:
            print(f'Category "{x}" too small, ignoring')
            return
        elif len(unique_values) == 2:
            print(f'Category "{x}" has 2 values, running t-test')
            # Extract values for both groups
            group1 = df[df[x] == unique_values[0]][y]
            group2 = df[df[x] == unique_values[1]][y]

            # Compute means
            mean1, mean2 = group1.mean(), group2.mean()

            # Determine which group has the higher mean
            if mean1 >= mean2:
                higher_group, lower_group = unique_values[0], unique_values[1]
            else:
                higher_group, lower_group = unique_values[1], unique_values[0]

            # Run an independent t-test
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)  # Welch’s t-test

            # Store results in a DataFrame
            ttest_results = pd.DataFrame({
                "Statistic": [t_stat],
                "p-value": [p_value]
            })

            # Assign Tukey group labels
            if p_value > alpha:  # Not significantly different
                tukey_groups = {higher_group: "a", lower_group: "a"}
            else:  # Significantly different
                tukey_groups = {higher_group: "a", lower_group: "b"}

            return ttest_results, None, tukey_groups  # Keep same output structure
        else:
            # If there are enough values, start the analysis
            print(f'Running one-way ANOVA for {y} regarding {x}')
            # Calculate means and medians for each value in a category
            rows = []
            for value in unique_values:
                df_value = df[df[x] == value]
                row = {'value': value, 'mean': float(df_value[y].mean())}
                rows.append(row)
            means_df = pd.DataFrame(rows, columns=['value', 'mean'])
            means_df.sort_values(by=['mean'], ascending=False, inplace=True)
            means_df.reset_index(drop=True, inplace=True)

            # Run ANOVA
            anova = ols(f'{y} ~ C({x})', data=df).fit()
            anova_results = sm.stats.anova_lm(anova, typ=2)
            tukey_results = pairwise_tukeyhsd(endog=df[y], groups=df[x], alpha=alpha)
            tukey_df = pd.DataFrame(data=tukey_results.summary().data[1:], columns=tukey_results.summary().data[0])

            # Create
            group_list = means_df['value'].unique()
            letter_order = ord('a')
            letter = chr(letter_order)
            tukey_table = pd.DataFrame({x: group_list, letter: [np.nan for _ in range(len(group_list))]})

            current = 0
            tukey_table[letter] = tukey_table[letter].astype("boolean")
            tukey_table.loc[tukey_table[x] == group_list[current], letter] = True
            finished = False
            while not finished:
                for i in range(0, len(group_list)):
                    if i != current:
                        pair = [group_list[current], group_list[i]]
                        # Find the corresponding value in the DataFrame
                        match = tukey_df[(tukey_df['group1'] == pair[0]) & (tukey_df['group2'] == pair[1])]
                        if match.empty:
                            match = tukey_df[(tukey_df['group1'] == pair[1]) & (tukey_df['group2'] == pair[0])]
                        reject = match['reject'].values[0]
                        # If there are groups that don't differ from the current, give them the same (current) letter
                        if not reject:
                            tukey_table.loc[tukey_table[x] == group_list[i], letter] = True

                # Check if there are groups left with no letter assigned
                #
                nan_rows = tukey_table.iloc[:, 1:].isna().all(
                    axis=1)  # Check rows where all values (except first column) are NaN

                if nan_rows.any():  # If any such row exists
                    current = nan_rows.idxmax()  # Get the index of the first row where all values are NaN
                    letter_order += 1
                    letter = chr(letter_order)
                    tukey_table.loc[tukey_table[x] == group_list[current], letter] = True
                else:
                    finished = True
            columns = tukey_table.columns
            tukey_groups = {}

            for n in tukey_table.index:
                group_name = tukey_table.loc[n, x]  # Get the group name
                tukey_groups[group_name] = ""  # Initialize as an empty string

                for column in sorted(columns[1:]):  # Sort to maintain order stability
                    if pd.notna(tukey_table.loc[n, column]) and tukey_table.loc[n, column]:  # Safe boolean check
                        tukey_groups[group_name] += column  # Append column name directly

            print(tukey_groups)  # Output the final dictionary
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
    plt.hist(data[column], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
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
    for i, value in enumerate(data[x].unique()):
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
    plt.hist(data1[column], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of the {column_formatted} (Exp. #1)")
    plt.xlabel(f"{column_formatted}")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Second experiment
    plt.subplot(1, 2, 2)
    plt.hist(data2[column], bins=10, color='salmon', edgecolor='black', alpha=0.7)
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
    # Set a position for Tukey's notation
    ylim_left = axes[0].get_ylim()  # Get y-axis limits for the second subplot
    ylim_right = axes[1].get_ylim()
    ylim_max = max(ylim_left[1], ylim_right[1])   # Adjust position as a percentage of the axis range
    ylim_min = min(ylim_left[0], ylim_right[0])
    y_offset = ylim_max - (ylim_max - ylim_min) * 0.035  # Adjust position as a percentage of the axis range

    # Add Tukey group annotations for Experiment 1
    for i, value in enumerate(data1[x].unique()):
        axes[0].text(
            i,
            y_offset,
            sign_gr1[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )
    # Add Tukey group annotations for Experiment 2
    for i, value in enumerate(data2[x].unique()):
        axes[1].text(
            i,
            y_offset,
            sign_gr2[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )

    # Format text
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

    # Create plot elements
    axes[0].set_title(f"Boxplot of {y_formatted} by {x_formatted} (Exp. #1)")
    axes[0].set_xlabel(f"{x_formatted}")
    axes[0].set_ylabel(f"{y_formatted}")
    axes[0].legend(title=f"{hue_formatted}",
                   loc='upper right',
                   bbox_to_anchor=(1, 0.95))
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
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
    :param data: Dataframe
    :param sign_gr1: Dictionary that holds the results of a Tukey's HSD for grouping #1 generated by get_tukey_significance_groups()
    :param sign_gr2: Dictionary that holds the results of a Tukey's HSD for grouping #2 generated by get_tukey_significance_groups()
    :param x: The parameter by which the boxes will be plotted
    :param y: The plotted parameter
    :param x1_units: The units of the x-axis #1
    :param x2_units: The units of the x-axis #2
    :param y_units: The units of the y-axis
    :param savefig: If specified, will save the figure
    :param filepath: The path to save the plot
    :param dpi: The dpi of the plot
    :return: Builds a boxplot plot
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
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

    # Set a position for Tukey's notation
    ylim_left = axes[0].get_ylim()  # Get y-axis limits for the subplots
    ylim_right = axes[1].get_ylim()
    ylim_max = max(ylim_left[1], ylim_right[1])   # Find min and max so that both notations are at the same level
    ylim_min = min(ylim_left[0], ylim_right[0])
    y_offset = ylim_max - (ylim_max - ylim_min) * 0.035  # Adjust position as a percentage of the axis range

    # Add Tukey group annotations for Experiment 1
    for i, value in enumerate(sign_gr1.keys()):
        axes[0].text(
            i,
            y_offset,
            sign_gr1[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )
    # Add Tukey group annotations for Experiment 2
    for i, value in enumerate(sign_gr2.keys()):
        axes[1].text(
            i,
            y_offset,
            sign_gr2[value],
            horizontalalignment='center',
            fontsize=16,
            color='black'
        )

    # Format text
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

    # Create plot elements
    axes[0].set_title(f"Boxplot of {y_formatted} by {x1_formatted}")
    axes[0].set_xlabel(f"{x1_formatted}")
    axes[0].set_ylabel(f"{y_formatted}")
    axes[0].legend(title=f"{x2_formatted}",
                   loc='upper right',
                   bbox_to_anchor=(1, 0.95))
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
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


def plot_scatter_with_quartiles(data1: pd.DataFrame,
                                data2: pd.DataFrame,
                                sign_gr1: dict,
                                sign_gr2: dict,
                                x: str,
                                y: str,
                                x_units: str = None,
                                y_units: str = None,
                                savefig: bool = False,
                                filepath: str = None,
                                dpi: int = 300):
    """
    Creates a scatterplot with means and 25/75 quartiles, connecting points with lines.

    :param data1: Dataframe #1
    :param data2: Dataframe #2
    :param sign_gr1: Tukey's significance group dict for #1
    :param sign_gr2: Tukey's significance group dict for #2
    :param x: The categorical variable for x-axis
    :param y: The numerical variable to plot
    :param x_units: Units of the x-axis
    :param y_units: Units of the y-axis
    :param savefig: If True, saves the figure
    :param filepath: The path to save the figure
    :param dpi: The resolution for saving the figure
    """

    # Compute statistics (mean and quartiles)
    stats1 = data1.groupby(x)[y].agg(['mean', lambda q: q.quantile(0.25), lambda q: q.quantile(0.75)]).reset_index()
    stats1.columns = [x, "mean", "q25", "q75"]  # Rename columns
    stats2 = data2.groupby(x)[y].agg(['mean', lambda q: q.quantile(0.25), lambda q: q.quantile(0.75)]).reset_index()
    stats2.columns = [x, "mean", "q25", "q75"]

    # # Get unique cultivars and treatments
    # unique_hues = df[hue].unique()
    # unique_x = df[x].unique()

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot first experiment
    plt.errorbar(stats1[x], stats1["mean"], yerr=[stats1["mean"] - stats1["q25"], stats1["q75"] - stats1["mean"]],
                 fmt='o-', capsize=5, label=f"Exp. #1 ({y})", color='blue')

    # Plot second experiment
    plt.errorbar(stats2[x], stats2["mean"], yerr=[stats2["mean"] - stats2["q25"], stats2["q75"] - stats2["mean"]],
                 fmt='o-', capsize=5, label=f"Exp. #2 ({y})", color='red')

    # Annotate Tukey groups **above Q75**
    for i, row in stats1.iterrows():
        plt.text(i + 0.035, row["q75"] - 0.04 * (stats1["q75"].max() - stats1["q25"].min()),
                 sign_gr1.get(row[x], ""), ha='left')

    for i, row in stats2.iterrows():
        plt.text(i + 0.035, row["q75"] - 0.04 * (stats2["q75"].max() - stats2["q25"].min()),
                 sign_gr2.get(row[x], ""), ha='left')

    # Labels & title
    plt.xlabel(x.replace("_", " ").title() + (f", {x_units}" if x_units else ""))
    plt.ylabel(y.replace("_", " ").title() + (f", {y_units}" if y_units else ""))
    plt.title(f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)

    # Save or Show
    if savefig:
        plt.savefig(filepath, dpi=dpi)
    else:
        plt.show()


def plot_multiple_scatterplots(df: pd.DataFrame,
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
    Creates a scatterplot with means and 25/75 quartiles, connecting points with lines.

    :param df: pd.DataFrame with data
    :param x: The parameter by which the boxes will be plotted
    :param y: The plotted parameter
    :param hue: The parameter that colors the dots in the scatter. Usually, x and hue are interchangeable
    :param x_units: The units of the x-axis
    :param y_units: The units of the y-axis
    :param hue_units: The units of the coloring parameter:
    :param savefig: If specified, will save the figure
    :param filepath: The path to save the plot
    :param dpi: The dpi of the plot
    :return:
    """

    # Format text
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

    # Create figure
    plt.figure(figsize=(12, 6))

    # Get unique cultivars and treatments
    cultivars = df[hue].unique()
    treatments = df[x].unique()

    # Choose a colormap
    colormap = plt.cm.viridis  # Change to plt.cm.magma for Magma

    # Generate colors from the colormap
    colors = colormap(np.linspace(0, 1, len(cultivars)))

    # Loop over each cultivar and plot its data
    for i, cultivar in enumerate(cultivars):
        # Filter data for this cultivar
        subset = df[df[hue] == cultivar]

        # Compute statistics (mean and quartiles) per treatment
        stats = subset.groupby(x)[y].agg(['mean', lambda q: q.quantile(0.25), lambda q: q.quantile(0.75)]).reset_index()
        stats.columns = [x, "mean", "q25", "q75"]  # Rename columns

        # Convert categorical treatments to numerical positions
        x_positions = np.array([list(treatments).index(t) for t in stats[x]])  # Keep categorical order

        # Apply small jitter (shift cultivars left/right slightly)
        x_jitter = (i - len(cultivars) / 2) * 0.03  # Small offset based on cultivar index
        x_positions = x_positions + x_jitter  # Apply jitter

        # Plot mean values with error bars (thicker dashes for quartiles)
        plt.errorbar(x_positions, stats["mean"],
                     yerr=[stats["mean"] - stats["q25"], stats["q75"] - stats["mean"]],
                     fmt='o-', capsize=8, capthick=3, elinewidth=2.5, label=cultivar, color=colors[i])

    # Set categorical x-axis labels
    plt.xticks(ticks=np.arange(len(treatments)), labels=treatments)

    # Labels & title
    plt.title(f"{y_formatted} by {x_formatted}")
    plt.xlabel(f"{x_formatted}")
    plt.ylabel(f"{y_formatted}")
    plt.legend(title=hue.replace("_", " ").title())
    plt.grid(True)

    # Show plot
    plt.show()

