import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_stats as plst
from collections import defaultdict
import os


if __name__ == '__main__':
    data_folder_path = 'C:/Users/dusen/Documents/PhD/Fruit Quality'
    file_name = 'fruit_quality.csv'
    categories = ['treatment']

    file_path = os.path.join(data_folder_path, file_name)
    df = pd.read_csv(file_path)
    print(df.to_string())

    groups = ['treatment', 'cultivar']
    values = ['TSS', 'citric_acid']
    units = {'TSS': '',
             'citric_acid': ''}

    # Set font size
    plt.rcParams.update({'font.size': 16})
    sns.set_context("notebook", font_scale=1.5)

    # Create nested dictionaries for all function outputs
    anovas = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    tukeys = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    sign_grs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Run ANOVAs
    for group in groups:
        for value in values:
            print(f"Grouping by {group}, parameter: {value}")
            anovas[group][value], tukeys[group][value], sign_grs[group][value] = plst.anova_and_tukey_stats2(
                df,
                value,
                group
            )
            print('next')

    # Plotting
    # Histograms
    for value in values:
        plst.plot_single_histogram(
            df,
            value,
            units[value])

    # Plotting boxplots with Tukey HSD significance annotations
    for group in groups:
        for value in values:
            # Decide hues
            if group == 'treatment':
                hue = 'cultivar'
            else:
                hue = 'treatment'

            plst.plot_single_boxplot(df,
                                     sign_grs[group][value],
                                     x=group,
                                     y=value,
                                     hue=hue,
                                     hue_units=units[value])

    plst.plot_multiple_scatterplots(df,
                                    x='treatment',
                                    y='TSS',
                                    hue='cultivar')