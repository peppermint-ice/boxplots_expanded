import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_stats as plst
from collections import defaultdict
import os


def descriptive_stats(data1: pd.DataFrame,
                      data2: pd.DataFrame,
                      value: str):
    """
    NOT USABLE GLOBALLY
    Prints a descriptive statistics about two dataframes for cultivar and treatment.
    :param data1: A dataframe for Exp. #1.
    :param data2: A dataframe for Exp. #2.
    :param value: A value to be observed.
    :return: Just a print
    """
    experiment1_stats = data1[value].agg(['min', 'max', 'mean', 'median']).to_frame(
        name="Experiment 1")
    experiment2_stats = data2[value].agg(['min', 'max', 'mean', 'median']).to_frame(
        name="Experiment 2")

    # Combine stats into a single table for easier comparison
    combined_stats = pd.concat(
        [experiment1_stats, experiment2_stats], axis=1)

    # Calculate statistics for each cultivar (disregarding treatment) and each treatment (disregarding cultivar)
    experiment1_cultivar_stats = data1.groupby('cultivar')[value].agg(
        ['min', 'max', 'mean', 'median']).reset_index()
    experiment1_treatment_stats = data1.groupby('treatment')[value].agg(
        ['min', 'max', 'mean', 'median']).reset_index()
    experiment2_cultivar_stats = data2.groupby('cultivar')[value].agg(
        ['min', 'max', 'mean', 'median']).reset_index()
    experiment2_treatment_stats = data2.groupby('treatment')[value].agg(
        ['min', 'max', 'mean', 'median']).reset_index()

    print(combined_stats.to_string())
    print("Exp. #1")
    print(experiment1_cultivar_stats.to_string())
    print(experiment1_treatment_stats.to_string())
    print("Exp. #2")
    print(experiment2_cultivar_stats.to_string())
    print(experiment2_treatment_stats.to_string())


def assign_treatment_and_cultivar(plant_number):
    if plant_number < 16:
        treatment = 'C'
    elif plant_number < 31:
        treatment = 'T1'
    elif plant_number < 46:
        treatment = 'T2'
    elif plant_number < 61:
        treatment = 'T3'
    else:
        return None, None
    mod_val = plant_number % 15
    if mod_val == 0 or 1 <= mod_val <= 5:
        cultivar = 'Mohammed'
    elif 6 <= mod_val <= 10:
        cultivar = 'Hahms Gelbe'
    else:  # 11 <= mod_val <= 14 (since 0 is handled above)
        cultivar = 'Red Robin'

    return treatment, cultivar


if __name__ == '__main__':
    data_folder_path = 'C:/Users/dusen/Documents/PhD/Fruit Quality'
    file_name = 'licor.csv'
    categories = ['treatment']

    file_path = os.path.join(data_folder_path, file_name)
    df = pd.read_csv(file_path)
    print(df.to_string())

    # Set experiment codenames or numbers, grouping rules, and a list of parameters with values
    exps = ["summer", "winter"]
    groups = ['treatment', 'cultivar', 'leaf_position']
    values = ['gsw', 'Adyn', 'Edyn']
    units = {'gsw': 'mol m-2 s-1',
             'Adyn': 'µmol m-2 s-1',
             'Edyn': 'mmol m-2 s-1'}

    # Set font size
    plt.rcParams.update({'font.size': 16})
    sns.set_context("notebook", font_scale=1.5)


    # ANOVA and Tukey's HSD

    # Create nested dictionaries for all function outputs
    anovas = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    tukeys = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    sign_grs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Run ANOVAs
    for exp in exps:
        for group in groups:
            for value in values:
                print(f"Exp {exp}, grouping by {group}, parameter: {value}")
                anovas[exp][group][value], tukeys[exp][group][value], sign_grs[exp][group][value] = plst.anova_and_tukey_stats2(
                    df[df['season'] == exp],
                    value,
                    group
                )
                print(anovas[exp][group][value])
                print(tukeys[exp][group][value])
                print(sign_grs[exp][group][value])
                print('next')
        for group in groups:
            for value in values:
                print(f"All data, grouping by {group}, parameter: {value}")
                anovas['all'][group][value], tukeys['all'][group][value], sign_grs['all'][group][
                    value] = plst.anova_and_tukey_stats2(
                    df,
                    value,
                    group
                )
                print('next')


    # Plotting
    # Paired histograms
    for value in values:
        plst.plot_paired_histograms(
            df[df['season'] == "summer"],
            df[df['season'] == "winter"],
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

            plst.plot_scatter_with_quartiles(df[df['season'] == 'summer'],
                                             df[df['season'] == 'winter'],
                                             sign_grs['summer'][group][value],
                                             sign_grs['winter'][group][value],
                                             x=group,
                                             y=value)

            plst.plot_single_boxplot(df,
                                     sign_grs['all'][group][value],
                                     x=group,
                                     y=value,
                                     hue=hue)

    plst.plot_multiple_scatterplots(df,
                                    x='treatment',
                                    y='gsw',
                                    hue='leaf_position',
                                    y_units='mol m-2 s-1')