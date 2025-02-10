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
    # Set paths
    filepath2104 = 'unified_21-04.xlsx'
    filepath1605 = 'unified_16-05.xlsx'
    filepath2410 = 'unified_24-10.xlsx'
    filepath2912 = 'unified_29-12.xlsx'

    # Set experiment codenames or numbers, grouping rules, and a list of parameters with values
    exps = [1, 2]
    groups = ['measurement', 'treatment', 'cultivar', 'treatment_cultivar', 'leaf_position']
    values = ['gsw', 'Adyn', 'Edyn']
    units = {'gsw': 'mol m-2 s-1',
             'Adyn': 'µmol m-2 s-1',
             'Edyn': 'mmol m-2 s-1'}

    # Set font size
    plt.rcParams.update({'font.size': 16})
    sns.set_context("notebook", font_scale=1.5)

    # Load data
    df2104 = pd.read_excel(filepath2104)
    df1605 = pd.read_excel(filepath1605)
    df2410 = pd.read_excel(filepath2410)
    df2912 = pd.read_excel(filepath2912)

    for df in [df2104, df2410]:
        for x in df.index:
            df.loc[x, 'measurement'] = 1
    for df in [df1605, df2912]:
        for x in df.index:
            df.loc[x, 'measurement'] = 2

    # Preprocess the data
    df_exp1 = pd.concat([df1605, df2104], ignore_index=True, axis=0)
    df_exp2 = pd.concat([df2410, df2912], ignore_index=True, axis=0)

    for x in df_exp1.index:
        df_exp1.loc[x, 'exp_number'] = 1
    for x in df_exp2.index:
        df_exp2.loc[x, 'exp_number'] = 2

    # Assign treatment and cultivar for the first dataset
    df_exp1[['treatment', 'cultivar']] = df_exp1['plant_number'].apply(
        lambda x: pd.Series(assign_treatment_and_cultivar(x))
    )
    df_exp2[['treatment', 'cultivar']] = df_exp2['plant_number'].apply(
        lambda x: pd.Series(assign_treatment_and_cultivar(x))
    )

    # Create fields for the leaf positions
    for df in [df_exp1, df_exp2]:
        for x in df.index:
            if df.loc[x, 'big/small'] == 1:
                df.loc[x, 'leaf_size'] = 'big'
            elif df.loc[x, 'big/small'] == 2:
                df.loc[x, 'leaf_size'] = 'small'
            if df.loc[x, 'bottom/middle/top'] == 1:
                df.loc[x, 'leaf_position'] = 'bottom'
            elif df.loc[x, 'bottom/middle/top'] == 2:
                df.loc[x, 'leaf_position'] = 'middle'
            elif df.loc[x, 'bottom/middle/top'] == 3:
                df.loc[x, 'leaf_position'] = 'top'
            df.loc[x, 'leaf'] = f"{df.loc[x, 'leaf_position']}_{df.loc[x, 'leaf_size']}"
            df.loc[x, 'treatment_cultivar'] = f"{df.loc[x, 'treatment']}_{df.loc[x, 'cultivar']}"

    columns = ['exp_number',
               'plant_number',
               'cultivar',
               'treatment',
               'measurement',
               'treatment_cultivar',
               'leaf_size',
               'leaf_position',
               'leaf',
               'Adyn',
               'Edyn',
               'gsw']
    df_exp1 = df_exp1[columns]
    df_exp2 = df_exp2[columns]
    df_exp1.reset_index(drop=True, inplace=True)
    df_exp2.reset_index(drop=True, inplace=True)
    df = pd.concat([df_exp1, df_exp2], axis=0, ignore_index=True)

    # Calculate statistics
    descriptive_stats(df_exp1,
                      df_exp2,
                      'gsw')

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
                anovas[exp][group][value], tukeys[exp][group][value], sign_grs[exp][group][value] = plst.anova_and_tukey_stats(
                    df[df['exp_number'] == exp],
                    value,
                    group
                )
                print('next')

    # Plotting
    # Paired histograms
    for value in values:
        plst.plot_paired_histograms(
            df[df['exp_number'] == 1],
            df[df['exp_number'] == 2],
            value,
            units[value])

    # Plotting paired boxplots with Tukey HSD significance annotations
    for group in groups:
        for value in values:
            # Decide hues
            if group == 'treatment':
                hue = 'cultivar'
            else:
                hue = 'treatment'

            plst.plot_boxplot_pairs(df[df['exp_number'] == 1],
                                    df[df['exp_number'] == 2],
                                    sign_grs[1][group][value],
                                    sign_grs[2][group][value],
                                    x=group,
                                    y=value,
                                    hue=hue)

    # Plotting single boxplots for treatment-cultivar groups
    anovas_tc = defaultdict(lambda: defaultdict(dict))
    tukeys_tc = defaultdict(lambda: defaultdict(dict))
    sign_grs_tc = defaultdict(lambda: defaultdict(dict))
    for exp in exps:
        for value in values:
            print(f"Exp {exp}, grouping by treatment and cultivar, parameter: {value}")
            try:
                anovas_tc[exp][value], tukeys_tc[exp][value], sign_grs_tc[exp][value] = plst.anova_and_tukey_stats(
                    df[df['exp_number'] == exp],
                    value,
                    'treatment_cultivar'
                )
            except AttributeError:
                print('something wrong with this one')
            print('next')

    for value in values:
        plst.plot_boxplot_pairs(df[df['exp_number'] == 1],
                                df[df['exp_number'] == 2],
                                sign_grs_tc[1][value],
                                sign_grs_tc[2][value],
                                x='treatment_cultivar',
                                y=value,
                                hue='measurement')