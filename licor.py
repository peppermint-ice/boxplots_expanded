import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_stats as plst
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
    if 1 <= plant_number % 15 <= 5:
        cultivar = 'Mohammed'
    elif 6 <= plant_number % 15 <= 10:
        cultivar = 'Hahms Gelbe'
    elif 11 <= plant_number % 15 <= 15:
        cultivar = 'Red Robin'
    else:
        return None, None
    return treatment, cultivar


if __name__ == '__main__':
    # Set paths
    data_folder_path = 'G:\My Drive\Dmitrii - Ph.D Thesis\Frost room Experiment Data\LiCOR\Data'
    filepath2104 = os.path.join(data_folder_path, 'unified_21-04.xlsx')
    filepath1605 = os.path.join(data_folder_path, 'unified_16-05.xlsx')
    filepath2410 = os.path.join(data_folder_path, 'unified_24-10.xlsx')
    filepath2912 = os.path.join(data_folder_path, 'unified_29-12.xlsx')

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

    # Create fields for the leaf positions
    for x in df_exp1.index:
        df_exp1.loc[x, 'exp_number'] = 1
        if df_exp1.loc[x, 'big/small'] == 1:
            df_exp1.loc[x, 'leaf_size'] = 'big'
        elif df_exp1.loc[x, 'big/small'] == 2:
            df_exp1.loc[x, 'leaf_size'] = 'small'
        if df_exp1.loc[x, 'bottom/middle/top'] == 1:
            df_exp1.loc[x, 'leaf_position'] = 'bottom'
        elif df_exp1.loc[x, 'bottom/middle/top'] == 2:
            df_exp1.loc[x, 'leaf_position'] = 'middle'
        elif df_exp1.loc[x, 'bottom/middle/top'] == 3:
            df_exp1.loc[x, 'leaf_position'] = 'top'
        df_exp1.loc[x, 'leaf'] = f"{df_exp1.loc[x, 'leaf_position']}_{df_exp1.loc[x, 'leaf_size']}"
    for x in df_exp2.index:
        df_exp2.loc[x, 'exp_number'] = 2
        if df_exp2.loc[x, 'big/small'] == 1:
            df_exp2.loc[x, 'leaf_size'] = 'big'
        elif df_exp2.loc[x, 'big/small'] == 2:
            df_exp2.loc[x, 'leaf_size'] = 'small'
        if df_exp2.loc[x, 'bottom/middle/top'] == 1:
            df_exp2.loc[x, 'leaf_position'] = 'bottom'
        elif df_exp2.loc[x, 'bottom/middle/top'] == 2:
            df_exp2.loc[x, 'leaf_position'] = 'middle'
        elif df_exp2.loc[x, 'bottom/middle/top'] == 3:
            df_exp2.loc[x, 'leaf_position'] = 'top'
        df_exp2.loc[x, 'leaf'] = f"{df_exp2.loc[x, 'leaf_position']}_{df_exp2.loc[x, 'leaf_size']}"

    columns = ['exp_number',
               'plant_number',
               'cultivar',
               'treatment',
               'measurement',
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

    # Assign treatment and cultivar for the first dataset
    df_exp1[['treatment', 'cultivar']] = df_exp1['plant_number'].apply(
        lambda x: pd.Series(assign_treatment_and_cultivar(x))
    )
    df_exp2[['treatment', 'cultivar']] = df_exp2['plant_number'].apply(
        lambda x: pd.Series(assign_treatment_and_cultivar(x))
    )

    print(df_exp1.to_string())
    print(df_exp2.to_string())

    # Calculate statistics
    descriptive_stats(df_exp1,
                      df_exp2,
                      'gsw')

    # ANOVA and Tukey's HSD
    anova_exp1_cultivar_adyn, tukey_exp1_cultivar_adyn, tukey_sign_groups_exp1_cultivar_adyn = plst.anova_and_tukey_stats(
        df_exp1,
        "Adyn",
        "cultivar"
    )
    
    anova_exp1_cultivar_edyn, tukey_exp1_cultivar_edyn, tukey_sign_groups_exp1_cultivar_edyn = plst.anova_and_tukey_stats(
        df_exp1,
        "Edyn",
        "cultivar"
    )

    anova_exp1_cultivar_gsw, tukey_exp1_cultivar_gsw, tukey_sign_groups_exp1_cultivar_gsw = plst.anova_and_tukey_stats(
        df_exp1,
        "gsw",
        "cultivar"
    )
    anova_exp2_cultivar_adyn, tukey_exp2_cultivar_adyn, tukey_sign_groups_exp2_cultivar_adyn = plst.anova_and_tukey_stats(
        df_exp2,
        "Adyn",
        "cultivar"
    )

    anova_exp2_cultivar_edyn, tukey_exp2_cultivar_edyn, tukey_sign_groups_exp2_cultivar_edyn = plst.anova_and_tukey_stats(
        df_exp2,
        "Edyn",
        "cultivar"
    )

    anova_exp2_cultivar_gsw, tukey_exp2_cultivar_gsw, tukey_sign_groups_exp2_cultivar_gsw = plst.anova_and_tukey_stats(
        df_exp2,
        "gsw",
        "cultivar"
    )
    
    

    # Plotting
    # Paired histograms for total weight
    plst.plot_paired_histograms(
        df_exp1,
        df_exp2,
        'gsw',
        'µmol m-2 s-1'
    )

    # Paired histograms for total fruits
    plst.plot_paired_histograms(
        df_exp1,
        df_exp2,
        'Adyn',
        'µmol m-2 s-1'
    )
    # Paired histograms for total fruits
    plst.plot_paired_histograms(
        df_exp1,
        df_exp2,
        'Edyn',
        'µmol m-2 s-1'
    )

    # Plotting paired boxplots with Tukey HSD significance annotations
    plst.plot_boxplot_pairs(experiment1_plant_stats,
                            experiment2_plant_stats,
                            tukey_sign_groups_exp1_cultivar,
                            tukey_sign_groups_exp2_cultivar,
                            x='cultivar',
                            y='yield_per_plant',
                            hue='treatment',
                            y_units='g')

    plst.plot_boxplot_pairs(experiment1_plant_stats,
                            experiment2_plant_stats,
                            tukey_sign_groups_exp1_treatment,
                            tukey_sign_groups_exp2_treatment,
                            'treatment',
                            'yield_per_plant',
                            'cultivar')
