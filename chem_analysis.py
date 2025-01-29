import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_stats as plst
import os


def assign_treatment_and_cultivar(plant_number):
    if 1 <= plant_number <= 15:
        treatment = 'C'
    elif 16 <= plant_number <= 30:
        treatment = 'T1'
    elif 31 <= plant_number <= 45:
        treatment = 'T2'
    elif 46 <= plant_number <= 60:
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


def assign_treatment_and_cultivar(plant_number):
    if 1 <= plant_number <= 15:
        treatment = 'C'
    elif 16 <= plant_number <= 30:
        treatment = 'T1'
    elif 31 <= plant_number <= 45:
        treatment = 'T2'
    elif 46 <= plant_number <= 60:
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
    yield_folder_path = 'G:\My Drive\Dmitrii - Ph.D Thesis\Frost room Experiment Data\Fruit Quality'
    exp_file_path = os.path.join(yield_folder_path, 'chemical_analysis_formatted.xlsx')

    # Set font size
    plt.rcParams.update({'font.size': 16})
    sns.set_context("notebook", font_scale=1.5)

    # Load data
    experiment_data = pd.read_excel(exp_file_path)

    # Preprocess the data
    filtered_experiment = experiment_data[
        experiment_data['group_of_plants'].isin(['control', 'dmitrii'])
    ].copy()

    for x in filtered_experiment.index:
        filtered_experiment.loc[
            x, "date"] = f"{str(filtered_experiment.loc[x, 'year'])}_{str(filtered_experiment.loc[x, 'month'])}_{str(filtered_experiment.loc[x, 'day'])}"

    # Remove rows with negative weight values from both datasets
    df = filtered_experiment[filtered_experiment["date"] != '2023_5_21']
    df = df[df['firmness'] >= 0]
    # All fruit
    print(len(df))

    df = df[df["firmness"] <= 35]
    df = df[df["red_index"] >= 0.17]
    # Red fruit
    print(len(df))



    # Removing rows with missing or blank values in key columns for both datasets
    df = df.dropna(subset=['treatment', 'cultivar'])
    df.reset_index(drop=True, inplace=True)

    # ANOVA and Tukey's HSD
    # All parameters by cultivar
    anova_firmness_cultivar, tukey_firmness_cultivar, tukey_sign_groups_firmness_cultivar = plst.anova_and_tukey_stats(
        df,
        "firmness",
        "cultivar"
    )
    anova_citric_acid_cultivar, tukey_citric_acid_cultivar, tukey_sign_groups_citric_acid_cultivar = plst.anova_and_tukey_stats(
        df,
        "citric_acid",
        "cultivar"
    )
    anova_ph_cultivar, tukey_ph_cultivar, tukey_sign_groups_ph_cultivar = plst.anova_and_tukey_stats(
        df,
        "pH",
        "cultivar"
    )
    anova_tss_cultivar, tukey_tss_cultivar, tukey_sign_groups_tss_cultivar = plst.anova_and_tukey_stats(
        df,
        "TSS",
        "cultivar"
    )
    anova_ascorbic_acid_cultivar, tukey_ascorbic_acid_cultivar, tukey_sign_groups_ascorbic_acid_cultivar = plst.anova_and_tukey_stats(
        df,
        "ascorbic_acid",
        "cultivar"
    )
    # All parameters by treatment
    anova_firmness_treatment, tukey_firmness_treatment, tukey_sign_groups_firmness_treatment = plst.anova_and_tukey_stats(
        df,
        "firmness",
        "treatment"
    )
    anova_citric_acid_treatment, tukey_citric_acid_treatment, tukey_sign_groups_citric_acid_treatment = plst.anova_and_tukey_stats(
        df,
        "citric_acid",
        "treatment"
    )
    anova_ph_treatment, tukey_ph_treatment, tukey_sign_groups_ph_treatment = plst.anova_and_tukey_stats(
        df,
        "pH",
        "treatment"
    )
    anova_tss_treatment, tukey_tss_treatment, tukey_sign_groups_tss_treatment = plst.anova_and_tukey_stats(
        df,
        "TSS",
        "treatment"
    )
    anova_ascorbic_acid_treatment, tukey_ascorbic_acid_treatment, tukey_sign_groups_ascorbic_acid_treatment = plst.anova_and_tukey_stats(
        df,
        "ascorbic_acid",
        "treatment"
    )

    # Plotting
    # Histograms for all parameters
    plst.plot_single_histogram(
        df,
        'firmness'
    )
    plst.plot_single_histogram(
        df,
        'citric_acid',
        '%'
    )
    plst.plot_single_histogram(
        df,
        'pH'
    )
    plst.plot_single_histogram(
        df,
        'TSS'
    )
    plst.plot_single_histogram(
        df,
        'ascorbic_acid',
        'mg/100g'
    )

    # Plotting coupled box plots with Tukey HSD significance annotations
    plst.plot_boxplot_coupled(df,
                              tukey_sign_groups_firmness_cultivar,
                              tukey_sign_groups_firmness_treatment,
                              x1 = 'cultivar',
                              x2 = 'treatment',
                              y = 'firmness')

    # Plotting coupled box plots with Tukey HSD significance annotations
    plst.plot_boxplot_coupled(df,
                              tukey_sign_groups_citric_acid_cultivar,
                              tukey_sign_groups_citric_acid_treatment,
                              x1 = 'cultivar',
                              x2 = 'treatment',
                              y = 'citric_acid',
                              y_units = '%')
    # Plotting coupled box plots with Tukey HSD significance annotations
    plst.plot_boxplot_coupled(df,
                              tukey_sign_groups_ph_cultivar,
                              tukey_sign_groups_ph_treatment,
                              x1 = 'cultivar',
                              x2 = 'treatment',
                              y = 'pH')
    # Plotting coupled box plots with Tukey HSD significance annotations
    plst.plot_boxplot_coupled(df,
                              tukey_sign_groups_tss_cultivar,
                              tukey_sign_groups_tss_treatment,
                              x1 = 'cultivar',
                              x2 = 'treatment',
                              y = 'TSS')
    # Plotting coupled box plots with Tukey HSD significance annotations
    plst.plot_boxplot_coupled(df,
                              tukey_sign_groups_ascorbic_acid_cultivar,
                              tukey_sign_groups_ascorbic_acid_treatment,
                              x1 = 'cultivar',
                              x2 = 'treatment',
                              y = 'ascorbic_acid',
                              y_units = 'mg/100g')