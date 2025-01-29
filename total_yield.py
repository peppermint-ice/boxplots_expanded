import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_stats as plst
import os


def descriptive_stats(data1: pd.DataFrame,
                      data2: pd.DataFrame):
    """
    NOT USABLE GLOBALLY
    Prints a descriptive statistics about two dataframes for cultivar and treatment.
    :param data1: A dataframe for Exp. #1.
    :param data2: A dataframe for Exp. #2.
    :return: Just a print
    """
    experiment1_stats = data1['weight'].agg(['min', 'max', 'mean', 'median']).to_frame(
        name="Experiment 1")
    experiment2_stats = data2['weight'].agg(['min', 'max', 'mean', 'median']).to_frame(
        name="Experiment 2")

    # Combine stats into a single table for easier comparison
    combined_stats = pd.concat(
        [experiment1_stats, experiment2_stats], axis=1)

    # Calculate statistics for each cultivar (disregarding treatment) and each treatment (disregarding cultivar)
    experiment1_cultivar_stats = data1.groupby('cultivar')['weight'].agg(
        ['min', 'max', 'mean', 'median']).reset_index()
    experiment1_treatment_stats = data1.groupby('treatment')['weight'].agg(
        ['min', 'max', 'mean', 'median']).reset_index()
    experiment2_cultivar_stats = data2.groupby('cultivar')['weight'].agg(
        ['min', 'max', 'mean', 'median']).reset_index()
    experiment2_treatment_stats = data2.groupby('treatment')['weight'].agg(
        ['min', 'max', 'mean', 'median']).reset_index()

    print(combined_stats.to_string())
    print("Exp. #1")
    print(experiment1_cultivar_stats.to_string())
    print(experiment1_treatment_stats.to_string())
    print("Exp. #2")
    print(experiment2_cultivar_stats.to_string())
    print(experiment2_treatment_stats.to_string())


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
    yield_folder_path = 'G:\My Drive\Dmitrii - Ph.D Thesis\Frost room Experiment Data\Yield'
    exp_1_folder_path = os.path.join(yield_folder_path, '2023_1')
    exp_2_folder_path = os.path.join(yield_folder_path, '2024_1')
    exp_1_file_path = os.path.join(exp_1_folder_path, 'frostroom2_log_final.xlsx')
    exp_2_file_path = os.path.join(exp_2_folder_path, 'frostroom3_log_final.xlsx')

    # Set font size
    plt.rcParams.update({'font.size': 16})
    sns.set_context("notebook", font_scale=1.5)

    # Load data
    experiment1_data = pd.read_excel(exp_1_file_path)
    experiment2_data = pd.read_excel(exp_2_file_path)

    # Preprocess the data
    filtered_experiment1 = experiment1_data[
        experiment1_data['group_of_plants'].isin(['control', 'dmitrii'])
    ].copy()

    # Deal with clustered data rows
    expanded_rows = []
    for _, row in filtered_experiment1.iterrows():
        if row['weight'] == 0 and not pd.isna(row['cluster count']) and not pd.isna(row['cluster_weight']):
            cluster_count = int(row['cluster count'])
            cluster_weight = row['cluster_weight']
            weight_per_plant = cluster_weight / cluster_count
            for _ in range(cluster_count):
                new_row = row.copy()
                new_row['weight'] = weight_per_plant
                expanded_rows.append(new_row)
        else:
            expanded_rows.append(row)
    experiment1_expanded = pd.DataFrame(expanded_rows)

    # Assign treatment and cultivar for the first dataset
    experiment1_expanded[['treatment', 'cultivar']] = experiment1_expanded['plant_number'].apply(
        lambda x: pd.Series(assign_treatment_and_cultivar(x))
    )
    # For the second dataset
    experiment2_data[['treatment', 'cultivar']] = experiment2_data['Plant Number (1-60):'].apply(
        lambda x: pd.Series(assign_treatment_and_cultivar(x))
    )

    # Create final tables with the required columns
    experiment1_final = experiment1_expanded[['plant_number', 'treatment', 'cultivar', 'weight']]
    experiment2_final = experiment2_data[['Plant Number (1-60):', 'treatment', 'cultivar', 'Weight (gr)']]
    experiment2_final.columns = ['plant_number', 'treatment', 'cultivar', 'weight']

    # Removing rows with missing or blank values in key columns for both datasets
    experiment1_final_cleaned = experiment1_final.dropna(subset=['plant_number', 'treatment', 'cultivar', 'weight'])
    experiment2_final_cleaned = experiment2_final.dropna(subset=['plant_number', 'treatment', 'cultivar', 'weight'])
    experiment1_final_cleaned.reset_index(drop=True, inplace=True)
    experiment2_final_cleaned.reset_index(drop=True, inplace=True)

    # Remove rows with negative weight values from both datasets
    df_exp1 = experiment1_final_cleaned[experiment1_final_cleaned['weight'] >= 0]
    df_exp2 = experiment2_final_cleaned[experiment2_final_cleaned['weight'] >= 0]

    # Calculate the number of fruits and total weight for each plant in both datasets
    experiment1_plant_stats = df_exp1.groupby('plant_number').agg(
        number_of_fruits_per_plant=('weight', 'count'),
        yield_per_plant=('weight', 'sum')
    ).reset_index()

    experiment2_plant_stats = df_exp2.groupby('plant_number').agg(
        number_of_fruits_per_plant=('weight', 'count'),
        yield_per_plant=('weight', 'sum')
    ).reset_index()
    # Merge plant-level stats with treatments and cultivars for both datasets
    experiment1_plant_stats = experiment1_plant_stats.merge(
        df_exp1[['plant_number', 'treatment', 'cultivar']].drop_duplicates(),
        on='plant_number',
        how='left'
    )

    experiment2_plant_stats = experiment2_plant_stats.merge(
        df_exp2[['plant_number', 'treatment', 'cultivar']].drop_duplicates(),
        on='plant_number',
        how='left'
    )

    # Calculate statistics
    descriptive_stats(df_exp1,
                      df_exp2)

    # ANOVA and Tukey's HSD
    anova_exp1_cultivar, tukey_exp1_cultivar, tukey_sign_groups_exp1_cultivar = plst.anova_and_tukey_stats(
        experiment1_plant_stats,
        "yield_per_plant",
        "cultivar"
    )
    anova_exp2_cultivar, tukey_exp2_cultivar, tukey_sign_groups_exp2_cultivar = plst.anova_and_tukey_stats(
        experiment2_plant_stats,
        "yield_per_plant",
        "cultivar"
    )
    anova_exp1_treatment, tukey_exp1_treatment, tukey_sign_groups_exp1_treatment = plst.anova_and_tukey_stats(
        experiment1_plant_stats,
        "yield_per_plant",
        "treatment"
    )
    anova_exp2_treatment, tukey_exp2_treatment, tukey_sign_groups_exp2_treatment = plst.anova_and_tukey_stats(
        experiment2_plant_stats,
        "yield_per_plant",
        "treatment"
    )

    # Plotting
    # Paired histograms for total weight
    plst.plot_paired_histograms(
        experiment1_plant_stats,
        experiment2_plant_stats,
        'yield_per_plant',
        'g'
    )

    # Paired histograms for total fruits
    plst.plot_paired_histograms(
        experiment1_plant_stats,
        experiment2_plant_stats,
        'number_of_fruits_per_plant'
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
