# boxplots_expanded
A small library to build single and paired boxplots with Tukey's Post-Hoc HSD notation above them.

The library consists of mostly just one file _plotting_stats.py_ with several functions in it. The other file _licor.py_ is an example. The four .xlsx files are a sample project.

# Features
## ANOVA & Tukey's HSD

Perform one-way ANOVA to detect significant differences between groups.

## Conduct Tukey's HSD post-hoc tests to identify group similarities.

Generate group annotations for visual representation in boxplots.

## Plotting Functions

- Single and paired histograms.

- Single and paired boxplots with Tukey's group annotations.

- Coupled boxplots for two grouping options from the same experiment.

## Descriptive Statistics

Generate summary statistics (min, max, mean, median) for experiments, cultivars, and treatments.

# Installation
```
git clone https://github.com/peppermint-ice/boxplots_expanded.git
cd boxplots_extanded
```
# Dependencies
```
pip install pandas matplotlib seaborn statsmodels
```
# Example
```
import pandas as pd
import plotting_stats as plst

# Load your data
data = pd.read_excel('path/to/your/data.xlsx')

# Perform ANOVA and Tukey's HSD
anova_results, tukey_results, tukey_groups = plst.anova_and_tukey_stats(
    data=data,
    values='gsw',
    groups='treatment'
)

# Plot the boxplot with Tukey annotations
plst.plot_single_boxplot(
    data=data,
    sign_gr=tukey_groups,
    x='treatment',
    y='gsw',
    hue='cultivar',
    y_units='mol m-2 s-1'
)
```
