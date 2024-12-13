import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_parquet('../../data/interim/01_data_processed.parquet')
df = df[df['label'] != 'rest']

acc_r2 = df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2
gyr_r2 = df['gyr_x'] ** 2 + df['gyr_y'] ** 2 + df['gyr_z'] ** 2

df['acc_r'] = np.sqrt(acc_r2)
df['gyr_r'] = np.sqrt(gyr_r2)

df['label'].unique()

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

df_bench = df[df['label'] == 'bench']
df_squat = df[df['label'] == 'squat']
df_row = df[df['label'] == 'row']
df_ohp = df[df['label'] == 'ohp']
df_dead = df[df['label'] == 'dead'] 

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

df_plot = df_bench

df_plot[df_plot['set'] == df_plot['set'].unique()[0]]['acc_x'].plot()
df_plot[df_plot['set'] == df_plot['set'].unique()[0]]['acc_y'].plot()
df_plot[df_plot['set'] == df_plot['set'].unique()[0]]['acc_z'].plot()
df_plot[df_plot['set'] == df_plot['set'].unique()[0]]['acc_r'].plot()

df_plot[df_plot['set'] == df_plot['set'].unique()[0]]['gyr_x'].plot()
df_plot[df_plot['set'] == df_plot['set'].unique()[0]]['gyr_y'].plot()
df_plot[df_plot['set'] == df_plot['set'].unique()[0]]['gyr_z'].plot()
df_plot[df_plot['set'] == df_plot['set'].unique()[0]]['gyr_r'].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000/200
LowPass = LowPassFilter()

df_bench[df_bench['set'] == df_bench['set'].unique()[0]]['gyr_r'].plot()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = df_bench[df_bench['set'] == df_bench['set'].unique()[0]]
squat_set = df_squat[df_squat['set'] == df_squat['set'].unique()[0]]
row_set = df_row[df_row['set'] == df_row['set'].unique()[0]]
ohp_set = df_ohp[df_ohp['set'] == df_ohp['set'].unique()[0]]
dead_set = df_dead[df_dead['set'] == df_dead['set'].unique()[0]]

bench_set['acc_r'].plot()

column = 'acc_r'
LowPass.low_pass_filter(
    squat_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10
)[column + '_lowpass'].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


def count_reps(
    dataset: pd.DataFrame,
    sampling_frequency: int,
    cutoff_frequency: float = 0.4,
    order: int = 10,
    column: str = 'acc_r',
    plot: bool = True,
    fig_name: Optional[str] = None
    ) -> int:

    data = LowPass.low_pass_filter(
        dataset,
        col=column,
        sampling_frequency=sampling_frequency,
        cutoff_frequency=cutoff_frequency,
        order=order
    )
    indices = argrelextrema(data[column + '_lowpass'].values, np.greater)
    peaks = data.iloc[indices]
    
    if plot:
        fig, ax = plt.subplots()
        plt.plot(dataset[f'{column}_lowpass'], label='Lowpass')
        plt.plot(dataset[f'{column}'], color='gray', linestyle='--', label='Raw')
        plt.plot(peaks[f'{column}_lowpass'], 'o', color='red', label='Peaks')
        ax.set_ylabel(f'{column}')
        exercise = dataset['label'].iloc[0].title()
        category = dataset['category'].iloc[0].title()
        plt.title(f'{category} {exercise}: {len(peaks)} Reps (cutoff: {cutoff_frequency})')
        plt.legend()
        if fig_name:
            plt.savefig(fig_name)
        plt.show()
    
    return len(peaks)

path = '../../reports/figures/'

count_reps(bench_set, 1000/200, cutoff_frequency=0.47, fig_name=path+'count_bench.png')
count_reps(squat_set, 1000/200, cutoff_frequency=0.38, fig_name=path+'count_squat.png')
count_reps(row_set, 1000/200, cutoff_frequency=0.75, fig_name=path+'count_row.png')
count_reps(row_set, 1000/200, cutoff_frequency=0.7, column='gyr_x', fig_name=path+'count_row_gyr_x.png')
count_reps(ohp_set, 1000/200, cutoff_frequency=0.45, fig_name=path+'count_ohp.png')
count_reps(dead_set, 1000/200, cutoff_frequency=0.4, fig_name=path+'count_dead.png')

# Used slightly different values from DE
cfs = {
    'bench': 0.47, 
    'ohp': 0.45, 
    'squat': 0.38, 
    'dead': 0.4, 
    'row': 0.75
}


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df['reps'] = df['category'].apply(lambda x: 5 if x=='heavy' else 10)

rep_df = df.groupby(['label', 'category', 'set']).max().reset_index()
rep_df['reps_pred'] = 0

for s in df['set'].unique():
    subset = df[df['set'] == s]
    label = subset['label'][0]
    cf = cfs[label]
    reps_pred = count_reps(subset, 1000/200, cutoff_frequency=cf)
    rep_df.loc[rep_df['set'] == s, 'reps_pred'] = reps_pred


i = 60
rep_df[['reps', 'reps_pred']].iloc[i:i+20,]


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

# avg number of reps the prediction is off by
erorr = mean_absolute_error(rep_df['reps'], rep_df['reps_pred']).round(2)

reps_grouped = rep_df.groupby(['label', 'category'])[['reps', 'reps_pred']].mean().reset_index()

reps_grouped.plot.bar(x=['label', 'category'])