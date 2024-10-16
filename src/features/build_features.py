import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
data_path = '../../data/interim/02_outliers_removed_chauvenet.parquet'
df = pd.read_parquet(data_path)

predictor_columns = list(df.columns[:6])

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

subset = df[df['set'] == 35]['gyr_y']

# Use pandas interpolate fn to fill NaN values in predictor cols:
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df['set'] == 25]['acc_y'].plot()
df[df['set'] == 50]['acc_y'].plot()

duration = df[df['set'] == 1].index[-1] - df[df['set'] == 1].index[0]

duration.seconds

for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]
    duration = stop - start
    df.loc[(df['set'] == s), 'duration'] = duration.seconds

duration_df = df.groupby(['category'])['duration'].mean()

duration_heavy = duration_df.iloc[0] / 5 # 5 reps per heavy set
duration_medium = duration_df.iloc[1] / 10 # 10 reps per medium set

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

LowPass = LowPassFilter()

fs = 1000/200 # 200 ms per sample so 5 Hz
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, 'acc_y', fs, cutoff, order=5)

subset = df_lowpass[df_lowpass['set'] == 45]

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset['acc_y'].reset_index(drop=True), label='raw_data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), label='butterworth filter')
ax[0].legend(loc='upper center',  bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc='upper center',  bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + '_lowpass']
    del df_lowpass[col + '_lowpass']

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# Use Elbow technique to determine optimum num components. Elbow is where rate of changes of variance diminishes

plt.figure(figsize=(10,10))
plt.bar(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel('Principal Component Num')
plt.ylabel('Explained Variance')
plt.savefig('../../reports/figures/PCA.png')
plt.show()
# DE concludes we should use 3 components but I think the answer is 2, is they account for 85% of variance. PC3 only adds 6.5%.

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca['set'] == 35]

subset[['pca_1', 'pca_2', 'pca_3']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
# Calculate cols for magnitude (r) of gyr and acc - allows impartiality to device orientation
# and change of orientation during a set.
# Too much unnecessary copying of dfs IMO. Eliminate in applications to bigger datasets.

df_squared = df_pca.copy()

acc_r2 = df_squared['acc_x'] ** 2 + df_squared['acc_y'] ** 2 + df_squared['acc_z'] ** 2
gyr_r2 = df_squared['gyr_x'] ** 2 + df_squared['gyr_y'] ** 2 + df_squared['gyr_z'] ** 2

df_squared['acc_r'] = np.sqrt(acc_r2)
df_squared['gyr_r'] = np.sqrt(gyr_r2)

subset = df_squared[df_squared['set'] == 14]

subset[['acc_r', 'gyr_r']].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction: Rolling avg
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns += ['acc_r', 'gyr_r']

ws = int(1000/200) # 1.0 s = 5 steps

# This applies the rolling function over the entire column
# Will introduce noise a start of each set by including data from prior exercise
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, 'mean')
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, 'std')

# Apply rolling function to each set independently
df_temporal_list = [] # Will collect a df for each set with rolled cols
for s in df_temporal['set'].unique():
    subset = df_temporal[df_temporal['set'] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, 'mean')
        subset = NumAbs.abstract_numerical(subset, [col], ws, 'std')
    df_temporal_list.append(subset)
    
df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

# Use center=True for rolling func?

subset[['acc_y', 'acc_y_temp_mean_ws_5', 'acc_y_temp_std_ws_5']].plot()

subset.columns

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index(drop=False)
FreqAbs = FourierTransformation() 

fs = int(1000/200)
ws = int(2800/200) # avg time for a set / sampling interval in ms

df_freq = FreqAbs.abstract_frequency(df_freq, ['acc_y'], ws, fs)

subset = df_freq[df_freq['set'] == 15]
subset[['acc_y']].plot()
subset[
    [
        'acc_y_max_freq',
        'acc_y_freq_weighted',
        'acc_y_pse',
        'acc_y_freq_1.429_Hz_ws_14',
        'acc_y_freq_2.5_Hz_ws_14'
    ]
].plot()

df_freq_list = []
for s in df_freq['set'].unique():
    print(f'Applying Fourier transformation to set {s}')
    subset = df_freq[df_freq['set'] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index('epoch (ms)', drop=True)

df_freq.columns

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# Remove rows with missing data (NaNs)
df_freq = df_freq.dropna()

# df.iloc[start:stop:step]
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ['acc_x', 'acc_y', 'acc_z']
k_values = range(2,10)
intertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    intertias.append(kmeans.inertia_) # sum of squared distances of samples to cluster centers (centroids)

plt.figure(figsize=(10,10))
plt.plot(k_values, intertias)
plt.xlabel('k')
plt.ylabel('Inertias (Sum of Squared Distances from Centroids)')
plt.savefig('../../reports/figures/kmeans_inertias.png')
plt.show()

# Select 5 clusters using elbow method from inertias vs k plot

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster['cluster'] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=c)
ax.set_xlabel('acc_x')
ax.set_ylabel('acc_y')
ax.set_zlabel('acc_z')
plt.title('k=5 Means Clustering')
plt.legend()
plt.savefig('../../reports/figures/Kmeans.png')
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for ex in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == ex]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=ex)
ax.set_xlabel('acc_x')
ax.set_ylabel('acc_y')
ax.set_zlabel('acc_z')
plt.title('Accelleration by Exercise')
plt.legend()
plt.savefig('../../reports/figures/accel_exercise.png')
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_parquet('../../data/interim/03_data_features.parquet')