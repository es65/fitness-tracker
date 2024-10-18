import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_parquet('../../data/interim/03_data_features.parquet')

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(['participant', 'category', 'set'], axis=1)

X = df_train.drop('label', axis=1)
y = df_train['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y)

fig, ax = plt.subplots(figsize=(10,5))
df_train['label'].value_counts().plot(
    kind='bar', ax=ax, color='lightblue', label='All'
)
y_train.value_counts().plot(
    kind='bar', ax=ax, color='dodgerblue', label='Train'
)
y_test.value_counts().plot(
    kind='bar', ax=ax, color='royalblue', label='Test'
)
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
square_features = ['acc_r', 'gyr_r']
pca_features = ['pca_1', 'pca_2', 'pca_3']
time_features = [f for f in df_train.columns if 'temp' in f]
freq_features = [f for f in df_train.columns if ('freq' in f) or ('pse' in f)]
cluster_features = ['cluster']

feature_set_1 = basic_features
feature_set_2 = list(set(basic_features + square_features + pca_features))# set() avoids duplicates.
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()

max_features = 10

# Completed in 4m 48s and used an entire CPU core. Could optimize to run on multiple cores?
# Look at n_jobs =-1 param in the SKLearn DecisionTreeClassifier to use all cores.
# Look at Parallel and delayed from joblib for training multiple trees in parallel.
# Prob need to change the way forward_selection() is implemented 
# My selected features are totally different from DE! Why??
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

selected_features = [
    'pca_1',
    'duration',
    'acc_x_freq_0.0_Hz_ws_14',
    'acc_z_freq_0.0_Hz_ws_14',
    'gyr_z_freq_1.071_Hz_ws_14',
    'acc_z_freq_2.143_Hz_ws_14',
    'gyr_r_freq_0.0_Hz_ws_14',
    'acc_y_freq_0.714_Hz_ws_14',
    'acc_x_temp_mean_ws_5',
    'gyr_z_freq_2.5_Hz_ws_14'
]

plt.figure(figsize=(10,5))
plt.plot(np.arange(1, max_features+1, 1), ordered_scores)
plt.xlabel('Number of Features')
plt.ylabel('Training Accuracy')
plt.xticks(np.arange(1, max_features+1, 1))
plt.title('Forward feature selection using simple decision tree')
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------