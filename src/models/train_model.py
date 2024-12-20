import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

##### Look up k-Fold Cross Validation ######

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features
]

feature_names = [
    'Feature Set 1',
    'Feature Set 2',
    'Feature Set 3',
    'Feature Set 4',
    'Selected Features'
]

iterations = 1
score_df = pd.DataFrame()

### Read up on Grid Search

for i, f in zip(range(len(possible_feature_sets)), feature_names): # just use enumerate()
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

# test accuracy
score_df.sort_values(by='accuracy', ascending=False, inplace=True)
score_df.reset_index(drop=True)

plt.figure(figsize=(10,10))
sns.barplot(x='model', y='accuracy', hue='feature_set', data=score_df)
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.ylim(0.7, 1.0)
plt.legend(loc="lower right")
plt.title('Test Accuracy by Model & Feature Set')
plt.savefig('../../reports/figures/Accuracy_vs_FeatureSet')
plt.show()

# Random forrest wins.

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = list(class_test_prob_y.columns)
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# My approach with sns.heatmap:
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes,
            yticklabels=classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# DE's code for nice confusion matrix display:
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# Not in DE code:

report = classification_report(y_test, class_test_y)
print('\nClassification Report:\n')
print(report)
 

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
df_participant = df.drop(['category', 'set'], axis=1)

participants = df_participant['participant'].unique()
participants.sort()

X_train = df_participant[df_participant['participant'] != 'A'].drop('label', axis=1)
X_test = df_participant[df_participant['participant'] == 'A'].drop('label', axis=1)
y_train = df_participant[df_participant['participant'] != 'A']['label']
y_test = df_participant[df_participant['participant'] == 'A']['label']

'label' in X_train.columns # Expect False
'A' in X_train['participant'].values # Expect False
len(X_train) == len(y_train) # Expect True
len(X_test) == len(y_test) # Expect True

# Should but did not test label dist in test and training sets

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

accuracy_A_rm = accuracy_score(y_test, class_test_y)

report_A_rm = classification_report(y_test, class_test_y)

print(f'Test accuracy for Random Forest with participant A removed: {accuracy_A_rm}')
print('\nClassification Report for participant A removed:\n')
print(report)

###

# Leave-One-Group-Out Cross-Validation (LOGO-CV) using participant
# Avoids data leakage where data from the same participant exercise ends up
# in both training and test data
# DE only did a single instance using participant A
# I implemented manually but can also use sklearn LeaveOneGroupOut()

LOGO_accuracies = {}
LOGO_reports = {}

for p in participants:
    
    X_train = df_participant[df_participant['participant'] != p].drop('label', axis=1)
    X_test = df_participant[df_participant['participant'] == p].drop('label', axis=1)
    y_train = df_participant[df_participant['participant'] != p]['label']
    y_test = df_participant[df_participant['participant'] == p]['label']

    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.random_forest(
        X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
    )

    LOGO_accuracies[p] = a = accuracy_score(y_test, class_test_y)

    LOGO_reports[p] = report = classification_report(y_test, class_test_y)

    print(f'\nTest accuracy for Random Forest with participant {p} removed: {a}')
    print(f'\nClassification Report for participant {p} removed:\n')
    print(report)

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------
 
 # Above

# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------

# Skipped


