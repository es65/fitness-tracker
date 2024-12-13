# Alternative implementation of Random Forest model
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import graphviz
from sklearn.tree import export_graphviz

# Columns: ['participant', 'label', 'feature1', 'feature2', ..., 'feature10']
# 'label' is the exercise label (the target variable), and 'participant' is used for LOGO-CV

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_parquet('../../data/interim/03_data_features.parquet')

labels = list(df['label'].unique())

features = [
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

# Target column
target = 'label'

# Participant column
groups = df['participant']

# Prepare features and target
X = df[features]
y = df[target]

# Set up Leave-One-Group-Out Cross-Validation (LOGO-CV)
logo = LeaveOneGroupOut()

# Initialize the classifier
clf = RandomForestClassifier(random_state=42)

# Store evaluation metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
conf_matrices = []

# Perform LOGO-CV
for train_idx, test_idx in logo.split(X, y, groups=groups):
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Append metrics for reporting
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    conf_matrices.append(conf_matrix)
    
    # Output classification report for this fold
    print(f"Classification Report for Participant Fold:")
    print(classification_report(y_test, y_pred))

# Calculate and print the average metrics across all folds
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average F1-Score: {np.mean(f1_scores):.4f}")

def plot_cm(cm, classes, title='Confusion Matrix', annot=True, fmt='d', cmap='Blues'):
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap,
                xticklabels=classes,
                yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()


for i, cm in enumerate(conf_matrices):
    plot_cm(cm, classes=labels)

# tree.plot_tree is pretty useless without adjustments (which I didn't try)
plt.figure(figsize=(20, 10))  # Set the size for better readability
tree.plot_tree(clf.estimators_[0], feature_names=features, filled=True)
plt.savefig('../../reports/figures/tree.png')
plt.show()

# Export the first tree in the forest to DOT format
tree_dot = export_graphviz(clf.estimators_[87],
                           out_file=None,  # Export to string
                           feature_names=features,
                           filled=True, 
                           rounded=True,
                           special_characters=True)

# Use graphviz to render the tree
graph = graphviz.Source(tree_dot)
graph.render('../../reports/figures/tree_gv3')  # This will save as a PDF or PNG file

len(clf.estimators_)

for train_idx, test_idx in logo.split(X, y, groups=groups):
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f'Participants in test data: {df.iloc[test_idx]['participant'].unique()}')
    
    print(f'Shape X Train: {X_train.shape}')
    print(f'Shape X Train: {X_test.shape}')
    print(f'Shape X Train: {y_train.shape}')
    print(f'Shape X Train: {y_test.shape}')
    
    print(f'Exercises y test: {y_test.unique()}')


df[df['participant'] == 'B']['label'].unique()