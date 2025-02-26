import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load Waterfowl Dataset
csv = "WaterfowlData.csv"
dataset = pd.read_csv(csv, on_bad_lines='skip', engine='python')

# Data Cleaning
columns_to_drop = ["battery-charge-percent", "battery-charging-current", "gps-time-to-fix", 
                   "orn:transmission-protocol", "tag-voltage", "sensor-type", "visible", "tag-local-identifier"]
dataset = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns])

# Convert categorical data to numeric
for col in dataset.columns:
    dataset[col], _ = pd.factorize(dataset[col])

# Split dataset
y = dataset['event-id']
X = dataset.drop(columns=['event-id'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, random_state=1)

# Convert to numpy arrays
X_train, X_validate, X_test = X_train.to_numpy(), X_validate.to_numpy(), X_test.to_numpy()
y_train, y_validate, y_test = y_train.to_numpy(), y_validate.to_numpy()

# Function to plot learning curve
def plot_learning_curve(model, title, X, y, ax=None):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    train_scores_mean, train_scores_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_scores_mean, test_scores_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="purple")
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color='purple', label="Cross-validation score")
    ax.legend(loc="best")
    return ax

# Function to plot ROC curve
def plot_roc_curve(model, X, y, ax=None):
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, color='hotpink')
    ax.plot([0,1],[0,1], 'k--')
    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    return ax

# Model evaluation function
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model: {name}, Accuracy: {accuracy}, F1 Score: {f1}")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle(f"{name} - Accuracy: {accuracy}, F1 Score: {f1}")
    plot_learning_curve(model, f"Learning Curve - {name}", X_train, y_train, ax=axes[0])
    plot_roc_curve(model, X_test, y_test, ax=axes[1])
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted Labels")
    axes[2].set_ylabel("True Labels")
    plt.show()
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Initialize and evaluate models
models = [("RandomForest", RandomForestClassifier())]
for name, model in models:
    evaluate_model(model, X_train, y_train, X_test, y_test, name)
