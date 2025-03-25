#import dependencies and other ML libraries
import tensorflow as tf
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Commonly used supporting libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

csv="ShortTermSetData(Aug-Sept).csv"

#Read data
#dataset=pd.read_csv(csv)
dataset = pd.read_csv(csv, on_bad_lines='skip', engine='python')
dataset.head()
dataset.shape

#Showing descriptive statistics
dataset.describe()
dataset.dtypes
#Checking for any null values
dataset.isnull().sum()
#Checking out data distribution
for x in dataset.columns:
  print(f"Column: {x}")
  print("-"*20)
  print(dataset[x].value_counts())
  print("")
# Drop irrelevant columns and verify, only if they exist
columns_to_drop = ["battery-charge-percent", "battery-charging-current", "gps-time-to-fix",
                   "orn:transmission-protocol", "tag-voltage", "sensor-type", "visible", "individual-local-identifier"]

for column in columns_to_drop:
    if column in dataset.columns:  # Check if column exists before dropping
        dataset = dataset.drop(columns=[column])
    # else:
    #     print(f"Column '{column}' not found in the DataFrame.") Print a message if the column is not found

dataset.columns #verification


# Convert timestamp column to datetime and then to Unix timestamp (seconds since epoch)
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], errors='coerce')
dataset['timestamp'] = dataset['timestamp'].astype('int64') // 10**9  # Converts to seconds

# Define column types
numeric_cols = ['event-id', 'location-long', 'location-lat', 'acceleration-raw-x', 'acceleration-raw-y', 'acceleration-raw-z', 
                'bar:barometric-height', 'external-temperature', 'gps:hdop', 'heading', 'gps:satellite-count', 'ground-speed', 'height-above-msl', 'gls:light-level', 
                'mag:magnetic-field-raw-x', 'mag:magnetic-field-raw-y', 'mag:magnetic-field-raw-z', 'tag-local-identifier']

categorical_cols = ['import-marked-outlier']  # Add other categorical columns
string_cols = ['individual-taxon-canonical-name', 'study-name']

# Apply transformations
for col in dataset.columns:
    if col in string_cols:
        dataset[col] = dataset[col].astype(str)  # Preserve as string
    elif col in categorical_cols:
        dataset[col], _ = pd.factorize(dataset[col])  # Encode categorical
    elif col in numeric_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')  # Ensure numeric values stay numeric
    else:
        dataset[col] = dataset[col]  # Default behavior

# One-Hot Encoding for Species Column
one_hot_encoder = OneHotEncoder(sparse_output=False)
species_encoded = one_hot_encoder.fit_transform(dataset[['individual-taxon-canonical-name']])

# Convert to DataFrame with meaningful column names
species_encoded_df = pd.DataFrame(species_encoded, columns=one_hot_encoder.get_feature_names_out())

# Concatenate with dataset and drop original species column
dataset = pd.concat([dataset, species_encoded_df], axis=1).drop(columns=['individual-taxon-canonical-name'])

# One-Hot Encoding for the 'study-name' Column
one_hot_encoder = OneHotEncoder(sparse_output=False)
study_name_encoded = one_hot_encoder.fit_transform(dataset[['study-name']])

# Convert to DataFrame with meaningful column names
study_name_encoded_df = pd.DataFrame(study_name_encoded, columns=one_hot_encoder.get_feature_names_out(['study-name']))

# Concatenate with the original dataset and drop the original 'study-name' column
dataset = pd.concat([dataset, study_name_encoded_df], axis=1).drop(columns=['study-name'])

# Check results
pd.options.display.max_columns = 50
dataset.head()

#Algorithm to start splitting between training and test sets

# Labels
y = dataset['event-id']
# Features
X = dataset.drop(columns = 'event-id')

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Split training data into training and validation
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, random_state=1)


# Convert to numpy arrays
X_train = X_train.to_numpy()
X_validate = X_validate.to_numpy()
X_test = X_test.to_numpy()


y_train = y_train.to_numpy()
y_validate = y_validate.to_numpy()
y_test = y_test.to_numpy()

#Checking shape of the new datasets

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of X_validate: {X_validate.shape}")

print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of y_validate: {y_validate.shape}")

train_class_counts = {label: (y_train == label).sum() for label in np.unique(y_train)}
test_class_counts = {label: (y_test == label).sum() for label in np.unique(y_test)}

print(f"Distribution of y_train: {train_class_counts}")
print(f"Distribution of y_test: {test_class_counts}")

def visualize_first_tree(model):
    from sklearn.tree import plot_tree
    plt.figure(figsize=(20,10))
    plot_tree(model.estimators_[0], filled=True, feature_names=X_train.columns, class_names=True, rounded=True, fontsize=8)
    plt.show()


def plot_learning_curve(model, title, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=2, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="purple")

    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color='purple', label="Cross-validation score")

    ax.legend(loc="best")
    
    plt.show()  
    return fig  # Return figure if needed for further manipulation

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def plot_roc_curve(model, X, y, ax=None):
    """
    Plots the ROC curve for a trained model. Works for both binary and multiclass classification.
    """
    # Check if the model supports probability predictions
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)  # Get class probabilities
    else:
        raise ValueError("Model does not support probability outputs.")

    # Binarize y labels for multiclass ROC handling
    y_bin = label_binarize(y, classes=np.unique(y))
    
    # Compute ROC AUC score for multiclass
    roc_auc = roc_auc_score(y_bin, y_pred_proba, average='macro', multi_class='ovr')

    # Plot the ROC Curve
    if ax is None:
        plt.figure()
        ax = plt.gca()
    
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.plot([0,1], [0,1], 'k--')  # Diagonal reference line
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='hotpink')


    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    return ax


# Loop over each model in an array of models
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    """
    Trains and evaluates a model using accuracy, F1-score, ROC curve, learning curve, and confusion matrix.

    Parameters:
    model -- The machine learning model to evaluate.
    X_train, y_train -- Training dataset.
    X_test, y_test -- Test dataset.
    name -- Name of the model for labeling.

    Returns:
    None
    """
    
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate Accuracy and F1 Score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print Results
    print(f"Model: {name}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Create Subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Learning Curve (Check if function exists)
    if 'plotLearningCurve' in globals():
        plotLearningCurve(model, f"Learning Curve - {name}", X_train, y_train, ax=axes[0])
    else:
        axes[0].set_title("Learning Curve Not Available")
        axes[0].axis("off")

    # ROC Curve (Handle unsupported models)
    try:
        plot_roc_curve(model, X_test, y_test, ax=axes[1])
    except ValueError as e:
        axes[1].set_title("ROC Curve Not Available")
        axes[1].axis("off")
        print(f"Warning: Could not plot ROC curve for {name}. Error: {e}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted Labels")
    axes[2].set_ylabel("True Labels")

    plt.show()
    print("\n")

    # Print Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Initializes an array called models.
models = []
models.append(('FOREST', RandomForestClassifier()))

# Testing Train and evaluate using only the first 100 samples
for name, model in models:
    print(f"Training {name} with first 100 entries...")
    model.fit(X_train[:100], y_train[:100])  # Train only on first 100 samples
    evaluate_model(model, X_train[:100], y_train[:100], X_test, y_test, name)

# Visualize the first tree
visualize_first_tree(models[0][1])  # Assuming the first model in the list is the RandomForestClassifier

# Train and evaluate using the full dataset
for name, model in models:
    print(f"Training {name} with the full dataset...")
    model.fit(X_train, y_train)  # Train on the entire dataset
    evaluate_model(model, X_train, y_train, X_test, y_test, name)

visualize_first_tree(models[0][1])  # Assuming the first model in the list is the RandomForestClassifier

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],  # Number of trees
    'max_depth': [None, 10, 20, 30, 40, 50],     # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],              # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],               # Minimum samples at a leaf node
    'max_features': ['sqrt', 'log2', None],       # Number of features to consider for splits
    'bootstrap': [True, False]                    # Whether to use bootstrap samples
}


# Initialize the base model
rf = RandomForestClassifier(random_state=42)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter settings to sample
    cv=3,       # Number of cross-validation folds
    scoring='accuracy',  # Metric to optimize (can also use 'f1', 'roc_auc', etc.)
    n_jobs=-1,  # Use all available CPU cores
    random_state=42,
    verbose=2
)

# Fit the model
random_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)

# Train the final model with the best parameters
best_rf = random_search.best_estimator_

# Evaluate the tuned model
evaluate_model(best_rf, X_train, y_train, X_test, y_test, "Tuned Random Forest")

# Initial model
initial_rf = RandomForestClassifier(random_state=42)
#evaluate_model(initial_rf, X_train, y_train, X_test, y_test, "Initial Random Forest")

# Tuned model
#evaluate_model(best_rf, X_train, y_train, X_test, y_test, "Tuned Random Forest")

import joblib

# Save the tuned model
joblib.dump(best_rf, 'tuned_random_forest_duck_flight_path.pkl')