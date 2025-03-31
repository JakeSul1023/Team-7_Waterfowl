# Optimized Random Forest Code to Prevent Kill Errors

import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import seaborn as sns

# Load and reduce dataset
csv = "ShortTermSetData(Aug-Sept).csv"
dataset = pd.read_csv(csv, on_bad_lines='skip', engine='python')

# Sample only 5000 entries to reduce memory load
dataset = dataset.sample(n=5000, random_state=42).copy()

# Drop irrelevant columns
columns_to_drop = [
    "battery-charge-percent", "battery-charging-current", "gps-time-to-fix",
    "orn:transmission-protocol", "tag-voltage", "sensor-type", "visible",
    "individual-local-identifier"
]
dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], inplace=True)

# Convert timestamp to Unix timestamp
if 'timestamp' in dataset.columns:
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], errors='coerce')
    dataset['timestamp'] = dataset['timestamp'].astype('int64') // 10**9

# Identify types of columns
numeric_cols = [
    'event-id', 'location-long', 'location-lat', 'acceleration-raw-x', 'acceleration-raw-y',
    'acceleration-raw-z', 'bar:barometric-height', 'external-temperature', 'gps:hdop',
    'heading', 'gps:satellite-count', 'ground-speed', 'height-above-msl', 'gls:light-level',
    'mag:magnetic-field-raw-x', 'mag:magnetic-field-raw-y', 'mag:magnetic-field-raw-z',
    'tag-local-identifier'
]
categorical_cols = ['import-marked-outlier']
string_cols = ['individual-taxon-canonical-name', 'study-name']

# Encode columns
for col in dataset.columns:
    if col in string_cols:
        dataset[col] = dataset[col].astype(str)
    elif col in categorical_cols:
        dataset[col], _ = pd.factorize(dataset[col])
    elif col in numeric_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# One-hot encode string columns
def one_hot_encode_column(df, column_name):
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column_name]))
    return pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)

for col in string_cols:
    if col in dataset.columns:
        dataset = one_hot_encode_column(dataset, col)

# Prepare data
dataset = dataset.dropna(subset=['event-id'])

X = dataset.drop(columns='event-id')

y = dataset['event-id']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Convert to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Train Random Forest on 100 samples for visualization
temp_model = RandomForestClassifier(n_estimators=5, max_depth=3, max_features='sqrt', random_state=42)
temp_model.fit(X_train[:100], y_train[:100])

# Visualize the first tree
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(temp_model.estimators_[0], filled=True, rounded=True, fontsize=8)
plt.show()

# Evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # --- Insert ROC-AUC Section Here ---
    from sklearn.preprocessing import label_binarize

    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
    
            # Use all possible class labels from both train and test
            all_classes = np.unique(np.concatenate([y_train, y_test]))
            y_test_bin = label_binarize(y_test, classes=all_classes)
    
            if y_prob.shape[1] == y_test_bin.shape[1]:
                roc_auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
                print(f"ROC-AUC Score: {roc_auc:.4f}")
            else:
                print("Warning: ROC-AUC skipped due to shape mismatch.")
        else:
            print("ROC-AUC Score: Not supported (model has no predict_proba).")
    except Exception as e:
        print(f"Error computing ROC-AUC: {e}")

    # ----------------------------------

    # Classification Report (first 5 classes)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("\nClassification Report (first 5 classes):")
    for label in list(report_dict.keys())[:5]:
        if isinstance(report_dict[label], dict):
            print(f"{label}: precision={report_dict[label]['precision']:.2f}, "
                  f"recall={report_dict[label]['recall']:.2f}, "
                  f"f1-score={report_dict[label]['f1-score']:.2f}")

    report_df = pd.DataFrame(report_dict).transpose()
    print("\nFull Classification Report (first 5 rows):")
    print(report_df.head())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Optional memory cleanup
    import gc
    del model
    gc.collect()



# Full training with simplified Random Forest
model = RandomForestClassifier(n_estimators=50, max_depth=10, max_features='sqrt', random_state=42)
evaluate_model(model, X_train, y_train, X_test, y_test, "Optimized RF")
