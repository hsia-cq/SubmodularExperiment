import itertools

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import math,random,warnings
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import (SelectKBest, chi2, f_classif, RFE, mutual_info_classif, SelectFromModel, SelectPercentile, GenericUnivariateSelect, RFECV)
from itertools import combinations

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
np.random.seed(42)

# Data loading and preprocessing
df = pd.read_csv('airline.csv')
df.drop(["Unnamed: 0", "id"], axis=1, inplace=True)
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].mean())
df = df.sample(n=1000, random_state=42)
le = LabelEncoder()
cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"]
for label in cat_cols:
    df[label] = le.fit_transform(df[label])
# Drop outliers
outliers_distance = df[df['Flight Distance'] > 3736.5]
df.drop(outliers_distance.index, inplace=True)
outliers_departure = df[df['Departure Delay in Minutes'] > 800]
df.drop(outliers_departure.index, inplace = True)
outlier_arrival = df[df['Arrival Delay in Minutes'] > 650]
df.drop(outlier_arrival.index, inplace = True)

scaler = StandardScaler()
for column in df.columns:
    if df[column].dtype == type(float) or df[column].dtype == type(int):
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))

X = df.drop(columns=['satisfaction'])
y = df['satisfaction']
# Assuming X, y are defined and preprocessed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_on_selected_features(X_train, y_train, X_test, y_test, feature_indices):
    # Return true performance on the test dataset
    feature_indices = [feature_indices] if isinstance(feature_indices, int) else feature_indices
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train.iloc[:, feature_indices], y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test.iloc[:, feature_indices]))
    return accuracy


def generate_error(eta_u = 1.5, eta_o=1.5):
    rnd = random.uniform(-math.log(eta_u), math.log(eta_o))
    error = math.e ** rnd
    return error

def r_step_greedy(X_train, y_train, X_test, y_test, max_K, R=2, eta_u=1.5, eta_o=1.5):
    selected_features = []
    available_features = list(range(X_train.shape[1]))
    accuracy_results = []

    while len(selected_features) < max_K:
        if len(selected_features) + R > max_K:
            R = max_K - len(selected_features)

        combinations = list(itertools.combinations([f for f in available_features if f not in selected_features], R))
        best_gain = -float('inf')
        best_combination = None
        base_accuracy = train_on_selected_features(X_train, y_train, X_test, y_test, selected_features) if selected_features else 0

        for combination in combinations:
            extended_features = selected_features + list(combination)
            extend_accuracy = train_on_selected_features(X_train, y_train, X_test, y_test, extended_features)
            gain = extend_accuracy - base_accuracy
            predicted_gain = gain * generate_error(eta_u, eta_o)

            if predicted_gain > best_gain:
                best_gain = predicted_gain
                best_combination = combination

        selected_features.extend(best_combination)
        available_features = [f for f in available_features if f not in best_combination]
        new_accuracy = train_on_selected_features(X_train, y_train, X_test, y_test, selected_features)
        accuracy_results.append(new_accuracy)
        print(f"R={R}, Selected features: {selected_features}, New Accuracy: {new_accuracy}")

    return selected_features, accuracy_results

def evaluate_and_print_results(X_train, y_train, X_test, y_test, max_K, R_values):
    for R in R_values:
        all_accuracies = []
        print(f"Testing R={R}")
        for _ in range(30):
            _, accuracies = r_step_greedy(X_train, y_train, X_test, y_test, max_K, R)
            all_accuracies.append(accuracies)

        accuracies_per_feature_count = list(map(list, zip(*all_accuracies)))
        medians = [np.median(acc) for acc in accuracies_per_feature_count]
        iqrs = [np.percentile(acc, 75) - np.percentile(acc, 25) for acc in accuracies_per_feature_count]

        print(f"Accuracies for R={R} (medians): {medians}")
        print(f"Accuracies for R={R} (IQRs): {iqrs}\n")

# Example usage
evaluate_and_print_results(X_train, y_train, X_test, y_test, 7, [1, 2, 3])