import numpy as np, pandas as pd, matplotlib.pyplot as plt
import math,random,warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import (SelectKBest, chi2, f_classif, RFE, mutual_info_classif, SelectFromModel, SelectPercentile, GenericUnivariateSelect, RFECV)

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess data
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('breast-cancer.csv')  # Update with the correct path
scaler = StandardScaler()

# Preparing the data
X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].map({'M': 1, 'B': 0})  # Assuming 'M' and 'B' are the diagnosis codes

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_on_selected_features(X_train, y_train, X_test, y_test, feature_indices):
    #return True performance on test dataset
    feature_indices = [feature_indices] if isinstance(feature_indices, int) else feature_indices
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train.iloc[:, feature_indices], y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test.iloc[:, feature_indices]))
    return accuracy

def generate_error(lbd,eta):
  rnd=random.uniform(-math.log(lbd), math.log(eta))
  error=math.e**rnd
  return error

def result_with_new_feature(X_train, y_train, X_test, y_test, current_features, new_feature):
    extended_feature_set = current_features + [new_feature] if current_features else [new_feature]
    extended_accuracy = train_on_selected_features(X_train, y_train, X_test, y_test, extended_feature_set)
    return extended_accuracy

def oracle_simulation(X_train, y_train, X_test, y_test, K, lambda_val, eta_val):
  selected_features = []
  available_features = set(range(X_train.shape[1]))  # Start with all features as available
  Accuracy_Results=[]
  while len(selected_features) < K:
    # Recalculate gains for all remaining available features using list comprehension
    base_accuracy = train_on_selected_features(X_train, y_train, X_test, y_test, selected_features) if selected_features else 0
    feature_performance=[] #[feature,etd_acc,actual_acc,predict_acc]
    for feature in available_features:
      extend_accuracy=result_with_new_feature(X_train, y_train, X_test, y_test, selected_features, feature)
      actual_gain=extend_accuracy-base_accuracy
      predict_gain=actual_gain*generate_error(lambda_val,eta_val)
      feature_performance.append((feature,extend_accuracy,actual_gain,predict_gain))
    #sort features based on predict_gain
    feature_performance.sort(reverse=True, key=lambda x: x[3])
    selected_feature=feature_performance[0][0]
    selected_features.append(selected_feature)
    available_features.remove(selected_feature)  # Remove the selected feature to avoid reselection
    print("new feature selected (idx,exp_result,actual_gain,predict_gain):", feature_performance[0])
    new_accuracy=train_on_selected_features(X_train, y_train, X_test, y_test, selected_features) if selected_features else 0
    print("performance with ",len(selected_features)," features: ", new_accuracy)
    Accuracy_Results.append(new_accuracy)
  return Accuracy_Results


def plot_oracle_performance_with_error_bars(X_train, y_train, X_test, y_test, max_K):
    eta_pairs = [(1.5, 1.5), (4, 4), (10, 10), (20, 20)]
    colors = ['#1C6AB1', '#D2AA3A', '#47AF79', '#6A5DC4']  # Colors for each eta pair
    line_styles = ['*-', 'o-', '^-', 's-']  # Line styles for each pair
    plt.figure(figsize=(10, 8))
    repeating = 30  # Number of simulations per eta value
    offset = 0.05

    for idx, (eta_u, eta_o) in enumerate(eta_pairs):
        all_accuracies = np.zeros((repeating, max_K))  # Array to store results from all runs

        for run in range(repeating):
            accuracies = oracle_simulation(X_train, y_train, X_test, y_test, max_K, eta_u, eta_o)
            all_accuracies[run, :] = accuracies  # Store accuracies for each K

        # Calculate median and IQR (Interquartile Range)
        median_accuracies = np.median(all_accuracies, axis=0)
        iqr_low = np.percentile(all_accuracies, 25, axis=0)  # 25th percentile
        iqr_high = np.percentile(all_accuracies, 75, axis=0)  # 75th percentile

        # Error as the range between the 25th and 75th percentiles
        lower_errors = median_accuracies - iqr_low
        upper_errors = iqr_high - median_accuracies
        errors = [lower_errors, upper_errors]  # As upper and lower deviations

        # Plot with error bars using IQR for error representation
        plt.errorbar(np.array(range(1, max_K + 1)) + idx * offset, median_accuracies, yerr=errors, fmt=line_styles[idx],
                     color=colors[idx],
                     label=f'ηᵤ=ηₒ={eta_u}', capsize=4, linewidth=3, capthick=2, markersize=5)

    plt.xlabel('Number of Features (K)', fontsize=26)
    plt.ylabel('Accuracy', fontsize=26)
    plt.ylim([0.5, 1.0])
    plt.xticks(range(1, max_K + 1), fontsize=26)
    plt.yticks(np.linspace(0.5, 1.0, 6), fontsize=26)
    plt.legend(fontsize=22, loc='lower right')
    plt.grid(True)
    plt.savefig(r'C:\Users\1\Desktop\eta_cancer_performance2.pdf', format='pdf', bbox_inches='tight')
    plt.show()

plot_oracle_performance_with_error_bars(X_train, y_train, X_test, y_test, 7)
