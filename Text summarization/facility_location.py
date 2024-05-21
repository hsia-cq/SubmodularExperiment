import os
import glob
import random
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

# Helper functions
def read_files(directory):
    """Read all files from the directory and return as a list of strings."""
    file_paths = glob.glob(os.path.join(directory, '*.txt'))
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            texts.append(file.read())
    return texts

def generate_error(eta_u, eta_o):
    """Generate random error for predictive submodular function."""
    rnd = random.uniform(-math.log(eta_u), math.log(eta_o))
    error = math.e ** rnd
    return error

# Submodular Objective Functions
def facility_location(S, V, similarities):
    """Facility location function."""
    return sum(max(similarities[i][j] for j in S) for i in V)

def facility_location_with_error(S, V, similarities, eta_u, eta_o):
    """Facility location function with prediction error."""
    total_value = 0
    for i in V:
        max_val = max(similarities[i][j] for j in S)
        error = generate_error(eta_u, eta_o)
        total_value += max_val * error
    return total_value

def greedy_submodular_maximization(func, V, similarities, k, eta_u=None, eta_o=None):
    """Generalized greedy submodular maximization to include error generation."""
    S = set()
    while len(S) < k:
        if eta_u is not None and eta_o is not None:
            next_sentence = max(V - S, key=lambda x: func(S.union({x}), V, similarities, eta_u, eta_o))
        else:
            next_sentence = max(V - S, key=lambda x: func(S.union({x}), V, similarities))
        S.add(next_sentence)
    return list(S)

def evaluate_summaries(articles, summaries, func, k, eta_u=None, eta_o=None, repeats=30):
    """Evaluate summaries and calculate comprehensive median and IQR ROUGE-1 scores."""
    vectorizer = TfidfVectorizer(stop_words='english')
    rouge = Rouge()
    all_scores = []
    for _ in range(repeats):
        for article, ref in zip(articles, summaries):
            sentences = article.split('.')
            X = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(X)
            V = set(range(len(sentences)))
            selected_indices = greedy_submodular_maximization(func, V, similarity_matrix, k, eta_u, eta_o)
            summary = ' '.join(sentences[i] for i in selected_indices)
            score = rouge.get_scores(summary, ref)[0]['rouge-1']['f']
            all_scores.append(score)
    median_score = np.median(all_scores)
    iqr_score = np.percentile(all_scores, 75) - np.percentile(all_scores, 25)
    return median_score, iqr_score

# Directories and settings
articles_dir = 'C:\\Users\\1\\Desktop\\business'
summaries_dir = 'C:\\Users\\1\\Desktop\\business_summary'
eta_u = 10
eta_o = 10
k = 5

# Read articles and summaries
articles = read_files(articles_dir)[:100]
summaries = read_files(summaries_dir)[:100]

facility_median, facility_iqr = evaluate_summaries(articles, summaries, facility_location, k)
facility_error_median, facility_error_iqr = evaluate_summaries(articles, summaries, facility_location_with_error, k, eta_u, eta_o)

# Print results
print("Facility Location Median ROUGE-1 Score:", facility_median)
print("Facility Location IQR ROUGE-1 Score:", facility_iqr)
print("Facility Location with Error Median ROUGE-1 Score:", facility_error_median)
print("Facility Location with Error IQR ROUGE-1 Score:", facility_error_iqr)
