import os
import glob
import random
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

def read_files(directory):
    """Read all text files from the directory and return as a list of strings."""
    file_paths = glob.glob(os.path.join(directory, '*.txt'))
    texts = [open(file_path, 'r', encoding='ISO-8859-1').read() for file_path in file_paths]
    return texts

def generate_error(eta_u, eta_o):
    """Generate a multiplicative random error based on a log-uniform distribution."""
    return math.exp(random.uniform(-math.log(eta_u), math.log(eta_o)))

def coverage(S, V, similarities, alpha):
    """Coverage function without error."""
    total_value = 0
    for i in V:
        sum_within_S = sum(similarities[i][j] for j in S)
        sum_overall = sum(similarities[i][k] for k in V)
        covered_value = min(sum_within_S, alpha * sum_overall)
        total_value += covered_value
    return total_value

def coverage_with_error(S, V, similarities, alpha, eta_u, eta_o):
    """Coverage function with added prediction error, applied element-wise."""
    total_value = 0
    for i in V:
        sum_within_S = 0
        for j in S:
            error = generate_error(eta_u, eta_o)
            sum_within_S += similarities[i][j] * error  # Apply error to each similarity measurement
        sum_overall = sum(similarities[i][k] for k in V)
        covered_value = min(sum_within_S, alpha * sum_overall)
        total_value += covered_value
    return total_value


def greedy_submodular_maximization(func, V, similarities, k, alpha, eta_u=None, eta_o=None):
    """Greedy algorithm for submodular maximization including error handling."""
    S = set()
    while len(S) < k:
        next_sentence = max(V - S, key=lambda x: func(S.union({x}), V, similarities, alpha, eta_u, eta_o) if eta_u and eta_o else func(S.union({x}), V, similarities, alpha))
        S.add(next_sentence)
    return list(S)

def evaluate_summaries(articles, summaries, func, k, alpha, eta_u=None, eta_o=None, repeats=30):
    vectorizer = TfidfVectorizer(stop_words='english')
    rouge = Rouge()
    all_f_measures = []

    for _ in range(repeats):
        for article, ref in zip(articles, summaries):
            sentences = article.split('.')
            X = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(X)
            V = set(range(len(sentences)))
            selected_indices = greedy_submodular_maximization(func, V, similarity_matrix, k, alpha, eta_u, eta_o)
            summary = ' '.join(sentences[i] for i in selected_indices)
            scores = rouge.get_scores(summary, ref)
            all_f_measures.append(scores[0]['rouge-1']['f'])

    median_f_measure = np.median(all_f_measures)
    iqr_f_measure = np.percentile(all_f_measures, 75) - np.percentile(all_f_measures, 25)
    return median_f_measure, iqr_f_measure

# Settings
articles_dir = 'C:\\Users\\1\\Desktop\\business'
summaries_dir = 'C:\\Users\\1\\Desktop\\business_summary'
eta_u = 10
eta_o = 10
alpha = 1.0
k = 5

articles = read_files(articles_dir)[:100]
summaries = read_files(summaries_dir)[:100]

# Evaluate without error
median_f_measure, iqr_f_measure = evaluate_summaries(articles, summaries, coverage, k, alpha)
print("Without Error - Median F-Measure:", median_f_measure, "IQR:", iqr_f_measure)

# Evaluate with error
median_f_measure_error, iqr_f_measure_error = evaluate_summaries(articles, summaries, coverage_with_error, k, alpha, eta_u, eta_o)
print("With Error - Median F-Measure:", median_f_measure_error, "IQR:", iqr_f_measure_error)