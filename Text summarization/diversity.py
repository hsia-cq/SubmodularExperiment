import os
import glob
import random
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rouge import Rouge

# Setting the number of cores for joblib to use
os.environ['LOKY_MAX_CPU_COUNT'] = '2'

def read_files(directory):
    """Read all files from the directory and return as a list of strings."""
    file_paths = glob.glob(os.path.join(directory, '*.txt'))
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            texts.append(file.read())
    return texts

def prepare_clusters(similarities, num_clusters):
    """Prepare clusters using KMeans based on the similarities matrix."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(similarities)
    clusters = {i: set(np.where(kmeans.labels_ == i)[0]) for i in range(num_clusters)}
    return clusters

def generate_error(eta_u, eta_o):
    """Generate random error for predictive submodular function."""
    rnd = random.uniform(-math.log(eta_u), math.log(eta_o))
    error = math.exp(rnd)
    return error

def diversity(S, V, similarities, clusters):
    """Calculate the diversity score based on the theoretical model."""
    N = len(V)
    diversity_score = 0
    for cluster in clusters.values():
        cluster_intersection = cluster.intersection(S)
        if cluster_intersection:
            cluster_similarity = sum(similarities[i][j] for i in cluster for j in cluster_intersection) / N
            diversity_score += math.sqrt(cluster_similarity)
    return diversity_score

def diversity_with_error(S, V, similarities, clusters, eta_u, eta_o):
    """Calculate the diversity score with added prediction error, applying error per similarity interaction."""
    N = len(V)
    diversity_score = 0
    for cluster in clusters.values():
        cluster_intersection = cluster.intersection(S)
        if cluster_intersection:
            cluster_similarity = 0
            for i in cluster:
                for j in cluster_intersection:
                    sim = similarities[i][j] * generate_error(eta_u, eta_o)  # Apply error directly to each similarity
                    cluster_similarity += sim
            diversity_score += math.sqrt(cluster_similarity / N)
    return diversity_score

def greedy_submodular_maximization(func, V, similarities, clusters, k, eta_u=None, eta_o=None):
    """Greedy algorithm for submodular maximization including error handling."""
    S = set()
    while len(S) < k:
        next_sentence = max(V - S, key=lambda x: func(S.union({x}), V, similarities, clusters, eta_u, eta_o) if eta_u and eta_o else func(S.union({x}), V, similarities, clusters))
        S.add(next_sentence)
    return list(S)

def evaluate_summaries(articles, summaries, func, k, num_clusters, eta_u=None, eta_o=None, repeats=30):
    vectorizer = TfidfVectorizer(stop_words='english')
    rouge = Rouge()
    all_f_measures = []

    texts = [' '.join(article.split('.')) for article in articles]  # Flatten articles
    X = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(X)
    clusters = prepare_clusters(similarities, num_clusters)  # Prepare clusters

    for _ in range(repeats):
        for article, ref in zip(articles, summaries):
            V = set(range(len(article.split('.'))))
            selected_indices = greedy_submodular_maximization(func, V, similarities, clusters, k, eta_u, eta_o)
            summary = ' '.join(article.split('.')[i] for i in selected_indices)
            scores = rouge.get_scores(summary, ref)
            all_f_measures.append(scores[0]['rouge-1']['f'])

    median_f_measure = np.median(all_f_measures)
    iqr_f_measure = np.percentile(all_f_measures, 75) - np.percentile(all_f_measures, 25)

    return median_f_measure, iqr_f_measure

# Setup and execution
articles_dir = 'C:\\Users\\1\\Desktop\\business'
summaries_dir = 'C:\\Users\\1\\Desktop\\business_summary'
eta_u = 10
eta_o = 10
k = 5
num_clusters = 2 # Number of clusters for diversity

articles = read_files(articles_dir)[:100]
summaries = read_files(summaries_dir)[:100]

# Evaluate without error
median_f_measure, iqr_f_measure = evaluate_summaries(articles, summaries, diversity, k, num_clusters)
print("Without Error - Median F-Measure:", median_f_measure, "IQR:", iqr_f_measure)

# Evaluate with error
median_f_measure_error, iqr_f_measure_error = evaluate_summaries(articles, summaries, diversity_with_error, k, num_clusters, eta_u, eta_o)
print("With Error - Median F-Measure:", median_f_measure_error, "IQR:", iqr_f_measure_error)
