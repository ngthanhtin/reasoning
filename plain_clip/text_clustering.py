# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

dataset = 'nabirds' # nabirds
def save_dict_to_json(data_dict, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data_dict, json_file)

if dataset != 'inat21':
    description_path = f"../plain_clip/descriptors/{dataset}/additional_chatgpt_descriptors_{dataset}.json"
else:
    # description_path = f"../plain_clip/descriptors/inaturalist2021/425_additional_chatgpt_descriptors_inaturalist.json"
    description_path = f"../plain_clip/descriptors/inaturalist2021/chatgpt_descriptors_inaturalist.json"

f = open(description_path, 'r')
documents = json.load(f)
documents = {k: f"{k}, {v[-1][9:]}"  for k,v in documents.items()}
#
# full_dict_documents = {}
# for i, (k, v) in enumerate(documents.items()):
#     full_dict_documents[i+1] = [k]
# file_path = f'class_{dataset}_clusters.json'
# save_dict_to_json(full_dict_documents, file_path)
#
docs2classes = {v:k for k, v in documents.items()}
documents = [v for v in documents.values()]
# split a sentence into multiple sentences
# documents = {k: v.split('.') for k,v in documents.items()}
# documents = {k: [f'{k}, {s}' for s in v] for k,v in documents.items()}


# %%
# Step 1: Vectorize the documents
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Step 2: Determine the optimal number of clusters (K) using silhouette score
max_clusters = 50  # Set a reasonable maximum number of clusters to consider
best_score = -1
best_k = 5
for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For k={k}, silhouette score: {silhouette_avg:.4f}")
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = k

print(f"Best number of clusters: {best_k}")

# Step 3: Perform K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Step 4: Print the clusters and their documents
clusters = {}
for doc, label in zip(documents, cluster_labels):
    if label not in clusters:
        clusters[label] = [doc]
    else:
        clusters[label].append(doc)


class_clusters = []
for cluster_id, docs in clusters.items():
    print(f"Cluster {cluster_id + 1}:")
    # print("\n".join(docs))
    classes = []
    for doc in docs:
        classes.append(docs2classes[doc])
    class_clusters.append(classes)
    
print("Length of Clusters: ", len(clusters.items()))

# %%
index2clusters = {}
for i in range(len(class_clusters)):
    index2clusters[i+1] = class_clusters[i]

# %%
file_path = f'class_{dataset}_clusters_{len(clusters.items())}.json'
save_dict_to_json(index2clusters, file_path)

# %%
