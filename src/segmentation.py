from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def segment_customers(rfm):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)

    n_samples = len(rfm)

    if n_samples < 3:
        rfm['Cluster'] = 0
        return rfm, None, 0

    best_k = 2
    best_score = -1
    max_k = min(6, n_samples - 1)

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(scaled)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(scaled, labels)

        if score > best_score:
            best_k = k
            best_score = score

    final_model = KMeans(n_clusters=best_k, random_state=42)
    rfm['Cluster'] = final_model.fit_predict(scaled)

    return rfm, final_model, best_score