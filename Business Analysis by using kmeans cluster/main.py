
# ============================================================
# BUSINESS ANALYTICS - CUSTOMER/ORDER CLUSTERING
# Algorithm: K-Means Clustering
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv("business_analytics_1000_rows.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 2. Remove irrelevant columns
# ------------------------------------------------------------

drop_cols = [
    "order_id",
    "customer_id",
    "order_date",
    "currency",
    "customer_satisfaction_1_to_5",
    "satisfaction_class"
]

# Drop only columns that actually exist
X = df.drop(
    columns=[col for col in drop_cols if col in df.columns]
).copy()

# Remove constant columns because they do not help distinguish groups
constant_cols = [
    col for col in X.columns if X[col].nunique(dropna=False) <= 1
]
X = X.drop(columns=constant_cols)

if X.shape[1] == 0:
    raise ValueError("No usable features remain after removing columns.")

# ------------------------------------------------------------
# 3. Identify numerical and categorical columns
# ------------------------------------------------------------

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ------------------------------------------------------------
# 4. Preprocess data
#    Numerical: fill missing values and scale
#    Categorical: fill missing values and one-hot encode
# ------------------------------------------------------------

transformers = []

if numeric_cols:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    transformers.append(("numeric", numeric_pipeline, numeric_cols))

if categorical_cols:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(
        ("categorical", categorical_pipeline, categorical_cols)
    )

preprocessor = ColumnTransformer(transformers=transformers)
X_processed = preprocessor.fit_transform(X)

# Convert sparse output to a dense array if needed
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

print("\nProcessed data shape:", X_processed.shape)

# ------------------------------------------------------------
# 5. Find a suitable number of clusters using the Elbow Method
# ------------------------------------------------------------

inertias = []
k_values = range(2, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_processed)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 6. Compare silhouette scores
# ------------------------------------------------------------

scores = {}

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_processed)

    score = silhouette_score(X_processed, labels)
    scores[k] = score
    print(f"K = {k}, Silhouette Score = {score:.4f}")

# Select K with the highest silhouette score
best_k = max(scores, key=scores.get)
print(f"\nSelected number of clusters: {best_k}")

# ------------------------------------------------------------
# 7. Train final K-Means model
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_processed)

# Add cluster labels to the original dataset
result = df.copy()
result["Cluster"] = cluster_labels

print("\nCluster sizes:")
print(result["Cluster"].value_counts().sort_index())

print("\nSilhouette Score:",
      round(silhouette_score(X_processed, cluster_labels), 4))

# ------------------------------------------------------------
# 8. Understand the clusters
# ------------------------------------------------------------

print("\nNumerical feature averages by cluster:")
if numeric_cols:
    print(result.groupby("Cluster")[numeric_cols].mean().round(2))

print("\nCategorical feature patterns by cluster:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(
        result.groupby("Cluster")[col]
        .agg(lambda values: values.mode().iloc[0]
             if not values.mode().empty else "Unknown")
    )

# ------------------------------------------------------------
# 9. Visualize clusters in two dimensions using PCA
# ------------------------------------------------------------

if X_processed.shape[1] >= 2:
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_processed)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.8
    )
    plt.title("K-Means Clustering (PCA Visualization)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig("clusters_visualization.png", dpi=300)
    plt.show()
else:
    print("\nPCA visualization skipped: fewer than 2 features.")

# ------------------------------------------------------------
# 10. Save results
# ------------------------------------------------------------

result.to_csv("business_clustering_results.csv", index=False)

print("\nClustering completed successfully!")
print("Saved: business_clustering_results.csv")
print("Saved: elbow_method.png")
print("Saved: clusters_visualization.png (when applicable)")