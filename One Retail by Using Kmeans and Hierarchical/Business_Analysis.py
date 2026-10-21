import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("dataset for star cluster simulstions/c_0000.csv")

x = df[["x", "y", "z"]].copy()

x= x.fillna(x.median())

scaler= StandardScaler()
x_scaled= scaler.fit_transform(x)


KMeans = KMeans(
    n_clusters=4,
    random_state=42, 
    n_init=10
)

df["cluster"] = KMeans.fit_predict(x_scaled)

print("\nObjects in each cluster: ")
print(df["cluster"].value_counts().sort_index())

score=silhouette_score(
    x_scaled,
    df["cluster"],
    sample_size=5000,
    random_state=42
)


print("\nSilhouette Score:", round(score, 4))


sample = df.sample(n=min(5000, len(df)), random_state=42)
fig=plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")


scatter = ax.scatter(
    sample["x"],
    sample["y"],
    sample["z"],
    c=sample["cluster"],
    cmap="viridis",
    s=5
)

ax.set_xlabel("X Position")
ax.set_xlabel("Y Position")
ax.set_xlabel("Z Position")
ax.set_title("K-Means Clustering of Objects")

plt.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.show()

df.to_csv("clustered_results.csv", index=False)

print("\nClustering completed!")
print("Results saved to clustered_results.csv")