import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Online_Retail.csv", encoding="ISO-8859-1")

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())
print(df.duplicated().sum())

print(df["Quantity"].mean())
print(df["Country"].value_counts().head(10))

# Remove rows without customer IDs
df = df.dropna(subset=["CustomerID"])


# Remove duplicate records
df = df.drop_duplicates()


# Remove cancelled invoices
df = df[
    ~df["InvoiceNo"].astype(str).str.startswith("C")
]


# Keep positive quantities and prices
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
]

print("Shape after cleaning:", df.shape)



# Convert invoice date to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Calculate spending for each transaction line
df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

# Set a reference date
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Group transactions by customer
customer_data = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalSpend", "sum"),
    TotalQuantity=("Quantity", "sum")
)

print(customer_data.head())
print(customer_data.shape)


features = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity"
]

X = customer_data[features]

print(X.head())
print("Feature shape:", X.shape)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaled shape:", X_scaled.shape)



model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = model.fit_predict(X_scaled)

print(customer_data.head())
print(customer_data["Cluster"].value_counts().sort_index())


score = silhouette_score(
    X_scaled,
    customer_data["Cluster"]
)

print("Silhouette Score:", score)