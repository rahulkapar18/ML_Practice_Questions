"""Market basket analysis on the Groceries_dataset.csv file.
Includes Apriori and a simple ECLAT implementation.
"""

from pathlib import Path
from collections import defaultdict
from itertools import combinations
from math import ceil
import time

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


# ------------------------------------------------------------
# 1. Load and inspect the dataset
# ------------------------------------------------------------
DATA_PATH = Path(
    r"C:\Users\parme\Downloads\ML Practice\Groceries by Using Apriori\Groceries_dataset.csv"
)
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.10

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATA_PATH}\n"
        "Check the file name and folder path in DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)
print("\n=== DATASET OVERVIEW ===")
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("Unique customers:", df["Member_number"].nunique())
print("Unique products:", df["itemDescription"].nunique())
print("Duplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# 2. Prepare dates and transaction IDs
# ------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="raise")

# This dataset does not provide a separate transaction ID.
# This uses customer + date as a practical approximation.
# If a customer shops multiple times on the same date, those visits
# will be combined into one basket.
df["TransactionID"] = (
    df["Member_number"].astype(str)
    + "_"
    + df["Date"].dt.strftime("%Y-%m-%d")
)

print("\n=== TRANSACTION IDs ===")
print(df[["TransactionID", "itemDescription"]].head(10))
total_transactions = df["TransactionID"].nunique()
print("Total transactions:", total_transactions)

# ------------------------------------------------------------
# 3. Build transaction-item basket
# ------------------------------------------------------------
basket = df.pivot_table(
    index="TransactionID",
    columns="itemDescription",
    aggfunc="size",
    fill_value=0,
)

# Convert counts to 0/1: each product is either absent or present.
basket = (basket > 0).astype(int)
print("\n=== BASKET ===")
print(basket.head())
print("Basket shape:", basket.shape)

# ------------------------------------------------------------
# 4. Apriori: frequent itemsets
# ------------------------------------------------------------
print("\n=== APRIORI: FREQUENT ITEMSETS ===")
apriori_start = time.perf_counter()
frequent_itemsets = apriori(
    basket.astype(bool),
    min_support=MIN_SUPPORT,
    use_colnames=True,
)
apriori_time = time.perf_counter() - apriori_start

frequent_itemsets["support_percentage"] = frequent_itemsets["support"] * 100
frequent_itemsets["itemset_length"] = frequent_itemsets["itemsets"].apply(len)
print(frequent_itemsets[["support", "support_percentage", "itemsets"]].head(10))
print("Total frequent itemsets:", len(frequent_itemsets))
print("Apriori execution time:", round(apriori_time, 4), "seconds")

multiple_items = frequent_itemsets[frequent_itemsets["itemset_length"] >= 2].copy()
print("\nItemsets with 2 or more products:")
print(multiple_items)
print("Total itemsets with 2 or more products:", len(multiple_items))

# ------------------------------------------------------------
# 5. Apriori: association rules
# ------------------------------------------------------------
print("\n=== APRIORI: ASSOCIATION RULES ===")
# mlxtend association_rules expects the original support and itemsets
# columns. Extra display columns are removed before calling it.
itemsets_for_rules = frequent_itemsets[["support", "itemsets"]].copy()

if (frequent_itemsets["itemset_length"] >= 2).any():
    rules = association_rules(
        itemsets_for_rules,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )
else:
    # Keep an empty rules DataFrame with the columns used below.
    rules = pd.DataFrame(
        columns=["antecedents", "consequents", "support", "confidence", "lift"]
    )

if not rules.empty:
    rules_sorted = rules.sort_values(by="lift", ascending=False).copy()
    print("Rules sorted by lift:")
    print(rules_sorted[["antecedents", "consequents", "support", "confidence", "lift"]])

    confidence_sorted = rules.sort_values(by="confidence", ascending=False)
    print("\nTop 5 rules by confidence:")
    print(confidence_sorted[["antecedents", "consequents", "support", "confidence", "lift"]].head(5))

    strong_rules = rules_sorted[rules_sorted["lift"] > 1]
    print("\nRules with lift > 1:")
    print(strong_rules[["antecedents", "consequents", "support", "confidence", "lift"]])
    print("Total rules with lift > 1:", len(strong_rules))

    rules["support_percentage"] = rules["support"] * 100
    rules["confidence_percentage"] = rules["confidence"] * 100
    rules_percentage = rules.sort_values(by="confidence", ascending=False)
    print("\nRules with percentages:")
    percentage_display = rules_percentage[
        ["antecedents", "consequents", "support_percentage", "confidence_percentage", "lift"]
    ].head(10).rename(
        columns={
            "antecedents": "IF product",
            "consequents": "THEN product",
            "support_percentage": "Support (%)",
            "confidence_percentage": "Confidence (%)",
            "lift": "Lift",
        }
    )
    print(percentage_display.to_string(index=False, col_space=14, float_format=lambda value: f"{value:.4f}"))

    # Plot top rules by confidence.
    top_rules = rules.sort_values(by="confidence", ascending=False).head(10).copy()
    top_rules["rule"] = top_rules.apply(
        lambda row: ", ".join(sorted(row["antecedents"]))
        + " -> "
        + ", ".join(sorted(row["consequents"])),
        axis=1,
    )
    top_rules["confidence_percentage"] = top_rules["confidence"] * 100
    plt.figure(figsize=(12, 6))
    plt.barh(top_rules["rule"], top_rules["confidence_percentage"])
    plt.xlabel("Confidence (%)")
    plt.ylabel("Association Rule")
    plt.title("Top Association Rules by Confidence")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    rules.to_csv("apriori_association_rules.csv", index=False)
    print("Association rules saved to apriori_association_rules.csv")
else:
    print(
        "No association rules were generated. Try lowering MIN_SUPPORT or MIN_CONFIDENCE."
    )

# ------------------------------------------------------------
# 6. Most frequently purchased products
# ------------------------------------------------------------
print("\n=== TOP 10 PRODUCTS ===")
product_counts = basket.sum().sort_values(ascending=False)
print(product_counts.head(10))
product_counts.head(10).sort_values().plot(kind="barh", figsize=(10, 6))
plt.xlabel("Number of Transactions")
plt.ylabel("Product")
plt.title("Top 10 Most Frequently Purchased Products")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. ECLAT: vertical transaction sets
# ------------------------------------------------------------
print("\n=== ECLAT ===")
item_transactions = defaultdict(set)
for transaction_id, product in zip(df["TransactionID"], df["itemDescription"]):
    item_transactions[product].add(transaction_id)

print("Example product transaction counts:")
for product, transactions in list(item_transactions.items())[:5]:
    print(product, len(transactions))

min_count = ceil(MIN_SUPPORT * total_transactions)
frequent_items = {
    product: transactions
    for product, transactions in item_transactions.items()
    if len(transactions) / total_transactions >= MIN_SUPPORT
}
print("Total transactions:", total_transactions)
print("Minimum transaction count required:", min_count)
print("Frequent single products:", len(frequent_items))

# ECLAT frequent pairs
frequent_pairs = []
for product1, product2 in combinations(frequent_items.keys(), 2):
    common_transactions = frequent_items[product1] & frequent_items[product2]
    support = len(common_transactions) / total_transactions
    if support >= MIN_SUPPORT:
        frequent_pairs.append(
            {
                "itemset": (product1, product2),
                "support": support,
                "support_percentage": support * 100,
            }
        )

# IMPORTANT FIX: declare columns so an empty result still has a support column.
frequent_pairs_df = pd.DataFrame(
    frequent_pairs,
    columns=["itemset", "support", "support_percentage"],
)
if not frequent_pairs_df.empty:
    frequent_pairs_df = frequent_pairs_df.sort_values(by="support", ascending=False)
print("\nFrequent ECLAT pairs:")
print(frequent_pairs_df.head(10).to_string(index=False))
print("Total frequent pairs:", len(frequent_pairs_df))

# ECLAT frequent triples
frequent_triples = []
for product1, product2, product3 in combinations(frequent_items.keys(), 3):
    common_transactions = (
        frequent_items[product1]
        & frequent_items[product2]
        & frequent_items[product3]
    )
    support = len(common_transactions) / total_transactions
    if support >= MIN_SUPPORT:
        frequent_triples.append(
            {
                "itemset": (product1, product2, product3),
                "support": support,
                "support_percentage": support * 100,
            }
        )

# IMPORTANT FIX: explicit columns prevent KeyError: 'support' when no triples exist.
frequent_triples_df = pd.DataFrame(
    frequent_triples,
    columns=["itemset", "support", "support_percentage"],
)
if not frequent_triples_df.empty:
    frequent_triples_df = frequent_triples_df.sort_values(by="support", ascending=False)
print("\nFrequent ECLAT triples:")
print(frequent_triples_df.head(10).to_string(index=False))
print("Total frequent triples:", len(frequent_triples_df))

# Plot frequent pairs if there are any.
if not frequent_pairs_df.empty:
    top_pairs = frequent_pairs_df.head(10).copy()
    top_pairs["pair_label"] = top_pairs["itemset"].apply(lambda items: " + ".join(items))
    plt.figure(figsize=(10, 6))
    plt.barh(top_pairs["pair_label"], top_pairs["support_percentage"])
    plt.xlabel("Support (%)")
    plt.ylabel("Product Pair")
    plt.title("Top Frequent Product Pairs — ECLAT")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Compare Apriori and ECLAT pairs.
apriori_pairs = {
    frozenset(itemset)
    for itemset in multiple_items["itemsets"]
    if len(itemset) == 2
}
eclat_pairs = {frozenset(itemset) for itemset in frequent_pairs_df["itemset"]}
print("\n=== APRIORI vs ECLAT PAIR COMPARISON ===")
print("Apriori frequent pairs:", len(apriori_pairs))
print("ECLAT frequent pairs:", len(eclat_pairs))
print("Pairs found by both:", len(apriori_pairs & eclat_pairs))
print("Pairs found only by Apriori:", len(apriori_pairs - eclat_pairs))
print("Pairs found only by ECLAT:", len(eclat_pairs - apriori_pairs))
if apriori_pairs == eclat_pairs:
    print("Both algorithms found the same frequent pairs.")
else:
    print("The frequent pair results differ; check that the basket and thresholds match.")

# Compare support values in a table.
# Apriori stores itemsets as frozensets, while ECLAT stores them as tuples.
# Convert both to the same sorted string key before merging.
apriori_pair_df = multiple_items[multiple_items["itemset_length"] == 2][
    ["itemsets", "support"]
].copy()
apriori_pair_df["itemset_key"] = apriori_pair_df["itemsets"].apply(
    lambda items: " | ".join(sorted(items))
)
apriori_pair_df = apriori_pair_df.rename(columns={"support": "apriori_support"})
apriori_pair_df = apriori_pair_df[["itemset_key", "apriori_support"]]

eclat_pair_comparison = frequent_pairs_df[["itemset", "support"]].copy()
eclat_pair_comparison["itemset_key"] = eclat_pair_comparison["itemset"].apply(
    lambda items: " | ".join(sorted(items))
)
eclat_pair_comparison = eclat_pair_comparison.rename(
    columns={"support": "eclat_support"}
)[["itemset_key", "eclat_support"]]

comparison_df = apriori_pair_df.merge(
    eclat_pair_comparison, on="itemset_key", how="outer"
).rename(columns={"itemset_key": "itemset"})
print("\nSupport comparison (Apriori vs ECLAT):")
print(comparison_df.to_string(index=False))

# ------------------------------------------------------------
# 8. ECLAT association rules from frequent pairs
# ------------------------------------------------------------
eclat_rules = []
for _, row in frequent_pairs_df.iterrows():
    product1, product2 = row["itemset"]
    transactions1 = frequent_items[product1]
    transactions2 = frequent_items[product2]
    common_transactions = transactions1 & transactions2

    support = len(common_transactions) / total_transactions
    confidence_1_to_2 = len(common_transactions) / len(transactions1)
    confidence_2_to_1 = len(common_transactions) / len(transactions2)
    support_1 = len(transactions1) / total_transactions
    support_2 = len(transactions2) / total_transactions
    lift_1_to_2 = confidence_1_to_2 / support_2
    lift_2_to_1 = confidence_2_to_1 / support_1

    eclat_rules.append(
        {
            "antecedents": product1,
            "consequents": product2,
            "support": support,
            "confidence": confidence_1_to_2,
            "lift": lift_1_to_2,
        }
    )
    eclat_rules.append(
        {
            "antecedents": product2,
            "consequents": product1,
            "support": support,
            "confidence": confidence_2_to_1,
            "lift": lift_2_to_1,
        }
    )

eclat_rules_df = pd.DataFrame(
    eclat_rules,
    columns=["antecedents", "consequents", "support", "confidence", "lift"],
)
if not eclat_rules_df.empty:
    eclat_rules_df = eclat_rules_df.sort_values(by="lift", ascending=False)
print("\nECLAT association rules:")
print(eclat_rules_df.head(10).to_string(index=False))
strong_eclat_rules = eclat_rules_df[eclat_rules_df["lift"] > 1]
print("\nECLAT rules with lift > 1:")
print(strong_eclat_rules.head(10).to_string(index=False))
print("Rules with lift > 1:", len(strong_eclat_rules))

# ------------------------------------------------------------
# 9. Save ECLAT results
# ------------------------------------------------------------
frequent_pairs_df.to_csv("eclat_frequent_pairs.csv", index=False)
frequent_triples_df.to_csv("eclat_frequent_triples.csv", index=False)
eclat_rules_df.to_csv("eclat_association_rules.csv", index=False)
print("\nECLAT results saved successfully!")

# ------------------------------------------------------------
# 10. Timing note
# ------------------------------------------------------------
# Apriori searches itemsets of multiple sizes; the ECLAT timing here would
# only measure pair generation, so these times are not directly comparable.
start_time = time.perf_counter()
timed_pairs = []
for product1, product2 in combinations(frequent_items.keys(), 2):
    common_transactions = frequent_items[product1] & frequent_items[product2]
    support = len(common_transactions) / total_transactions
    if support >= MIN_SUPPORT:
        timed_pairs.append((product1, product2, support))
eclat_pair_time = time.perf_counter() - start_time
print("ECLAT pair-generation time:", round(eclat_pair_time, 4), "seconds")
print("Script finished.")
