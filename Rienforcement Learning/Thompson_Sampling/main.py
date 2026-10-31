import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("ad_click_dataset.csv")

print("Dtaset Shape:", df.shape)

print(df.head())
print(df.columns)

print(df.info())

print(df.isnull().sum())

print("Duplicates:", df.duplicated().sum())

print(df["ad_position"].value_counts())

click_rates=df.groupby("ad_position")["click"].mean()
print(click_rates)

bandit_df = df[["id", "ad_position", "click"]].copy()

print(bandit_df.head())
print("Shape:", bandit_df.shape)

print("Unique IDs:", bandit_df['id'].nunique())
print("Duplicated IDs:", bandit_df["id"].duplicated().sum())
print(
    bandit_df[
        bandit_df.duplicated(keep=False)
    ].sort_values("id").head(10)
)

bandit_df=bandit_df.dropna(
    subset=["ad_position", "click"]
)

print("Shape after cleaning:", bandit_df.shape)
print(bandit_df.isnull().sum())

click_rates = (
    bandit_df.groupby("ad_position")["click"]
    .mean()
)

print("Click Rates:")
print(click_rates)

# Define available advertisement positions
arms = ["Bottom", "Side", "Top"]

print("Available Arms:", arms)

# Store historical click rates as reward probabilities
reward_probs = click_rates.to_dict()

print("Reward Probabilities:")
print(reward_probs)

# Select an advertisement
selected_arm = "Bottom"

# Generate a random reward
reward = np.random.binomial(
    1,
    reward_probs[selected_arm]
)

print("Selected Advertisement:", selected_arm)
print("Reward:", reward)

# Initialize Alpha for all advertisements
alpha = {arm: 1 for arm in arms}

# Initialize Beta for all advertisements
beta = {arm: 1 for arm in arms}

print("Initial Alpha:", alpha)
print("Initial Beta:", beta)

if reward == 1:
    alpha[selected_arm] += 1

else:
    beta[selected_arm] += 1

print("Updated Alpha:", alpha)
print("Updated Beta:", beta)


# Generate one random sample for each advertisement
samples = {}

for arm in arms:
    samples[arm] = np.random.beta(alpha[arm], beta[arm])

print("Sampled Probabilities:", samples)

selected_arm = max(samples, key=samples.get)

print("Selected Advertisement:", selected_arm)
print("Selected Sample:", samples[selected_arm])

# Step 1: Generate samples
samples = {}

for arm in arms:
    samples[arm] = np.random.beta(alpha[arm], beta[arm])

# Step 2: Select the advertisement
selected_arm = max(samples, key=samples.get)

# Step 3: Simulate the user response
reward = np.random.binomial(1, reward_probs[selected_arm])

# Step 4: Update Alpha or Beta
if reward == 1:
    alpha[selected_arm] += 1
else:
    beta[selected_arm] += 1

# Step 5: Display results
print("Selected Advertisement:", selected_arm)
print("Reward:", reward)
print("Updated Alpha:", alpha)
print("Updated Beta:", beta)

# Number of advertisement impressions
n_rounds = 1000

# Store how many times each advertisement is selected
selection_counts = {arm: 0 for arm in arms}

# Store total clicks
total_rewards = 0

# Thompson Sampling loop
for i in range(n_rounds):

    # Generate random samples
    samples = {}

    for arm in arms:
        samples[arm] = np.random.beta(alpha[arm], beta[arm])

    # Select advertisement with highest sample
    selected_arm = max(samples, key=samples.get)

    # Count advertisement selection
    selection_counts[selected_arm] += 1

    # Simulate user click
    reward = np.random.binomial(1, reward_probs[selected_arm])

    # Update Alpha or Beta
    if reward == 1:
        alpha[selected_arm] += 1
    else:
        beta[selected_arm] += 1

    # Update total clicks
    total_rewards += reward

# Display results
print("Advertisement Selection Counts:", selection_counts)
print("Total Clicks:", total_rewards)
print("Total Impressions:", n_rounds)
print("Overall Click Rate:", total_rewards / n_rounds)
print("Final Alpha:", alpha)
print("Final Beta:", beta)

# Reset Thompson Sampling
alpha = {arm: 1 for arm in arms}
beta = {arm: 1 for arm in arms}

# Experiment settings
n_rounds = 1000

selection_counts = {arm: 0 for arm in arms}
click_counts = {arm: 0 for arm in arms}

total_rewards = 0

# Run Thompson Sampling
for i in range(n_rounds):

    # Generate samples
    samples = {}

    for arm in arms:
        samples[arm] = np.random.beta(alpha[arm], beta[arm])

    # Select the best sampled advertisement
    selected_arm = max(samples, key=samples.get)

    selection_counts[selected_arm] += 1

    # Simulate click
    reward = np.random.binomial(
        1, reward_probs[selected_arm]
    )

    # Update click counts
    click_counts[selected_arm] += reward

    # Update Alpha and Beta
    if reward == 1:
        alpha[selected_arm] += 1
    else:
        beta[selected_arm] += 1

    total_rewards += reward

# Display results
print("Selection Counts:", selection_counts)
print("Click Counts:", click_counts)
print("Total Clicks:", total_rewards)
print("Overall Click Rate:", total_rewards / n_rounds)
print("Final Alpha:", alpha)
print("Final Beta:", beta)


