import pandas as pd
df= pd.read_csv("/content/bandit_practice_dataset.csv")
print("Shape:", df.shape)
print(df.head())
print("\nColumns names:")
print(df.columns)
print("\nAverage reward per arm:")
print(df[["reward_A", "reward_B", "reward_C"]].mean())

import random

random.seed(42)
epsilon=0.1
arms=["reward_A", "reward_B", "reward_C"]
counts={arm: 0 for arm in arms}
total_rewards ={arm: 0 for arm in arms}
estimated_rewards ={arm: 0.0 for arm in arms}
cumulative_reward = 0.0
for i, row in df.iterrows():
  if i< len(arms):
    choose_arm = arms[i]
  elif random.random() < epsilon:
    choose_arm = random.choice(arms)
  else:
    choose_arm = max(
        estimated_rewards,
        key=estimated_rewards.get
    )    

  reward = row[choose_arm]
  counts[choose_arm] += 1
  total_rewards[choose_arm] += reward
  cumulative_reward += reward

  estimated_rewards[choose_arm]=(
      total_rewards[choose_arm]/counts[choose_arm]
  )
print("\n----Epsilon-Greedy Results---")
print("Epsilon:", epsilon)
print("Selection counts:", counts)
print("Total reward:", cumulative_reward)
print("Estimated average rewards:", estimated_rewards)




import math

# Task 4: UCB algorithm for advertisements

ads = {
    "Shoes": "reward_A",
    "Headphones": "reward_B",
    "Smartphone": "reward_C"
}

# Count how many times each ad is shown
counts_ucb = {ad: 0 for ad in ads}

# Store total clicks (rewards) for each ad
total_clicks_ucb = {ad: 0 for ad in ads}

# Store estimated average reward
estimated_ctr_ucb = {ad: 0.0 for ad in ads}

cumulative_reward_ucb = 0

# Process every round
for i, row in df.iterrows():

    # Show every advertisement once initially
    if i < len(ads):
        chosen_ad = list(ads.keys())[i]

    else:
        ucb_scores = {}

        # Calculate the UCB score for each advertisement
        for ad in ads:
            average_reward = estimated_ctr_ucb[ad]

            exploration_bonus = math.sqrt(
                (2 * math.log(i + 1)) / counts_ucb[ad]
            )

            ucb_scores[ad] = average_reward + exploration_bonus

        # Choose the ad with the highest UCB score
        chosen_ad = max(ucb_scores, key=ucb_scores.get)

    # Get the reward for the chosen advertisement
    reward = row[ads[chosen_ad]]

    # Update count and total reward
    counts_ucb[chosen_ad] += 1
    total_clicks_ucb[chosen_ad] += reward
    cumulative_reward_ucb += reward

    # Update estimated average reward
    estimated_ctr_ucb[chosen_ad] = (
        total_clicks_ucb[chosen_ad] / counts_ucb[chosen_ad]
    )

# Display the final results
print("\n--- UCB Advertisement Results ---")

for ad in ads:
    print(f"\nAdvertisement: {ad}")
    print("Times shown:", counts_ucb[ad])
    print("Clicks:", int(total_clicks_ucb[ad]))
    print(
        "Estimated click rate:",
        round(estimated_ctr_ucb[ad] * 100, 2),
        "%"
    )

print("\nTotal reward:", cumulative_reward_ucb)
