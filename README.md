# Dynamic Pricing Modeling

Ever noticed how prices on apps like Uber or Amazon seem to change all the time? That's because of a strategy called Dynamic Pricing. It's all about adjusting prices based on what customers are willing to pay. This strategy is used in many industries, from ride-hailing to online shopping, to boost profits by finding the best price for maximum demand, revenue, or profit, depending on the company's objective.

In this repository, we'll dive into how dynamic pricing works, focusing on three methods (over the time this repository is developed):

1. Using a Multi-Armed Bandit (MAB)
2. Contextual Bandit to consider other factors that affect demand
3. Rule Based Dynamic Pricing

## 1. Multi-Armed Bandit

Imagine you're in a casino facing several slot machines (bandits), each with its own unknown payout rate. You want to figure out which machine gives the highest payout, but you can only play one machine at a time.
The challenge is to decide which machines to play to maximize your total winnings over many rounds. You can try different machines to learn their payout rates (exploration), but you also want to stick to the best machine you've found so far to win more (exploitation). 

This problem is called the **"multi-armed bandit" problem**, where you're trying to balance exploring new options and exploiting the best option you've found to maximize your rewards. 

In the context of dynamic pricing for a ride-hailing app like Gojek, the MAB algorithm could be used to determine the best price to offer to users in real-time. Each "arm" represents a different price point, and the algorithm learns over time which price point leads to the most bookings or revenue.

There are several algorithms we can use under MAB as follow:

- **Epsilon-Greedy ($\epsilon \text{ Greedy}$)**: Balances exploration (trying out different options) and exploitation (choosing the best-known option) by introducing a parameter called epsilon (ε).
- **Upper Confident Bound (UCB)**  Balances exploration and exploitation by choosing actions based on their estimated value and uncertainty.
- **Thompson Sampling**: Probabilistic approach that uses Bayesian inference to balance exploration and exploitation.

In this repository, one of the methodologies that is explored extensively is the **Thompson Sampling.**
