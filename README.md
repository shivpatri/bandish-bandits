## The Multi-Armed Bandit Problem

The Multi-Armed Bandit problem is a classic dilemma in probability theory and machine learning. It involves an agent who, at each of a sequence of trials, must choose one out of several "arms" (or actions) to play. Each arm provides a random reward from an unknown probability distribution. The objective of the agent is to maximize the cumulative reward over a series of trials.

![jpg](Bandits_files/bandits.JPG)

### Key Concepts:

*   **Bandits (or Arms)**: These represent different choices or actions available to the agent. Each bandit has an associated, often unknown, reward distribution.
*   **Rewards**: When an arm is pulled, a reward is received, drawn from that arm's specific distribution. The goal is to maximize the sum of these rewards.
*   **Trials**: The problem unfolds over a series of discrete time steps, or trials.
*   **Exploration vs. Exploitation**: This is the core trade-off:
    *   **Exploitation**: Choosing the arm that currently appears to be the best (based on past experience) to maximize immediate reward.
    *   **Exploration**: Choosing a less-known arm to gather more information about its potential reward, which might lead to higher rewards in the long run.

### Why is it a problem?

If the reward distributions of the arms were known, the agent would simply always choose the arm with the highest expected reward. However, since these distributions are unknown, the agent must learn them through experience. This means the agent has to balance trying out new arms (exploration) with sticking to arms that have performed well so far (exploitation).

### Common Algorithms:

Several algorithms are designed to address the exploration-exploitation trade-off problems, such as:

*   **Epsilon-Greedy**: With probability $\epsilon$ (epsilon), explore by choosing a random arm; otherwise, exploit by choosing the arm with the highest estimated average reward.
*   **Upper Confidence Bound (UCB)**: Selects arms based on their estimated average reward plus an exploration bonus that favors arms that have been sampled fewer times or have high variance in their rewards.

### Applications:

Multi-armed bandit problems are highly relevant in various fields, including:

*   **Online Advertising**: Deciding which ad to show to a user to maximize click-through rates.
*   **Clinical Trials**: Choosing which drug treatment to administer to patients to maximize success rates.
*   **Website Optimization (A/B Testing)**: Deciding which version of a webpage layout to use to maximize user engagement.
*   **Resource Allocation**: Allocating limited resources to different projects with uncertain outcomes.

You can know more about the problem in this [video](https://youtu.be/8CquWcViBfg).


We will compare the performance of UCB1 and Epsilon greedy strategies for this problem in the following experiments.


### Bandit Reward Distributions

To analyze the performance of the UCB and Epsilon-Greedy algorithms, we use six distinct probability distributions for the bandits. Below are the probability distribution plots for each arm:

---

#### 1. Uniform Distribution (Bandit 0)
#### 2. Trapezoidal Distribution (Bandit 1)
#### 3. Beta Distribution (Bandit 2)
#### 4. Truncated Normal Distribution (Bandit 3)
#### 5. Triangular Distribution (Bandit 4)
#### 6. Wigner Semicircle Distribution (Bandit 5)
    
![png](Bandits_files/Bandits_4_0.png)
    

### The UCB Algorithm

The **Upper Confidence Bound (UCB1)** algorithm is based on the principle of "optimism in the face of uncertainty." It aims to solve the exploration-exploitation trade-off by assigning a value to each arm that represents a plausible upper limit on its true mean reward.

#### 1. Hoeffding's Inequality
UCB1 is derived from **Hoeffding's Inequality**, which provides an upper bound on the probability that the sum of bounded independent random variables deviates from its expected value. For a random variable $X_i \in [0, 1]$:

$$P\left( |\bar{X}_i - \mu_i| \ge \delta \right) \le 2e^{-2n_i\delta^2}$$

Where:
- $\bar{X}_i$ is the empirical mean of arm $i$.
- $\mu_i$ is the true mean of arm $i$.
- $n_i$ is the number of times arm $i$ has been pulled.
- $\delta$ is the confidence width.

#### 2. The Upper Bound Calculation
In the UCB1 algorithm, we want the true mean $\mu_i$ to be less than our estimate $\bar{X}_i + \delta$ with high probability. By setting the probability of failure to $1/t^2$ (where $t$ is the total number of trials), we solve for $\delta$:

$$2e^{-2n_i\delta^2} = \frac{1}{t} \implies \delta_i = \sqrt{\frac{2 \ln t}{n_i}}$$

#### 3. The UCB1 Selection Rule (Weighted Implementation)
As seen in our simulation code, we use a weighted version of the selection rule to provide finer control over the agent's behavior. At each time step $t$, the agent selects the arm $i$ that maximizes:

$$\text{UCB}_i(t) = ({w_\text{explore}} \times \bar{X}_i) + (w_\text{exploit} \times \sqrt{\frac{2 \ln t}{n_i}})$$

- **Exploitation Term ($w_{exploit} \times \bar{X}_i$)**: Favors arms that have performed well in the past. The **Exploitation Weight** (defined as $w_\text{exploit} = 1 - w_\text{explore}$) allows us to scale the importance of known historical performance.
- **Exploration Term ($w_{explore} \times \sqrt{\frac{2 \ln t}{n_i}}$)**: Favors arms that have not been pulled many times. The **Exploration Weight** allows us to manually tune how much the agent prioritizes gathering new information. As $t$ increases, this term grows slowly for all arms, but as $n_i$ increases for a specific arm, its exploration bonus shrinks, reflecting increased certainty.

### The Epsilon-Greedy Algorithm

The **Epsilon-Greedy** algorithm is one of the simplest yet most effective strategies for the multi-armed bandit problem. It balances exploration and exploitation using a single parameter, $\epsilon$ (epsilon), which represents the probability of exploring.

#### 1. The Selection Rule
At each trial $t$, the agent flips a biased coin:
- **Exploitation (Probability $1 - \epsilon$):** The agent chooses the arm with the highest current empirical mean reward (the "greedy" action).
  $$a_t = \text{argmax}_{i} \bar{X}_i$$
- **Exploration (Probability $\epsilon$):** The agent ignores what it has learned and chooses an arm uniformly at random from all available options.

#### 2. The Trade-off
- **A high $\epsilon$** (e.g., 0.5) means the agent spends a lot of time gathering information. While it will find the best arm quickly, it will continue to pull sub-optimal arms frequently, limiting its total reward.
- **A low $\epsilon$** (e.g., 0.01) means the agent exploits its current knowledge heavily. It may take a long time to discover which arm is truly the best, but once it does, it spends most of its time pulling that arm.

#### 3. Limitations
Unlike UCB, standard Epsilon-Greedy does not account for the *uncertainty* or variance of an arm's reward. It treats an arm pulled once the same as an arm pulled a thousand times if their average rewards are currently equal.




    
![png](Bandits_files/Bandits_10_0.png)
    


### High-Resolution TPU Simulation: Visualizing Convergence Density

To gain a deeper understanding of the algorithms beyond simple average performance, we use **JAX and TPU acceleration** to run 40 parallel simulations for each parameter configuration.

#### **Why Heatmaps?**
A single line plot only shows the mean outcome, which can hide significant variance or 'failed' runs. By generating 2D histograms (heatmaps), we can see the **density of outcomes**:
- **Vertical Spread**: Shows how much the reward varies across different random seeds at a specific trial.
- **Intensity (Color)**: Highlights where most simulations 'cluster.' A tight cluster around the optimal mean (10.0) indicates high reliability.
- **Ghost Bands**: Horizontal streaks below the optimal mean represent scenarios where the agent got stuck on a sub-optimal arm for an extended period.

#### **Experimental Goals**
1. **Observe Stability**: Contrast the 'tightness' of UCB1's convergence against the permanent variance of Epsilon-Greedy.
2. **Parameter Sensitivity**: Visually confirm how increasing $w$ or $\epsilon$ physically shifts the probability mass toward or away from the optimal reward.
3. **Identify 'Jump' Moments**: Spot the specific trials where UCB1 suddenly corrects its path from a sub-optimal arm to the best one.



    Running TPU-optimized simulations...



    
![png](Bandits_files/Bandits_13_1.png)
    


### Analysis of Simulation Results

Based on the high-resolution TPU heatmaps, we can observe distinct behaviors between the two algorithms across different parameter settings:

#### **1. Upper Confidence Bound (UCB1)**
*   **Pros:**
    *   **Efficient Convergence:** For optimal weights (e.g., $w=0.1$ to $0.5$), UCB quickly identifies the best arm (Bandit 1, Mean=10) and the average reward converges tightly around the optimal value.
    *   **Mathematical Bound:** Unlike Epsilon-Greedy, UCB has a sub-linear regret bound. It reduces exploration naturally as it becomes more confident, leading to higher long-term stability.
    *   **High Reward Ceiling:** Once the 'optimal' arm is found, the distribution of rewards becomes very narrow, showing that the algorithm consistently exploits the best choice.
*   **Cons:**
    *   **Sensitivity to Scaling ($w$):** If $w$ is too low (0.01), it may get stuck in a local optimum early on. If $w$ is too high (0.99), it over-explores, causing the average reward to dip significantly even after thousands of trials.
    *   **Initial Volatility:** The heatmaps show significant 'spread' in the early trials as the algorithm tries every arm to build confidence intervals.

#### **2. Epsilon-Greedy**
*   **Pros:**
    *   **Simplicity:** Extremely easy to implement and understand.
    *   **Robustness:** Even with a small epsilon (0.01), it eventually finds the optimal arm, though it takes much longer than UCB.
*   **Cons:**
    *   **Static Exploration:** The algorithm continues to pull random sub-optimal arms forever at a rate of $\epsilon$. This creates a 'reward ceiling' that is always strictly below the true optimal mean ($10.0$).
    *   **Slow Convergence:** At low $\epsilon$ (0.01), the heatmap shows a very slow upward trend, meaning it spends many trials pulling inferior arms before the 'greedy' action takes over.
    *   **Parameter Trade-off:** At high $\epsilon$ (0.3), the algorithm finds the best arm fast, but the average reward is significantly lower because it is wasting 30% of its trials on random choices.

### **Summary Comparison**
| Feature | UCB1 | Epsilon-Greedy |
| :--- | :--- | :--- |
| **Exploration Strategy** | Optimistic (Uncertainty-based) | Random (Fixed probability) |
| **Long-term Performance** | Higher (Exploration decays) | Lower (Exploration is constant) |
| **Ease of Tuning** | Moderate (Requires range scaling) | Easy (Intuitive 0 to 1 range) |

### Mathematical Insight: Noticible Late Convergence Behavior in UCB1$

In the heatmap for UCB ($w=0.5$), you might notice some horizontal bands that suddenly curve upward toward the optimal mean ($10.0$) late in the simulation. This visualizes a critical moment in the algorithm's logic:

*   **The Trap**: Early on, a sub-optimal arm (like Bandit 2 or 3) might return a string of lucky high rewards. Because the exploration weight $w$ is relatively high, the algorithm stays "optimistic" about that arm for a long time.
*   **The Turning Point**: The exploration term $\sqrt{\frac{2 \ln t}{n_i}}$ shrinks very slowly. At $w=0.5$, it provides enough "cushion" to keep the agent pulling a decent-but-not-best arm.
*   **The Correction**: Eventually, as $n_i$ (the number of pulls of the sub-optimal arm) becomes very large, the exploration bonus for that specific arm diminishes. Simultaneously, the bonus for the *truly* optimal arm (which hasn't been pulled as much) remains relatively high due to its lower sample count.
*   **The Jump**: There is a mathematical threshold where the calculated $\text{UCB}_{optimal}$ finally exceeds the $\text{UCB}_{suboptimal}$. At that exact trial, the agent abruptly switches its primary choice to the optimal arm, causing the sharp upward curve in the average reward observed in the heatmap.
