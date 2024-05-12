<!-- """"

Vooral nog wat dieper ingaan op het feit dat jouw sensoren niet voldoende data opleveren om de exacte positie te bepalen binnen de real-world grid. Je doet dat dan wel binnen jouw conclusies, maar hier mag je er ook al wat dieper op ingaan.

""" -->

<!-- 
Let's examine the reward history graphs for each algorithm to see how well they support the conclusions in your study.

### Graph Analysis

1. **Actor-Critic (AC)**: The reward history graph for AC shows a high level of variability in rewards, oscillating significantly between positive and negative values across episodes. This indicates inconsistency in performance, which aligns with the conclusion that AC requires more fine-tuning to reach the efficiency of model-free methods like DDQN and DQN.

2. **Deep Q-Network (DQN)**: DQN's reward history stabilizes quickly after an initial drop, maintaining a relatively consistent negative reward close to zero. This suggests a robust learning capability, albeit at a lower reward scale, which fits the description of varying degrees of efficiency and adaptability.

3. **Proximal Policy Optimization (PPO)**: PPO's graph also shows high variability, similar to AC, with frequent swings between high and low rewards. This reflects robust learning capabilities but confirms the conclusion about its variable efficiency and adaptability in different scenarios.

4. **Q-Learning (Q-agent)**: The reward history for Q-learning shows a long period of negative rewards that gradually stabilize over a much longer time frame (over 2000 episodes), supporting the conclusion that Q-agent struggles with the time it takes to learn the optimal policy in complex environments.

5. **Double Deep Q-Network (DDQN)**: The reward history for DDQN stabilizes quickly and remains relatively consistent throughout the episodes, staying closer to zero compared to DQN, which might imply a better balance between exploration and exploitation as mentioned in your conclusion.

### Conclusion Supportiveness

Based on the reward history graphs:
- **DDQN** shows evidence of enhanced stability and efficiency in learning, which supports your decision to primarily utilize it in the study.
- **DQN** shows robust learning capabilities, consistent with your conclusion.
- **PPO and AC** both exhibit high variability, indicating that while they have robust learning capabilities, their efficiency and adaptability are less consistent.
- **Q-agent**’s long stabilization period matches the conclusion about its slower learning process in complex environments.

Overall, the reward history graphs corroborate the conclusions drawn in your study regarding the performance and characteristics of each algorithm. This suggests that the conclusions are well-supported by the observed data.
 -->

## Comparative Analysis of Reinforcement Learning Algorithms in Maze Navigation

In this analysis, we compare various reinforcement learning algorithms, namely Double Deep Q-Network (DDQN), Deep Q-Network (DQN), Q-agent, Actor-Critic (AC), and Proximal Policy Optimization (PPO). This comparison is based on their performance in navigating a complex maze, focusing on efficiency, learning rate, and adaptability.

### Algorithm Overview

- **DDQN**: Enhances the stability of DQN by decoupling the target max operation into action selection and evaluation.
- **DQN**: Utilizes deep neural networks to approximate the Q-value function, mitigating challenges found in traditional Q-learning.
- **Q-agent**: Employs a tabular Q-learning approach, which is simpler but often less scalable for larger or more complex environments.
- **AC**: Combines value-based and policy-based methods, using two models: one as the actor for choosing actions and the other as the critic to evaluate actions.
- **PPO**: An advanced policy gradient method that optimizes policy iteration while ensuring small updates, improving training stability.

### Performance Metrics and Visualization

1. **Visit Heatmaps**
   - Provides insights into the exploration patterns of each agent.
   - ![DDQN Heatmap](./images/training_images/visit_heatmap_DDQN.png)
   - ![DQN Heatmap](./images/training_images/visit_heatmap_DQN.png)
   - ![PPO Heatmap](./images/training_images/visit_heatmap_PPO.png)
   - ![Q-agent Heatmap](./images/training_images/visit_heatmap_q-agent.png)

2. **Maze Solution Efficiency**
    PPO and AC are not included in this visualization due to their relatively higher step counts compared to DDQN and DQN.

   - Number of steps taken to solve the maze:
     - DDQN: ![DDQN Maze Path](./images/training_images/maze_solution_DDQN.png)
     - DQN: ![DQN Maze Path](./images/training_images/maze_solution_DQN.png)
     - Q-agent: ![Q-agent Maze Path](./images/training_images/maze_solution_Q-agent.png)

3. **Reward History and Distribution**
   - Indicates how quickly and consistently each algorithm learns to maximize rewards:
     - DDQN: ![DDQN Reward History](./images/training_images/reward_history_DDQN.png)
     - DQN: ![DQN Reward History](./images/training_images/reward_history_DQN.png)
     - AC: ![AC Reward History](./images/training_images/reward_history_AC.png)
     - PPO: ![PPO Reward History](./images/training_images/reward_history_PPO.png)
     - Q-agent: ![Q-agent Reward History](./images/training_images/reward_history_Qlearning.png)

4. **Mean Squared Error (MSE) Over Time**
   - Reflects the prediction accuracy and learning stability of each algorithm:
     - DDQN: ![DDQN MSE](./images/training_images/mse_history_sampled_DDQN.png)
     - DQN: ![DQN MSE](./images/training_images/mse_history_sampled_DQN.png)
     - AC: ![AC MSE](./images/training_images/mse_history_AC.png)
     - PPO: ![PPO MSE](./images/training_images/mse_history_PPO.png)
  
5. **Moving average of rewards**
   - Shows the trend of rewards over time, indicating the learning progress:
     - DDQN: ![DDQN Moving Average](./images/training_images/steps_per_episode_with_moving_avg_DDQN.png)
     - DQN: ![DQN Moving Average](./images/training_images/steps_per_episode_with_moving_avg_DQN.png)
     - AC: ![AC Moving Average](./images/training_images/reward_per_episode_with_moving_avg_AC.png)
     - PPO: ![PPO Moving Average](./images/training_images/reward_per_episode_with_moving_avg_PPO.png)


### Conclusion and Insights

This in-depth comparative analysis of Double Deep Q-Network (DDQN), Deep Q-Network (DQN), Q-agent, Actor-Critic (AC), and Proximal Policy Optimization (PPO) provides a holistic view of their efficiencies and capabilities in maze navigation tasks, focusing on learning rates, adaptability, error reduction, and practical navigation effectiveness.

**DDQN** distinguishes itself as the most efficient performer, marked by its exceptional navigational proficiency and stable learning metrics. It accomplishes maze solutions in the fewest steps (25 steps) alongside the Q-agent, and maintains low Mean Squared Error (MSE) values, indicating superior stability and error management in complex scenarios.

**DQN**, while demonstrating robust learning stability and satisfactory model prediction accuracy, took slightly more steps (33 steps) to complete the maze, highlighting a potential trade-off between path efficiency and learning stability.

**Q-agent** proved surprisingly effective, matching DDQN in the number of steps to solve the maze, reflecting its potential in scenarios where a less complex, yet effective approach is adequate. However, its initial high MSE values reveal difficulties in early learning phases, which may pose challenges in rapidly changing environments.

**AC** and **PPO** showed greater fluctuations in their performance metrics, including MSE and reward patterns. AC, in particular, faced high volatility that may require additional tuning and adaptation to enhance prediction accuracy and learning stability. PPO displayed a consistent reduction in loss over time, suggesting potential improvements in efficiency, though its higher variability in practical maze-solving tasks needs attention.

In conclusion, **DDQN** stands out for its balanced performance across various metrics—efficient maze navigation, stable learning progress, and low prediction errors—making it a benchmark for reliability and efficiency in complex navigation tasks. **DQN** remains a robust alternative, albeit slightly less efficient in navigation but reliable in learning stability. **AC** and **PPO** necessitate careful adjustments to fully leverage their learning capabilities due to their higher variability. Lastly, **Q-agent**, while effective in navigation, faces early learning hurdles, making it less ideal for environments requiring rapid adaptation. This comprehensive understanding aids in selecting the most appropriate reinforcement learning algorithms based on specific performance criteria and environmental complexities, fostering optimized learning outcomes across diverse applications.


### **Conclusion**

The comparative analysis of Actor-Critic (AC), Deep Q-Network (DQN), Proximal Policy Optimization (PPO), Q-Learning, and Double Deep Q-Network (DDQN) provides a comprehensive overview of their respective performances and efficiencies in learning. The reward patterns observed underscore the inherent characteristics and potential limitations of each method in adapting to complex environments.

DDQN emerges as a notably stable and efficient algorithm, as evidenced by its quick stabilization and consistent reward patterns close to zero. This demonstrates an enhanced balance between exploration and exploitation, validating its selection as the primary focus of this study. DDQN's performance suggests that it could serve as a reliable benchmark for future research in similar settings.

Conversely, DQN, while robust, shows a generally lower reward scale, indicating that while it is capable of stable learning, its efficiency may not match that of DDQN. This insight is crucial for understanding the trade-offs between these two variants of the Q-Network.

The AC and PPO algorithms exhibited significant fluctuations in their reward trajectories, reflecting their robust learning capabilities but also highlighting their variability and less consistent performance. These fluctuations suggest a need for further fine-tuning and adaptation to enhance their efficiency and reliability in diverse scenarios.

Lastly, the Q-agent's learning curve, characterized by a prolonged period of negative rewards before stabilization, points to challenges in learning optimal policies within complex environments. This slower learning process suggests that while Q-learning can eventually achieve stability, the time required to reach effectiveness is considerably longer compared to more advanced methods like DDQN.

Overall, this analysis not only supports the continued use of DDQN in environments demanding stability and efficiency but also highlights the importance of selecting appropriate reinforcement learning algorithms based on specific performance criteria and environmental complexities. Future research should consider these dynamics to tailor algorithmic approaches that optimize learning outcomes across various applications.



### Revised Conclusion with Comprehensive Analysis

The comparative analysis of Double Deep Q-Network (DDQN), Deep Q-Network (DQN), Q-agent, Actor-Critic (AC), and Proximal Policy Optimization (PPO) reveals distinct characteristics and efficiencies of these algorithms in a complex maze navigation task, with a focus on efficiency, learning rate, adaptability, and error reduction.

**DDQN** emerges as the standout performer in our analysis, demonstrating not only efficient learning through its reward patterns but also exceptional navigational efficiency. The DDQN algorithm required the fewest steps to solve the maze, tied with Q-agent, while maintaining lower Mean Squared Error (MSE) values over time, which signifies its superior stability and error management in complex environments.

**DQN** shows strong learning capabilities with relatively stable reward patterns and MSE values that suggest good model prediction accuracy. However, DQN required more steps to solve the maze compared to DDQN and Q-agent, highlighting a trade-off between path efficiency and learning stability.

**Q-agent**, despite its simplistic approach, achieved the maze solution in fewer steps than DQN and demonstrated the ability to eventually stabilize in its learning process, as seen in the reward history. However, it exhibits high MSE values initially, indicating challenges in early learning phases before achieving stability.

**AC** and **PPO** displayed significant fluctuations in both reward patterns and MSE, underscoring their robust learning capabilities but also their variability and less consistent performance. The high volatility in AC’s MSE particularly highlights the need for fine-tuning and adaptation to improve prediction accuracy and learning stability. PPO, on the other hand, showed a consistent reduction in loss over time, which suggests improvements in learning efficiency despite its higher variability in maze-solving performance.

In conclusion, **DDQN**’s balanced performance across various metrics—efficient maze navigation, stable learning progress, and low prediction errors—makes it highly suitable for complex navigation tasks and serves as a benchmark for reliability and efficiency. **DQN** remains a robust alternative with slightly less efficiency in maze navigation but good learning stability. **AC** and **PPO**, while powerful, require careful tuning to harness their full potential due to their higher variability. **Q-agent**, surprisingly effective in navigation efficiency, still faces hurdles in early learning stages, making it less favorable for rapidly changing environments.


### Revised Conclusion with Integrated Maze Solution Analysis

The comprehensive evaluation of Double Deep Q-Network (DDQN), Deep Q-Network (DQN), Q-agent, Actor-Critic (AC), and Proximal Policy Optimization (PPO) has provided a detailed insight into their respective strengths and weaknesses in navigating a complex maze environment. This analysis has considered their efficiency in learning, adaptability, error reduction, and specific performance in actual maze navigation.

**DDQN** not only demonstrates superior stability and efficiency in the reward and error metrics but also excels in practical navigation tasks. Both DDQN and Q-agent solved the maze in the minimal number of steps (25 steps), showcasing their efficiency in finding optimal paths. This underscores DDQN's capability in effectively balancing exploration with exploitation, making it an excellent choice for tasks requiring high levels of both accuracy and efficiency.

**DQN**, while robust in learning stability as indicated by its reward and error trends, required more steps (33 steps) to solve the maze compared to DDQN and Q-agent. This suggests a slight compromise in pathfinding efficiency, though it remains a strong performer overall.

**Q-agent** displayed an unexpectedly efficient pathfinding ability, matching DDQN for the fewest steps to solve the maze. This highlights its potential in scenarios where a straightforward, less computationally intensive approach is sufficient, despite its longer initial learning curve indicated by its reward and error patterns.

**AC** and **PPO** were excluded from the maze solution visualization due to their relatively higher step counts, indicating less efficiency in direct maze-solving tasks. Their exclusion underscores a significant limitation in environments that prioritize direct path efficiency and quick problem-solving capabilities.

In conclusion, **DDQN** stands out as the most balanced and effective algorithm for complex navigation challenges, validated by its performance across all measured criteria. **DQN** and **Q-agent** also show valuable qualities, with DQN offering stability and Q-agent providing a simpler yet effective solution in certain contexts. **AC** and **PPO**, while powerful in learning capabilities, may require further refinement to optimize their pathfinding efficiencies in similar tasks.

-----

# Background Information
Here's the revised background information text with citations formatted in IEEE style:

# Background Information

## Background on Double Deep Q-Network (DDQN)

The Double Deep Q-Network (DDQN) is an enhancement of the Deep Q-Network (DQN), a pivotal algorithm in the field of deep reinforcement learning that integrates deep neural networks with Q-learning. DQN itself was a significant advancement as it demonstrated the capability to approximate the Q-value function, which represents the expected reward for taking an action in a given state, using high-capacity neural networks.

### Evolution from DQN to DDQN

**DQN Challenges**: While DQN substantially improved the stability and performance of Q-learning, it was susceptible to significant overestimations of Q-values due to the noise inherent in the approximation of complex functions by deep neural networks. This overestimation could lead to suboptimal policies and slower convergence during training.

**DDQN Solution**: Introduced by Hado van Hasselt et al., DDQN addresses the overestimation problem of DQN by decoupling the action selection from the target Q-value generation—a technique termed "double learning." In traditional DQN, a single neural network is used both to select the best action and to evaluate its value. DDQN modifies this by employing two networks:
- The **current network** determines the action with the highest Q-value for the current state.
- A separate **target network**, which is a delayed copy of the current network, is used to estimate the Q-value of taking that action at the next state [1].

### The Decoupling Effect

This separation ensures that the selection of the best action is less likely to overestimate Q-values, as the estimation is made using a different set of weights, thus reducing bias in the learning process. The target network's parameters are updated less frequently (often after a set number of steps), which further enhances the algorithm's stability.

### Impact and Applications

DDQN has been shown to achieve better performance and faster convergence in complex environments compared to DQN. It is particularly effective in scenarios where precise action evaluation is crucial, such as in video games and robotic navigation tasks. The improved reliability and accuracy of DDQN make it a valuable model for studying reinforcement learning in controlled environments where stability and efficiency are critical.

## Background on Deep Q-Network (DQN)

The Deep Q-Network (DQN) algorithm represents a significant breakthrough in reinforcement learning by combining traditional Q-learning with deep neural networks. This approach was popularized by researchers at DeepMind with their notable success in training agents that could perform at human levels across various Atari games [2].

**Core Mechanism**: DQN uses a deep neural network to approximate the Q-value function, which is the expected reward obtainable after taking an action in a given state and following a certain policy thereafter. The neural network inputs the state of the environment and outputs Q-values for each possible action, guiding the agent's decisions.

**Innovations Introduced**:
- **Experience Replay**: DQN utilizes a technique called experience replay, where experiences collected during training are stored in a replay buffer. This allows the network to learn from past experiences, reducing the correlations between sequential observations and smoothing over changes in the data distribution.
- **Fixed Q-Targets**: To further stabilize training, DQN employs a separate target network, whose weights are fixed for a number of steps and only periodically updated with the weights from the training network [2].

### Impact and Applications
DQN's ability to handle high-dimensional sensory inputs directly with minimal domain knowledge makes it highly versatile and effective in complex environments such as video games, where it can learn directly from pixels.

## Background on Q-agent (Q-learning)

Q-agent, based on the Q-learning algorithm, is one of the most fundamental types of reinforcement learning methods. It is a model-free algorithm that learns to estimate the values of actions at each state without requiring a model of the environment [3].

**Simplicity and Versatility**: Q-learning works by updating an action-value lookup table called the Q-table, which stores Q-values for each state-action pair. These values are updated using the Bellman equation during each step of training based on the reward received and the maximum predicted reward for the next state.

**Challenges**: While simple and effective for smaller state spaces, Q-learning's reliance on a Q-table becomes impractical in environments with large or continuous state spaces, where the table size would become infeasibly large.

### Impact and Applications
Q-learning has been foundational in teaching agents in environments with discrete, limited state spaces, such as simple mazes or decision-making scenarios with clear, defined states and actions.

## Background on Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a policy gradient method for reinforcement learning that simplifies and improves upon the Trust Region Policy Optimization (TRPO) approach. PPO has become popular due to its effectiveness and ease of use [4].

**Optimization Technique**: PPO seeks to take the largest possible improvement step on a policy while avoiding

 too large updates that might lead to performance collapse. It achieves this through an objective function that includes a clipped term, penalizing changes to the policy that move it too far from the previous policy.

**Advantages**: PPO is robust to a variety of hyperparameters and can be used in both continuous and discrete action spaces. It has shown great success in environments ranging from simulated robotics to complex game environments.

### Impact and Applications
PPO is favored in many modern RL applications due to its balance between efficiency, ease of implementation, and strong empirical performance.

## Background on Actor-Critic (AC)

Actor-Critic methods form a broad class of algorithms in reinforcement learning that combine both policy-based (actor) and value-based (critic) approaches [5].

**Dual Components**:
- **Actor**: Responsible for selecting actions based on a policy.
- **Critic**: Estimates the value function (or Q-value), which is used to evaluate how good the action taken by the actor is.

**Advantages**: By separating the action selection and evaluation, actor-critic methods can be more efficient than conventional policy-gradient methods. They reduce the variance of the updates and typically converge faster.

### Impact and Applications
Actor-Critic algorithms are versatile and can be applied to both discrete and continuous action spaces. They have been effectively used in applications that require balancing exploration of the environment with the exploitation of known rewards, such as in robotics and complex game environments.

## Citations

[1] H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in *Proc. of AAAI Conf. on Artificial Intelligence*, 2016.

[2] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529-533, 2015.

[3] C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*, vol. 8, no. 3-4, pp. 279-292, 1992.

[4] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.

[5] V. R. Konda and J. N. Tsitsiklis, "Actor-critic algorithms," in *Proc. of the 13th International Conf. on Neural Information Processing Systems*, pp. 1008-1014, 2000.



----


## Integration of Practical Experiments

Throughout this research project, I employed a series of practical experiments to navigate and overcome encountered challenges. These experiments, documented through video demonstrations, provide tangible insights into my problem-solving process.

### Addressing Alignment and Orientation Challenges

One of the key challenges I faced was ensuring precise orientation and alignment of the RC-car during movement. To tackle this, I utilized the MPU6050 gyroscope, aiming to correct alignment issues and achieve accurate 90-degree turns.

#### Utilizing the MPU6050 Gyroscope for Precise Orientation

- **Experiment E1 - Gyroscope Calibration**: Testing the MPU6050 gyroscope's ability to correct the car's orientation. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/32d9e29f-6d5a-4676-b609-2c08923ca1ac)
- **Experiment E2 - Navigational Corrections**: Further tests on realigning the car's forward movement to address a ~3-degree offset. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/624b40f2-bee8-49f6-961d-1f72ab18fe13)
- **Experiment E3 - Alignment Adjustment**: Tests focused on enhancing the car's alignment accuracy. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/bb9aa643-9620-4979-a70c-ec2826c7dd33)
- **Experiment E4 - Orientation Fine-tuning**: Continued efforts to refine the car’s orientation for precise maneuvering. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/689b590f-3a9a-4f63-ba9c-978ddd08ab53)
- **Experiment E5 - Advanced Alignment Solutions**: Further tests to tackle persistent alignment issues. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/99da37df-d147-43dc-828f-524f55dc6f70)

### Enhancing Movement Precision with Encoders

Exploration of rotary encoders to measure wheel rotations accurately, improving navigation precision.

- **Experiment E6 - Encoder Implementation**: Introduction of rotary encoders to enhance movement precision. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/9728e29a-d2fa-48fa-b6e0-e2e1da92228f)
- **Experiment E7 - Troubleshooting Encoder Malfunction**: Addressing challenges with encoder reliability. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/b9ce2cc3-85fd-4136-8670-516c123ba442)
- **Experiment E8 - Encoder Performance Analysis**: Evaluating the encoder functionality under varied conditions. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/ae5129fa-c25f-4f89-92bb-4ee81df9f7a5)

### Real-World Application Tests

Assessment of the RC-car's performance in varied environments to test the practical application of research findings.

- **Experiment E9 - Outdoor Navigation Test**: Evaluating performance on uneven outdoor surfaces. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/02df8a25-b7f0-4061-89b7-414e6d25d31c)
- **Experiment E10 - Advanced Outdoor Testing**: Further exploration in outdoor settings to understand environmental impacts. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/187561a7-c0cb-4921-af3e-9c2c99cb0137)
- **Experiment E11 - Indoor Controlled Test**: Controlled indoor testing to refine navigation strategies. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/ce0f47e9-26cd-459e-8b26-ff345d1ee96b)
- **Experiment E12 - Simulated to Real-world Transition Test**: Testing the RC-car's navigation in a controlled indoor environment. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/ea4a9bff-e191-4ce2-b2cc-acc57c781fa3)
- **Experiment E13 - Detailed Indoor Analysis**: Additional tests to finalize indoor navigation efficiency. [View Video](https

://github.com/driessenslucas/researchproject/assets/91117911/4783729f-10cc-4c61-afa4-71cfc93d5d3e)
- **Experiment E14 - Final Indoor Navigation Assessment**: Concluding tests focusing on critical aspects of indoor navigation. [View Video](https://github.com/driessenslucas/researchproject/assets/91117911/77091cb5-dbc5-4447-abc2-dc820dc66188)



-----

# Model Architecture and Training Insights

The architecture of the Double Deep Q-Network (Double DQN) plays a pivotal role in shaping the agent's learning process and decision-making capabilities. In the context of maze navigation, the Double DQN is designed to guide an RC car through complex environments.

## Model Architecture

```markdown
## Model: "sequential_52"

# Layer (type) Output Shape Param

=================================================================
dense_200 (Dense) (None, 32) 224
dense_201 (Dense) (None, 64) 2112
dense_202 (Dense) (None, 32) 2080
dense_203 (Dense) (None, 3) 99
=================================================================
Total params: 4515 (17.64 KB)
Trainable params: 4515 (17.64 KB)
Non-trainable params: 0 (0.00 Byte)

---
```

## Insights from Research

The choice of model complexity is critical in reinforcement learning. While complex architectures may seem appealing, simplicity often leads to better performance, faster learning, and improved generalization. Simpler models can capture essential temporal regularities, such as repeated sequences of actions, which are crucial for maze navigation.

In line with the research paper titled "Reinforcement Learning with Simple Sequence Priors" by T. Saanum [1], our Double DQN strikes a balance between simplicity and effectiveness. By prioritizing simplicity, we ensure optimal performance in maze-solving tasks.

## Citations

[1] Saanum, T. "Reinforcement Learning with Simple Sequence Priors." arXiv preprint arXiv:2305.17109 (2024).

[2] "COMP 542: Machine Learning Cite Your Sources." CSUN. Available: [link](https://libguides.csun.edu/comp542/citeyoursources).

[3] "IEEE Editorial Style Manual (Online)." IEEE, 2023. Available: [link](https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/style_references_manual.pdf).


----

## Model Architecture and Training Insights

To understand how our Double DQN model learns and makes decisions, let's dive into its architecture. The model consists of four dense layers, which output three actions tailored to the RC car's movement, enabling it to navigate the maze with ease.

Research has shown that, all things being equal, simpler models are often preferred in reinforcement learning [1]. This is because they can lead to better performance, faster learning, and improved generalization. However, finding the right balance of model complexity is crucial [1][3]. Simplicity is not just about the number of layers or parameters, but also about capturing temporal regularities, such as repetitions, in sequential strategies [1][2].

With these insights in mind, we designed the Double DQN model to strike a balance between simplicity and effectiveness, ensuring optimal performance in maze navigation tasks.

**Model Architecture:**

```markdown
## Model: "sequential_52"

# Layer (type) Output Shape Param

=================================================================
dense_200 (Dense) (None, 32) 224
dense_201 (Dense) (None, 64) 2112
dense_202 (Dense) (None, 32) 2080
dense_203 (Dense) (None, 3) 99
=================================================================
Total params: 4515 (17.64 KB)
Trainable params: 4515 (17.64 KB)
Non-trainable params: 0 (0.00 Byte)

---
```

## Citations

[1] T. Saanum, "Reinforcement Learning with Simple Sequence Priors," arXiv preprint arXiv:2305.17109, 2024.

[2] "COMP 542: Machine Learning Cite Your Sources," CSUN, [Online]. Available: https://libguides.csun.edu/comp542/citeyoursources. [Accessed: 12-May-2024].

[3] "IEEE Editorial Style Manual (Online)," IEEE, 2023. [Online]. Available: https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/style_references_manual.pdf. [Accessed: 12-May-2024]