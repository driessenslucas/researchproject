---
title: "Exploring the Feasibility of Sim2Real Transfer in Reinforcement Learning"
titlepage: true
toc: true
toc-own-page: true
author: "Lucas Driessens"
date: "June 1, 2024"
keywords: [Markdown, Example]
titlepage-text-color: "000000"
titlepage-rule-color: "FFFFFF"
titlepage-rule-height: 0
geometry: margin=1in
toc-title: Table of Contents
caption-justification: justified
titlepage-logo: mctlogo.png
header-logo: header_image.png
header-title: "Bachelor Thesis Howest"
text1: Bachelor Multimedia and Creative Technologies
text2: University of Applied Sciences
text3: Howest Kortrijk
text4: 2024
lof: true
lof-own-page: True
output:
  markdown::pdf_document:
    fig_caption: yes
abstract: |
  This research explores the feasibility of transferring a trained reinforcement learning (RL) agent from a simulation to the real world, focusing on maze navigation. The primary objective is to determine if and how an RL agent, specifically a Double Deep Q-Network (DDQN), can successfully navigate a physical maze after being trained in a virtual environment.

  \noindent \newline First, we explore suitable virtual environments for training an RL agent and evaluate the most effective reinforcement learning techniques for this application. The study then addresses the challenges in translating simulation-trained behaviors to real-world performance, such as sensor data interpretation and movement replication.

  \noindent \newline Results show that the DDQN agent, trained in a simulated maze, can navigate a physical maze with some challenges in sensor data interpretation and more importantly movement replication. Practical solutions, including sensor calibration and algorithm adjustments, were implemented to improve real-world performanceResults show that the DDQN agent, trained in a simulated maze, can navigate a physical maze with some challenges in sensor data interpretation and, more importantly, movement replication. Practical solutions, including sensor calibration and algorithm adjustments, were implemented to improve real-world performance.

  \noindent \newline This study contributes to AI and robotics by providing insights and methodologies for sim-to-real transfer in RL, with potential applications extending beyond robotics to other fields where simulation-based training is essential.This study contributes to AI and robotics by providing insights and methodologies for sim-to-real transfer in RL, with potential applications extending beyond robotics to other fields where simulation-based training is essential.
preface: |
  This bachelor thesis, titled “Exploring the Feasibility of Sim2Real Transfer in Reinforcement Learning,” is the final project of my studies in Multimedia & Creative Technology at Howest, University of Applied Sciences. The main question it tackles is: “Can a trained RL agent be successfully transferred from a simulation to the real world?” This question highlights my curiosity about how virtual simulations can be used in real-life applications and addresses a significant challenge in AI: making systems that can work in real-world conditions.

  \noindent \newline My interest in Sim2Real transfer started during the 'Advanced AI' classes and the 'Research Project' module. Learning about reinforcement learning and seeing how simulated environments can mimic complex real-world behaviors got me excited to explore their practical uses. This thesis delves into the theory behind Sim2Real transfer and tests its feasibility through various experiments aimed at improving the process and making it more reliable.

  \noindent \newline The research combines theoretical studies with practical experiments. The theoretical part provides a solid background, while the experiments test how well RL agents perform in different controlled scenarios. By evaluating these agents, the research aims to find the best strategies for successfully transferring them from simulations to real-world applications.
  
  \noindent \newline Lucas Driessens  
  \noindent \newline 01-06-2024
acknowledgements: |
  I extend heartfelt thanks to my coach and supervisor, Gevaert Wouter, for his invaluable guidance and insights throughout this research. His expertise and mentorship have been fundamental to my scholarly and personal growth. Gratitude is also due to Amaury Van Naemen for his technical support with 3D printing components crucial for my experiments. His assistance was pivotal in navigating the practical challenges of my research.

  \noindent \newline I would also like to thank Sam De Beuf for not only being my internship supervisor but also my external promotor for this thesis, and for allowing me the freedom to conduct research within my internship. This opportunity helped me grow significantly as a person and gain valuable soft skills, such as learning to accept feedback, considering different solutions, and being flexible in problem-solving.

  \noindent \newline Additional appreciation goes to the faculty and staff at Howest, whose commitment to fostering an innovative educational environment has profoundly influenced my development. Their dedication to excellence and support for student initiatives have been instrumental in shaping my academic journey.
---

<!-- pandoc thesis_new.md --o thesis_new.pdf -H deeplist.tex -f markdown-implicit_figures  --template template.tex --lua-filter pagebreak.lua -->
<!-- pandoc --from markdown --to html5 --standalone --toc --number-sections --citeproc --wrap=preserve --highlight-style=kate --mathml -->

\pagebreak

# Glossary of Terms

1. **Artificial Intelligence (AI)**: The simulation of human intelligence processes by machines, especially computer systems, enabling them to perform tasks that typically require human intelligence.

2. **Double Deep Q-Network (DDQN)**: An enhancement of the Deep Q-Network (DQN) algorithm that addresses the overestimation of action values, thus improving learning stability and performance.

3. **Epsilon Decay**: A technique in reinforcement learning that gradually decreases the rate of exploration over time, allowing the agent to transition from exploring the environment to exploiting known actions for better outcomes.

4. **Mean Squared Error (MSE)**: A loss function used in regression models to measure the average squared difference between the estimated values and the actual value, useful for training models by minimizing error.

5. **Motion Processing Unit (MPU6050)**: A sensor device combining a MEMS (Micro-Electro-Mechanical Systems) gyroscope and a MEMS accelerometer, providing comprehensive motion processing capabilities.

6. **Policy Network**: In reinforcement learning, a neural network model that directly maps observed environment states to actions, guiding the agent's decisions based on the current policy.

7. **Raspberry Pi (RPI)**: A small, affordable computer used for various programming projects, including robotics and educational applications.

8. **RC Car**: A remote-controlled car used as a practical application platform in reinforcement learning experiments, demonstrating how algorithms can control real-world vehicles.

9. **Reinforcement Learning (RL)**: A subset of machine learning where an agent learns to make decisions by taking actions within an environment to achieve specified goals, guided by a system of rewards and penalties.

10. **Sim2Real Transfer**: The practice of applying models and strategies developed within a simulated environment to real-world situations, crucial for bridging the gap between theoretical research and practical application.

11. **Target Network**: Utilized in the DDQN framework, a neural network that helps stabilize training by providing consistent targets for the duration of the update interval.

12. **Virtual Environment**: A simulated setting designed for training reinforcement learning agents, offering a controlled, risk-free platform for experimentation and learning.

13. **Wheel Slippage**: A phenomenon where the wheels of a vehicle lose traction, causing them to spin without propelling the vehicle forward, often encountered in real-world scenarios with uneven terrain.

14. **Ultrasonic Distance Sensor (HC-SR04)**: A sensor that uses ultrasonic waves to measure distance, commonly employed in robotics for obstacle detection and navigation.

15. **Over the air updates (OTA)**: A method of remotely updating software or firmware on devices, allowing for seamless upgrades and maintenance without physical access to the device.

16. **autonomous vehicle (AV):** A self-driving vehicle capable of navigating and operating without human intervention, relying on sensors, algorithms, and AI to perceive and interact with the environment.

\pagebreak

# List of Abbreviations

1. **AI** - Artificial Intelligence
2. **DDQN** - Double Deep Q-Network
3. **DQN** - Deep Q-Network
4. **ESP32** - Espressif Systems 32-bit Microcontroller
5. **HC-SR04** - Ultrasonic Distance Sensor
6. **MSE** - Mean Squared Error
7. **MPU6050** - Motion Processing Unit (Gyroscope + Accelerometer)
8. **PPO** - Proximal Policy Optimization
9. **RC** - Remote Controlled
10. **RPI** - Raspberry Pi
11. **RL** - Reinforcement Learning
12. **RCMazeEnv** - RC Maze Environment (Custom Virtual Environment for RL Training)
13. **Sim2Real** - Simulation to Reality Transfer
14. **OTA** Over the air updates
15. **AV** - autonomous vehicle

\pagebreak

# Introduction

## Navigating the Maze: Sim‑to‑Real Transfer in Reinforcement Learning

In our increasingly interconnected world, the boundaries between virtual simulations and tangible reality are blurring. Consider the task of training a robot in a computer simulation to navigate a physical maze. This challenge raises critical questions about the transferability of learned behaviors from the digital realm to the real world. This thesis explores the captivating intersection of Reinforcement Learning (RL) and sim-to-real transfer.

## Sim‑to‑Real Transfer: Bridging the Gap

Sim-to-real transfer involves translating learned behaviors from simulated environments to effective actions in the real world. While simulations provide a safe and controlled space for training, they often differ significantly from reality due to factors like sensor noise, friction, lighting conditions, and unexpected obstacles. This research aims to address these discrepancies.

## The Maze Navigation Challenge: RC Cars and Algorithms

This study focuses on maze navigation using a remote-controlled (RC) car equipped with sensors. The car learns optimal paths, avoids dead ends, and optimizes its trajectory in a simulated maze. The key question is whether this digital training can translate seamlessly to navigating a physical maze, where real-world challenges like friction and uneven terrain await.

## The Expedition: Four Key Steps

1. **Simulator Design:** Creating a realistic maze simulator that captures physical nuances such as wheel slippage, sensor noise, and limited field of view. This simulator allows the virtual car to explore and learn through trial and error, ensuring the RL agent is well-prepared for real-world conditions.

2. **Transfer Learning Strategies:** Employing techniques like domain adaptation, fine-tuning, and meta-learning to bridge the gap between simulation and reality. These strategies ensure the RL agent can adapt its learned behaviors to varied and unpredictable real-world conditions.

3. **Sensor Calibration:** Calibrating the RC car's sensors, and motors to match their virtual counterparts. This involves exploring sensor fusion and adaptation methods to maintain the integrity of the agent’s learned behaviors and ensure accurate real-world operation.

4. **Robust Policies:** Developing policies resilient to noisy data and unexpected scenarios. This involves training the RL agent with a variety of disturbances in the simulation to handle real-world complexities, such as dynamic obstacles and environmental changes.

## Beyond Mazes: A Broader Canvas

While this research focuses on maze navigation, its implications extend far beyond. The principles of sim-to-real transfer can be applied to autonomous drones in urban landscapes, self-driving cars avoiding pedestrians, or medical robots operating in cluttered hospital rooms. Sim-to-real transfer is the key to making these scenarios feasible.

So buckle up (or tighten your wheel nuts), as we embark on this thrilling expedition. The RC car awaits, ready to unravel the mysteries of both simulation and reality.

<!-- ## Background on Reinforcement Learning

Reinforcement Learning (RL) employs a computational approach where agents learn to optimize their action sequences through trials and errors, engaging with their environment to maximize rewards over time. This learning framework is built upon the foundation of Markov Decision Processes (MDP), which includes:

- **States ($S$)**: A definitive set of environmental conditions.
- **Actions ($A$)**: A comprehensive set of possible actions for the agent.
- **Transition Probabilities ($P(s_{t+1} | s_t, a_t)$)**: The likelihood of moving from state $s_t$ to state $s_{t+1}$ after the agent takes action $a_t$ at time $t$.
- **Rewards ($R(s_t, a_t)$)**: The reward received when transitioning from state $s_t$ to state $s_{t+1}$ due to action $a_t$.

The principles of Reinforcement Learning, particularly the dynamics of Markov Decision Processes involving states $S$, actions $A$, transition probabilities $P(s_{t+1} | s_t, a_t)$, and rewards $R(s_t, a_t)$, form the foundation of how agents learn from and interact with their environment to optimize decision-making over time. This understanding is crucial in the development of autonomous vehicles, improving navigational strategies, decision-making capabilities, and adaptation to real-time environmental changes. The seminal work by R.S. Sutton and A.G. Barto significantly elucidates these principles and complexities of RL algorithms \hyperref[ref18]{[18]}. -->

# Background on Reinforcement Learning

The challenge of sim-to-real transfer is pivotal in the deployment of autonomous systems, influencing applications ranging from robotic navigation to self-driving vehicles (Sutton & Barto, 2018; van Hasselt et al., 2016). Recent advancements in RL, such as the introduction of Proximal Policy Optimization (Schulman et al., 2017) and Soft Actor-Critic algorithms (Haarnoja et al., 2018), have shown promise in various domains. However, the discrepancy between simulated and real environments, often referred to as the 'reality gap' (Koos et al., 2013), poses a major hurdle.

Several approaches have been proposed to bridge this gap. Domain randomization, for instance, involves training models on a variety of simulated environments with different parameters to improve their robustness (Tobin et al., 2017). Another promising technique is domain adaptation, which seeks to align the simulated and real-world data distributions (Ganin et al., 2016). Despite these advancements, challenges remain, particularly in ensuring the transferability of learned behaviors in complex, dynamic environments (Zhao et al., 2020).

This thesis builds on these foundations by exploring the feasibility of transferring RL agents trained in a simulated maze environment to a real-world RC car setup. By leveraging the Double Deep Q-Network (DDQN) architecture, known for its reduced overestimation bias (van Hasselt et al., 2016), this study aims to enhance the reliability of sim-to-real transfer in maze navigation tasks. The chosen approach addresses the limitations of prior methods by integrating robust policy development and comprehensive sensor calibration, providing a novel contribution to the field.

Reinforcement Learning (RL) employs a computational approach where agents learn to optimize their action sequences through trials and errors, engaging with their environment to maximize rewards over time. This learning framework is built upon the foundation of Markov Decision Processes (MDP), which includes:

- **States ($S$)**: A definitive set of environmental conditions.
- **Actions ($A$)**: A comprehensive set of possible actions for the agent.
- **Transition Probabilities ($P(s_{t+1} | s_t, a_t)$)**: The likelihood of moving from state $s_t$ to state $s_{t+1}$ after the agent takes action $a_t$ at time $t$.
- **Rewards ($R(s_t, a_t)$)**: The reward received when transitioning from state $s_t$ to state $s_{t+1}$ due to action $a_t$.

The principles of Reinforcement Learning, particularly the dynamics of Markov Decision Processes involving states $S$, actions $A$, transition probabilities $P(s_{t+1} | s_t, a_t)$, and rewards $R(s_t, a_t)$, form the foundation of how agents learn from and interact with their environment to optimize decision-making over time. This understanding is crucial in the development of autonomous vehicles, improving navigational strategies, decision-making capabilities, and adaptation to real-time environmental changes. The seminal work by R.S. Sutton and A.G. Barto significantly elucidates these principles and complexities of RL algorithms \hyperref[ref18]{[18]}.

<!-- \pagebreak -->

# Research Questions

This investigation centers around the question: "Is it possible to transfer a trained RL-agent from a simulation to the real world? (case: maze)" To address this question, we'll explore various aspects of RL training and implementation:

1. **Which virtual environments exist to train a virtual RC-car?**: determine which virtual environments are most effective for RL training.
2. **Which reinforcement learning techniques can I best use in this application?**: Identifying RL techniques suitable for autonomous navigation.
3. **Can the simulation be transferred to the real world? Explore the difference between how the car moves in the simulation and in the real world.**: Assessing how well the agent adapts to real-world dynamics.
4. **Does the simulation have any useful contributions? In terms of training time or performance.**: Evaluating training effectiveness and optimizing performance through simulation.
5. **How can the trained model be transferred to the real RC car? How do we need to adjust the agent and the environment for it to translate to the real world?**: Discussing necessary adjustments for real-world application.

<!--
## Main Research Question

**Is it possible to transfer a trained RL-agent from a simulation to the real world? (case: maze)**

## Sub Research Questions

1. Which virtual environments exist to train a virtual RC-car?

2. Which reinforcement learning techniques can I best use in this application?

3. Can the simulation be transferred to the real world? Explore the difference between how the car moves in the simulation and in the real world.

4. Does the simulation have any useful contributions? In terms of training time or performance?

5. How can the trained model be transferred to the real RC car? (sim2real) How do you need to adjust the agent and the environment for it to translate to the real world? -->

# Methodology

This section explores the Reinforcement Learning Maze Navigation (RCMazeEnv) method, which utilizes a Double Deep Q-Network (DDQNAgent) architecture. We will delve into the maze environment setup, the design of the DDQN agent, and the comprehensive training algorithm, incorporating mathematical functions to describe the system's mechanics.

## Environment Setup (RCMazeEnv)

RCMazeEnv, a custom 12x12 cell maze built on OpenAI Gym, features walls ('1') and paths ('0'). The agent starts at position (1,1) aiming to reach (10,10), equipped with forward, left, and right sensors.

To aid with navigation, the agent has sensors providing readings in three directions: front, left, and right. These sensors measure the distance to the nearest wall in their respective directions, crucial for decision-making. The environment's state space ($\mathcal{S}$) includes the agent's current position $(x, y)$, orientation $\theta$ (north, east, south, or west), and sensor readings $\{s_{\text{front}}, s_{\text{left}}, s_{\text{right}}\}$. The agent's goal is efficient maze navigation, reaching the goal while avoiding collisions with walls or getting stuck in corners all while optimizing its path based on sensor inputs and past experiences.

<!-- ![Real life Maze Build (Image created by author)](./images/final_test/final_maze_build.jpeg "Real life Maze Build (Image created by author)") -->
\begin{figure}[htbp]
    \centering
    \begin{minipage}{0.8\textwidth}
        \includegraphics[width=4in]{./images/final_test/final_maze_build.jpeg}
    \end{minipage}
    \caption{Real life Maze Build (Image created by author)}
\end{figure}

### Web Application Interface

A web application was developed to serve as a control interface for the RC car, allowing real-time monitoring and control of the car's movements. The interface displays sensor readings and includes an emergency stop feature.

<!-- ![Web App (Image created by author)](./images/thesis/web_app.png "Web App (Image created by author)") -->
\begin{figure}[htbp]
    \centering
    \begin{minipage}{0.8\textwidth}
        \includegraphics[width=4in]{./images/thesis/web_app.png}
    \end{minipage}
    \caption{Web App (Image created by author)}
\end{figure}

## Agent Design (DDQNAgent)

The agent uses a Double Deep Q-Network (DDQN) architecture to learn the optimal policy $\pi^*$. DDQN is an enhancement over the standard DQN, aiming to reduce overestimation of Q-values by separating action selection from evaluation \hyperref[ref19]{[19]}.

- **Policy Network**: Estimates the Q-value $Q(s, a; \theta)$ for taking action $a$ in state $s$, with weights $\theta$. This network selects actions based on the current policy.
- **Target Network**: Independently parameterized by weights $\theta^-$, it estimates the target Q-value for updating the policy network. The target network mirrors the policy network's architecture but updates less frequently to provide stable target values.

The DDQN update equation modifies the Q-function:

$$
Y_t^{DDQN} = R_{t+1} + \gamma Q\left(S_{t+1}, \underset{a}{\mathrm{argmax}}\, Q(S_{t+1}, a; \theta); \theta^-\right)
$$

Where:

- $R_{t+1}$ is the reward received after taking action $a$ in state $s$.
- $\gamma$ is the discount factor.
- $\underset{a}{\mathrm{argmax}}\, Q(S_{t+1}, a; \theta)$ selects the action using the policy network.
- $Q\left(S_{t+1}, a; \theta^-\right)$ evaluates the action using the target network.

This approach reduces overestimation by separating the max operation in the target, mitigating overoptimism observed in Q-learning \hyperref[ref20]{[20]}.

The action space $\mathcal{A}$ and other agent setup details remain consistent. DDQN significantly improves stability and performance by addressing Q-value overestimation, although its effectiveness varies depending on the task compared to traditional DQN approaches \hyperref[ref21]{[21]}.

## Training Process

The training process involves using experience replay, where transitions $(s, a, r, s')$ are stored in a replay buffer denoted as $D$. Our objective is to train a Double Deep Q-Network (DDQN) by minimizing the loss function $L(\theta)$. This loss function quantifies the discrepancy between the current Q-values and the target Q-values:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)}\left[\left(r + \gamma Q(s', \underset{a'}{\mathrm{argmax}}\, Q(s', a'; \theta); \theta^-) - Q(s, a; \theta)\right)^2\right]
$$

Where:

- $s$ represents the current state.
- $a$ corresponds to the action taken.
- $r$ denotes the received reward.
- $s'$ signifies the subsequent state.
- $\theta^-$ refers to the weights of the target network.
- $\gamma$ represents the discount factor.

To enhance training stability, we periodically update the target network's weights with those of the policy network. Additionally, we employ an epsilon-greedy strategy for action selection. Initially, we prioritize exploration (with $\epsilon$ set to 0.99), gradually reducing exploration as training progresses with a decay rate of 0.99973 . This balance between exploration and exploitation contributes to the DDQN's overall performance.

# Reward Function and completion components

In the context of maze navigation, designing an effective reward function is crucial for guiding an agent's learning process. Below, I outline the key components of the reward function used in our framework:

1. **Goal Achievement Bonus ($R_{\text{goal}}$):**

   - Reaching the goal is the primary objective of the maze navigation task.
   - Upon achieving this objective, the agent receives a substantial reward: $R_{\text{goal}} = +500$.
   - However, if the agent takes an excessively long route to reach the goal (more than 1000 steps), it gets a penalty: $R_{\text{goal}} = -200$.
   - This mechanism encourages efficient navigation while still rewarding successfully reaching the goal

2. **Proximity Reward ($R_{\text{proximity}}$):**

   - Encourages the agent to minimize its distance to the goal over time.
   - The reward decreases as the distance to the goal increases: $R_{\text{proximity}} = \frac{50}{d_{\text{goal}} + 1}$.
   - Here, $d_{\text{goal}}$ represents the Euclidean distance to the goal.

3. **Progress Reward ($R_{\text{progress}}$):**

   - Provides immediate feedback based on the agent's movement relative to the goal.
   - If the distance to the goal decreases, the agent receives a positive reward: $R_{\text{progress}} = +50$.
   - if the distance increases, it gets a penalty: $R_{\text{progress}} = -25$.
   - This encourages smarter navigation decisions.

4. **Exploration Penalty ($R_{\text{revisit}}$):**

   - Discourages repetitive exploration of the same areas.
   - The agent receives a penalty for re-entering previously visited cells: $R_{\text{revisit}} = -10$.
   - This promotes exploration of new paths and prevents the agent from getting stuck.

5. **Efficiency Penalty ($R_{\text{efficiency}}$):**
   - Every step the agent takes incurs a small penalty: $R_{\text{efficiency}} = -5$.
   - Balances the need for exploration with the goal of reaching the destination efficiently.

## Termination conditions

To determine whether the environment has reached a "done" or "ended" state, 2 conditions have been established. These conditions include: surpassing 3000 steps and the RC car reaching the goal position of $(10,10)$.

The termination condition can be expressed as:

$$
\text{terminate}(steps, position) = \begin{cases}
\text{true, "Exceeded max steps"} & \text{if } steps > 3000 \\
\text{true, "Goal reached"} & \text{if } position = (10, 10) \\
\text{false, "Continue"} & \text{otherwise}
\end{cases}
$$

## Expanding Real-World Testing

In this study, I conducted experiments indoors to closely replicate theoretical conditions. The tests were performed on a hard cloth surface to minimize ground-related issues and ensure a consistent testing environment. This step was crucial because during real-world testing, the RC car encountered challenges on uneven surfaces.

However, the exploration wasn't limited to indoor setups alone. I also aimed to assess the adaptability and resilience of my proposed solutions in outdoor environments. Taking the experiments outdoors posed significant challenges due to the differences in ground conditions. Outdoor landscapes are diverse and unpredictable, which exposed limitations in my current method's ability to handle such variations. This highlighted the need for further research and improvements in the methods used, such as the hardware limitations.

# Experimental Outcomes and Implementation Details

## Simulation Design and Agent Framework

- **RCMazeEnv**: This environment was specifically designed for the study. It simulates a robotic car navigating through a maze, aiming to replicate real-world physics and limitations. The maze layout, the robotic car's movement capabilities, and its sensor configurations all contribute to creating an authentic testbed for reinforcement learning algorithms.

- **Double Deep Q-Network (DDQN)**: Instead of relying on a single neural network, DDQN employs two networks to enhance standard reinforcement learning techniques. By doing so, it mitigates the overvaluation of Q-values. The policy network and the target network collaborate, continuously interpreting sensor data and improving the learning process.

## Implementation Insights

- **Interaction Between Environment and Agent**: The DDQN agent adapts dynamically to the environment. It relies on sensor feedback to make decisions and fine-tune its path through the maze. This ongoing learning cycle is demonstrated on a simulation platform, allowing us to observe how the agent's tactics evolve over time.

- **Application in the Real World**: Transitioning from virtual training to a physical RC robot involved significant hardware adjustments and fine-tuning. Challenges like aligning sensor data and achieving precise movement control were addressed to ensure a seamless transfer from the virtual model to practical use.

## Performance Evaluation

To evaluate the agent's effectiveness in navigating the maze, I used several metrics such as manually looking at the decision and seeing if the agent would get the car stuck, Reward, loss and epsilon history per episode during training. These metrics provided insights into the agent's learning progress and performance and allowed for quick adjustments to reward functions or hyperparameters.

## Distinctive Elements

- **Physical Maze and Digital Interface**: A real maze was built to mirror the virtual `RCMazeEnv`, which is one of the requirements of ensuring the RC robot's ability to navigate the real maze. In addition, a web application was made to not only control the RC car (starting and stopping movement) but also as a virtual twin. Allowing us to see the sensor readings and decision making in real time.

<!--TODO: Rewrite this, is exludes lot of important info while being too difficult to read + it drifts too much from the actual research questions 8520
  -->
<!-- # Analysis and Results: Addressing Research Questions

## Virtual Environments for RF-Car Training

Choosing the right virtual environment is crucial for RF-car training. Among the various platforms—Unity 3D, AirSim, CARLA, OpenAI Gym, and ISAAC Gym—I selected OpenAI Gym for its flexibility in creating custom environments and compatibility with Python. This choice supports seamless integration with advanced AI coursework and facilitates effective sim-to-real transfer practices \hyperref[ref1]{[1]}.

While Unity 3D and AirSim offer realistic simulations, their complexity extends beyond Python, limiting their accessibility for my project. CARLA, designed for traditional vehicle models, is less suited for RF-cars, and ISAAC Gym, focused on robotics, doesn't align perfectly with my goals. OpenAI Gym's simplicity and reinforcement learning focus make it an ideal fit.

## Reinforcement Learning Techniques for Virtual RF-Car Training

Among the reinforcement learning techniques—Deep Q-Network (DQN), Double Deep Q-Network (DDQN), and Proximal Policy Optimization (PPO)—DDQN best meets the requirements for this project. DDQN's architecture addresses the overestimation bias in DQN, enhancing Q-value approximation accuracy, which is crucial for navigating complex, sensor-driven RF-car environments.

While DQN excels in processing high-dimensional sensory inputs, it struggles in unpredictable dynamic environments. DDQN overcomes this limitation. Although PPO focuses on direct policy optimization, it lacks the precision in value estimation needed for RF-car training. Empirical trials confirm DDQN's superior performance in intricate maze-like virtual RF-car scenarios \hyperref[ref3]{[3]}.

## Sim-to-Real Transfer Challenges and Solutions

Transferring simulation models to real-world applications involves addressing discrepancies in sensor data interpretation, action synchronization, and physical dynamics. Solutions such as sensor data normalization and action synchronization mechanisms were implemented to align simulation outcomes with real-world performance \hyperref[ref5]{[5]}.

Initially, I overlooked the importance of introducing failsafe mechanisms and adjusting motor control timings, but these proved essential in reducing collision risks and movement inaccuracies during sim-to-real transfer. Iterative testing and adaptation play a vital role in this process \hyperref[ref6]{[6]}.

## Contributions of Simulation in RF-Car Training

Simulation training offers significant efficiency, safety, and computational advantages. It allows for uninterrupted, automated training sessions, eliminating real-world risks. Leveraging powerful computing resources accelerates the training process, making simulation indispensable for RF-car development \hyperref[ref7]{[7]}.

Comparing simulation and real-world training outcomes highlights the practicality and effectiveness of simulation in developing autonomous driving models.

## Practical Application of Simulated Training to Real-World RF-Cars

Applying a trained model to a physical RC car requires several adjustments. Effective sim-to-real adaptation involves fine-tuning sensor interpretations, implementing action synchronization measures, and adjusting physical dynamics to mirror the simulation. These steps ensure successful application in real-world scenarios, facilitating robust and reliable autonomous driving systems \hyperref[ref10]{[10]}.

### Simulation Test Video

Watch the Double Deep Q-Network (DDQN) in action in this test video. It gives a real sense of how the algorithm navigates through the maze. The video can be found in `chapter 23: Video References`.

### DDQN Simulation Test

See the DDQN simulation in the Video Reference section for a detailed demonstration of the agent's performance. -->

# Addressing Research Questions

## 1. Which Virtual Environments Exist to Train a Virtual RC-Car?

Selecting the right virtual environment is crucial for effective RL training of a virtual RC car. Several platforms are available, including Unity 3D, AirSim, CARLA, OpenAI Gym, and ISAAC Gym. For this project, I chose OpenAI Gym due to its flexibility in creating custom environments and compatibility with Python. This choice supports seamless integration with advanced AI coursework and facilitates effective sim-to-real transfer practices \hyperref[ref6]{[6]}.

Unity 3D and AirSim offer highly realistic simulations but their complexity and limited Python support pose accessibility issues. CARLA, while excellent for traditional vehicle models, is less suited for RF-cars. ISAAC Gym, primarily focused on robotics, does not align perfectly with the goals of this project. OpenAI Gym’s simplicity and reinforcement learning focus make it the ideal fit for this application \hyperref[ref6]{[6]}.

## 2. Which Reinforcement Learning Techniques Can I Best Use in This Application?

For autonomous navigation of a virtual RC car in a maze, several reinforcement learning techniques were considered, including Deep Q-Network (DQN), Double Deep Q-Network (DDQN), and Proximal Policy Optimization (PPO). DDQN was chosen as the most suitable technique for this project. DDQN's architecture addresses the overestimation bias present in DQN, enhancing the accuracy of Q-value approximations, which is crucial for navigating complex, sensor-driven RF-car environments \hyperref[ref6]{[6]}.

DQN, although effective in processing high-dimensional sensory inputs, struggles in unpredictable dynamic environments. PPO, which focuses on direct policy optimization, lacks the precision in value estimation needed for RF-car training. Empirical trials confirmed that DDQN outperforms these methods in intricate maze-like virtual RF-car scenarios \hyperref[ref6]{[6]}.

## 3. Can the Simulation be Transferred to the Real World? Explore the Difference Between How the Car Moves in the Simulation and in the Real World.

Transferring simulation-trained models to real-world applications involves addressing discrepancies in sensor data interpretation, action synchronization, and physical dynamics. These differences can affect how well the agent adapts to real-world dynamics. For instance, real-world sensors may introduce noise and inaccuracies not present in the simulation, and the physical dynamics of the car, such as friction and wheel slippage, can differ significantly \hyperref[ref6]{[6]}.

To mitigate these issues, techniques such as sensor data normalization and action synchronization mechanisms were implemented to align simulation outcomes with real-world performance. Introducing failsafe mechanisms and adjusting motor control timings proved essential in reducing collision risks and movement inaccuracies during sim-to-real transfer. Iterative testing and adaptation played a vital role in this process \hyperref[ref6]{[6]}.

## 4. Does the Simulation Have Any Useful Contributions? In Terms of Training Time or Performance.

Simulation training offers significant advantages, including efficiency, safety, and computational power. It allows for uninterrupted, automated training sessions, eliminating the risks associated with real-world testing. Leveraging powerful computing resources accelerates the training process, making simulation indispensable for RF-car development \hyperref[ref6]{[6]}.

Comparing simulation and real-world training outcomes highlights the practicality and effectiveness of simulation in developing autonomous driving models. The ability to conduct extensive training in a controlled environment significantly enhances the model's performance and robustness before real-world deployment \hyperref[ref6]{[6]}.

## 5. How Can the Trained Model be Transferred to the Real RC Car? How Do We Need to Adjust the Agent and the Environment for It to Translate to the Real World?

Applying a trained model to a physical RC car requires several adjustments. Effective sim-to-real adaptation involves fine-tuning sensor interpretations, implementing action synchronization measures, and adjusting physical dynamics to mirror the simulation \hyperref[ref6]{[6]}. These steps include:

- **Sensor Calibration**: Ensuring the sensors used in the real RC car provide data in a format compatible with the trained model.
- **Motor Control Adjustments**: Adjusting motor control timings to match the physical dynamics of the real car.
- **Failsafe Mechanisms**: Introducing mechanisms to handle unexpected scenarios and reduce collision risks.
- **Incremental Testing**: Conducting iterative tests in real environments to validate and refine the agent's performance.

These adjustments are essential to ensure the successful application of the model in real-world scenarios, facilitating robust and reliable autonomous driving systems.

<!-- # Model Architecture and Training Insights

To understand how our Double DQN model learns and makes decisions, let's dive into its architecture. The model consists of four dense layers, which output three actions tailored to the RC car's movement, enabling it to navigate the maze with ease.

Research has shown that, all things being equal, simpler models are often preferred in reinforcement learning \hyperref[ref27]{[27]}. This is because they can lead to better performance, faster learning, and improved generalization. However, finding the right balance of model complexity is crucial \hyperref[ref27]{[27]}. Simplicity is not just about the number of layers or parameters, but also about capturing temporal regularities, such as repetitions, in sequential strategies \hyperref[ref27]{[27]}.

With these insights in mind, we designed the Double DQN model to strike a balance between simplicity and effectiveness, ensuring optimal performance in maze navigation tasks. -->

# Model Architecture and Training Insights

To understand how our Double DQN model learns and makes decisions, let's dive into its architecture. The model consists of four dense layers, which output three actions tailored to the RC car's movement, enabling it to navigate the maze with ease.

The Double DQN architecture addresses the issue of overestimation in action-value functions, a common problem in reinforcement learning. By decoupling the action selection and action evaluation processes during the target value calculation, the Double DQN mitigates the propagation of overestimation errors. This leads to more accurate value estimates and improved policy performance \hyperref[ref19]{[19]} \hyperref[ref20]{[20]}.

In terms of model design, Double DQN utilizes two networks: the primary network and the target network. The primary network is used to select the best action, while the target network evaluates the value of that action. This method effectively reduces the bias introduced by the max operator used in standard DQN updates \hyperref[ref19]{[19]} \hyperref[ref21]{[21]}. Empirical studies have shown that Double DQN not only improves stability and performance but also requires minimal changes to the original DQN architecture, making it a practical and efficient enhancement \hyperref[ref20]{[20]}.

Research has shown that, all things being equal, simpler models are often preferred in reinforcement learning. This is because they can lead to better performance, faster learning, and improved generalization. However, finding the right balance of model complexity is crucial. Simplicity is not just about the number of layers or parameters but also about capturing temporal regularities, such as repetitions, in sequential strategies \hyperref[ref27]{[27]} \hyperref[ref19]{[19]}.

With these insights in mind, we designed the Double DQN model to strike a balance between simplicity and effectiveness, ensuring optimal performance in maze navigation tasks. By leveraging the strengths of simpler models while addressing critical performance issues, the Double DQN maintains a robust and efficient architecture for reinforcement learning applications.

**Model Architecture:**

```markdown
# Model: "sequential_52"

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

<!-- ![Model Architecture (Image created by author)](./images/thesis/model_architecture.png "Model Architecture (Image created by author)") -->
\begin{figure}[htbp]
    \centering
    \begin{minipage}{0.8\textwidth}
        \includegraphics[width=4in]{./images/thesis/model_architecture.png}
    \end{minipage}
    \caption{Model Architecture (Image created by author)}
\end{figure}

## Training Parameters

The training of the Double DQN agent was governed by the following parameters:

- **Discount Factor (`DISCOUNT`)**: 0.90
- **Batch Size**: 128
  - Number of steps (samples) used for training at a time.
- **Update Target Interval (`UPDATE_TARGET_INTERVAL`)**: 2
  - Frequency of updating the target network.
- **Epsilon (`EPSILON`)**: 0.99
  - Initial exploration rate.
- **Minimum Epsilon (`MIN_EPSILON`)**: 0.01
  - Minimum value for exploration rate.
- **Epsilon Decay Rate (`DECAY`)**: 0.99973
  - Rate at which exploration probability decreases.
- **Number of Episodes (`EPISODE_AMOUNT`)**: 170
  - Total episodes for training the agent.
- **Replay Memory Capacity (`REPLAY_MEMORY_CAPACITY`)**: 2,000,000
  - Maximum size of the replay buffer.
- **Learning Rate**: 0.001
  - The rate at which the model learns from new observations.

## Training Procedure

1. **Initialization**: Start with a high exploration rate (`EPSILON`) allowing the agent to explore the environment extensively.
2. **Episodic Training**: For each episode, the agent interacts with the environment, collecting state, action, reward, and next state data.
3. **Replay Buffer**: Store these experiences in a replay memory, which helps in breaking the correlation between sequential experiences.
4. **Batch Learning**: Randomly sample a batch of experiences from the replay buffer to train the network.
5. **Target Network Update**: Every `UPDATE_TARGET_INTERVAL` episodes, update the weights of the target network with those of the policy network.
6. **Epsilon Decay**: Gradually decrease the exploration rate (`EPSILON`) following the decay rate (`DECAY`), shifting the strategy from exploration to exploitation.
7. **Performance Monitoring**: Continuously monitor the agent's performance in terms of rewards and success rate in navigating the maze.

<!-- ## Visual Insights and Further Exploration

This project takes an innovative approach to sim-to-real transfer in reinforcement learning, which I've showcased through a series of visual elements and demonstrations. These visuals range from the meticulous setup of our physical maze to the user-friendly design of our web application.

- **Maze Visualization:**

The following image provides a more detailed look at the real-world maze setup. This physical representation mirrors the virtual maze environment.

![Real life Maze Build (Image created by author)](./images/final_test/final_maze_build.jpeg "Real life Maze Build (Image created by author)"){ width=50% }

- **Web Application Interface:**

This web application serves as a control interface for the RC car, allowing me to easily monitor what the RC car sees (due to the sensor values being displayed) and emergency stop the car if needed.

![Web App (Image created by author)](./images/thesis/web_app.png "Web App (Image created by author)"){ width=100% }

- **Simulation Test Video:**
Watch the Double Deep Q-Network (DDQN) in action in this test video. It gives a real sense of how the algorithm navigates through the maze. Found in `chapter 23: Video References`.

- **DDQN Simulation test:** See DDQN Simulation in the Video reference section. -->

## Evaluation Metrics Overview
<!-- TODO: Replace with more natural sounding language -->
### Simulation Metrics

#### Episodic Performance

- **Objective and Goal:** The aim of this metric is to monitor the agent’s progress in mastering the maze. By evaluating the learning curve, I see how efficiently the agent can navigate to the end of the maze over successive trials. This gives us insights into its ability to optimize strategies and adapt over time.
- **How it’s Assessed:** I measure the number of episodes the agent needs before it can consistently complete the maze. A reduction in episodes over time is a good indicator that the agent is learning and adapting well.
- **Analytical Techniques:** To examine episodic performance, we either conduct statistical analyses or create visual plots, such as learning curves. These tools help us track and visualize changes in performance throughout the training period.
- **Accuracy and Consistency Measures:** To maintain accuracy and consistency, I ensure data integrity and control experimental conditions. Averaging results across multiple trials helps smooth out any randomness in the learning process, providing a clearer picture of the agent’s performance.

#### Step Efficiency

- **Objective and Goal:** This metric evaluates the agent's decision-making efficiency and ability to optimize its path through the maze. By measuring the steps the agent takes to solve the maze, fewer steps indicate a more efficient and smarter learning process.
- **How it's Assessed:** I keep track of the steps required to reach the maze's endpoint in each episode and analyze the reduction in steps over time.
- **Analytical Techniques:** I use quantitative analysis to examine trends in step count. Smoothing techniques may be applied to provide a clearer view of the overarching trends amidst episode-to-episode variability.
- **Accuracy and Consistency Measures:** To ensure reliable metrics, I replicate tests and average results, maintaining the same maze configuration for all experiments.

#### MSE Loss Measurement

$$
MSE(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N-1} (y_i - \hat{y}_i)^2
$$

where: \newline
  \newline $y_i$ represents the actual value.
  \newline $\hat{y}_i$ represents the predicted value.
  \newline $N$ is the total number of observations.

- **Objective and Goal:** This metric quantifies the accuracy of the agent's predictions by measuring the squared discrepancies between predicted values and actual outcomes, providing a clear gauge of learning precision.
- **How it's Assessed:** Using the provided mathematical formula, I average the squared differences across all predictions for an episode or series of episodes.
- **Analytical Techniques:** Calculating MSE is straightforward, but understanding its trend requires examining how it correlates with different stages of the agent’s learning, such as initial acquisition of knowledge versus later strategy refinement.
- **Accuracy and Consistency Measures:** Regularly evaluating against a validation set or maintaining a consistent testing framework ensures reliable insights into the agent's predictive accuracy and learning trajectory.

#### Reward Trend Analysis

- **Objective and Goal:** This analysis helps determine how effectively the agent's actions lead to positive outcomes, which are indicative of its learning and strategy development.
- **How it's Assessed:** By tracking and analyzing the rewards the agent accumulates over time, looking for trends that show an increase in reward collection.
- **Analytical Techniques:** Employing time series analysis or plotting cumulative rewards can vividly illustrate improvements in the agent's decision-making and learning.
- **Accuracy and Consistency Measures:** Averaging trends over several runs and keeping the reward structures consistent throughout the experiments to ensure comparability.

#### Epsilon Decay Tracking

- **Objective and Goal:** This metric monitors how well the agent balances exploration of new paths with exploitation of known successful strategies, key for adapting learning methods effectively.
- **How it's Assessed:** By observing the decline in the epsilon parameter over episodes, which indicates the agent’s shift from exploring to exploiting.
- **Analytical Techniques:** Plotting epsilon values across episodes helps visualize how the agent’s learning strategy evolves over time.
- **Accuracy and Consistency Measures:** Applying the epsilon decay strategy uniformly across all training sessions and maintaining consistent experimental conditions to ensure comparability of results.

### Real-World Metrics

Transitioning to real-world application involved assessing how well the strategies developed in simulation held up when the agent faced a physical maze with real obstacles and constraints.

- **Maze Navigation**: Observing the RC car as it maneuvered through a real-world maze served as direct proof of how effectively the training translated from simulation to reality. This hands-on test demonstrated the practical utility of the trained agent in navigating complex paths.
- **Sensor Data Analysis**: By examining the real-time sensor data during navigation trials, I gained a deeper insight into how the agent interacts with its physical environment. This analysis was crucial for evaluating the agent’s ability to avoid obstacles and optimize its pathfinding strategies efficiently.

# Background on Reinforcement Learning Algorithms

## Background on Double Deep Q-Network (DDQN)

The Double Deep Q-Network (DDQN) is an enhancement of the Deep Q-Network (DQN), a pivotal algorithm in the field of deep reinforcement learning that integrates deep neural networks with Q-learning. DQN itself was a significant advancement as it demonstrated the capability to approximate the Q-value function, which represents the expected reward for taking an action in a given state, using high-capacity neural networks.

### Evolution from DQN to DDQN

**DQN Challenges**: While DQN substantially improved the stability and performance of Q-learning, it was susceptible to significant overestimations of Q-values due to the noise inherent in the approximation of complex functions by deep neural networks. This overestimation could lead to suboptimal policies and slower convergence during training.

**DDQN Solution**: Introduced by Hado van Hasselt et al., DDQN addresses the overestimation problem of DQN by decoupling the action selection from the target Q-value generation—a technique termed "double learning." In traditional DQN, a single neural network is used both to select the best action and to evaluate its value. DDQN modifies this by employing two networks:

- The **current network** determines the action with the highest Q-value for the current state.
- A separate **target network**, which is a delayed copy of the current network, is used to estimate the Q-value of taking that action at the next state [22].

### The Decoupling Effect

This separation ensures that the selection of the best action is less likely to overestimate Q-values, as the estimation is made using a different set of weights, thus reducing bias in the learning process. The target network's parameters are updated less frequently (often after a set number of steps), which further enhances the algorithm's stability.

### Impact and Applications

DDQN has been shown to achieve better performance and faster convergence in complex environments compared to DQN. It is particularly effective in scenarios where precise action evaluation is crucial, such as in video games and robotic navigation tasks. The improved reliability and accuracy of DDQN make it a valuable model for studying reinforcement learning in controlled environments where stability and efficiency are critical.

## Background on Deep Q-Network (DQN)

The Deep Q-Network (DQN) algorithm represents a significant breakthrough in reinforcement learning by combining traditional Q-learning with deep neural networks. This approach was popularized by researchers at DeepMind with their notable success in training agents that could perform at human levels across various Atari games [23].

**Core Mechanism**: DQN uses a deep neural network to approximate the Q-value function, which is the expected reward obtainable after taking an action in a given state and following a certain policy thereafter. The neural network inputs the state of the environment and outputs Q-values for each possible action, guiding the agent's decisions.

**Innovations Introduced**:

- **Experience Replay**: DQN utilizes a technique called experience replay, where experiences collected during training are stored in a replay buffer. This allows the network to learn from past experiences, reducing the correlations between sequential observations and smoothing over changes in the data distribution.
- **Fixed Q-Targets**: To further stabilize training, DQN employs a separate target network, whose weights are fixed for a number of steps and only periodically updated with the weights from the training network [23].

### DQN Advantages and Applications

DQN's ability to handle high-dimensional sensory inputs directly with minimal domain knowledge makes it highly versatile and effective in complex environments such as video games, where it can learn directly from pixels.

## Background on Q-agent (Q-learning)

Q-agent, based on the Q-learning algorithm, is one of the most fundamental types of reinforcement learning methods. It is a model-free algorithm that learns to estimate the values of actions at each state without requiring a model of the environment [24].

**Simplicity and Versatility**: Q-learning works by updating an action-value lookup table called the Q-table, which stores Q-values for each state-action pair. These values are updated using the Bellman equation during each step of training based on the reward received and the maximum predicted reward for the next state.

**Challenges**: While simple and effective for smaller state spaces, Q-learning's reliance on a Q-table becomes impractical in environments with large or continuous state spaces, where the table size would become infeasibly large.

### Q-learning Applications

Q-learning has been foundational in teaching agents in environments with discrete, limited state spaces, such as simple mazes or decision-making scenarios with clear, defined states and actions.

## Background on Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a policy gradient method for reinforcement learning that simplifies and improves upon the Trust Region Policy Optimization (TRPO) approach. PPO has become popular due to its effectiveness and ease of use [25].

**Optimization Technique**: PPO seeks to take the largest possible improvement step on a policy while avoiding

too large updates that might lead to performance collapse. It achieves this through an objective function that includes a clipped term, penalizing changes to the policy that move it too far from the previous policy.

**Advantages**: PPO is robust to a variety of hyperparameters and can be used in both continuous and discrete action spaces. It has shown great success in environments ranging from simulated robotics to complex game environments.

### PPO Applications

PPO is favored in many modern RL applications due to its balance between efficiency, ease of implementation, and strong empirical performance.

## Background on Actor-Critic (AC)

Actor-Critic methods form a broad class of algorithms in reinforcement learning that combine both policy-based (actor) and value-based (critic) approaches [26].

**Dual Components**:

- **Actor**: Responsible for selecting actions based on a policy.
- **Critic**: Estimates the value function (or Q-value), which is used to evaluate how good the action taken by the actor is.

**Advantages**: By separating the action selection and evaluation, actor-critic methods can be more efficient than conventional policy-gradient methods. They reduce the variance of the updates and typically converge faster.

### Actor-Critic Applications

Actor-Critic algorithms are versatile and can be applied to both discrete and continuous action spaces. They have been effectively used in applications that require balancing exploration of the environment with the exploitation of known rewards, such as in robotics and complex game environments.

# Comparative Analysis of Reinforcement Learning Algorithms in Maze Navigation

In this analysis, I compare various reinforcement learning algorithms, namely Double Deep Q-Network (DDQN), Deep Q-Network (DQN), Q-agent, Actor-Critic (AC), and Proximal Policy Optimization (PPO). This comparison is based on their performance in navigating a complex maze, focusing on efficiency, learning rate, and adaptability.

## Performance Metrics and Visualization

### 1. Visit Heatmaps

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/visit_heatmap_DDQN.png}
\captionof{figure}{DDQN Heatmap (Image created by author)}
\end{minipage}%
\hspace{.05\textwidth}
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/visit_heatmap_DQN.png}
\captionof{figure}{DQN Heatmap (Image created by author)}
\end{minipage}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/visit_heatmap_PPO.png}
\captionof{figure}{PPO Heatmap (Image created by author)}
\end{minipage}%
\hspace{.05\textwidth}
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth * 1.2]{./images/training_images/visit_heatmap_Q-agent.png}
\captionof{figure}{Q-agent Heatmap (Image created by author)}
\end{minipage}
\end{figure}

**Commentary**: The visit heatmaps provide a visual representation of the exploration patterns for different algorithms. The heatmap for DDQN shows a concentrated path, indicating the agent's ability to efficiently learn and focus on the optimal routes through the maze. Similarly, DQN exhibits focused exploration but with slightly more dispersion compared to DDQN, suggesting a robust learning process. On the other hand, PPO and Q-agent demonstrate widespread exploration across the maze, which indicates less efficient learning and decision-making. These dispersed patterns reflect the algorithms' struggle to consistently identify and follow optimal paths, resulting in suboptimal navigation strategies.

### 2. Maze Solution Efficiency

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/maze_solution_DDQN.png}
\captionof{figure}{DDQN Maze Path (Image created by author)}
\end{minipage}%
\hfill
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/maze_solution_DQN.png}
\captionof{figure}{DQN Maze Path (Image created by author)}
\end{minipage}%
\hfill
\end{figure}

\begin{figure}[htbp]
\begin{minipage}{.45\textwidth}
\includegraphics[width=\linewidth]{./images/training_images/maze_solution_Q-agent.png}
\captionof{figure}{Q-agent Maze Path (Image created by author)}
\end{minipage}
\end{figure}

**Commentary**: The maze solution paths highlight the efficiency of each algorithm in navigating the maze. DDQN demonstrates the shortest and most direct path, indicating superior learning and optimization in its decision-making process. DQN also performs well, albeit with a slightly longer path, reflecting its effective yet slightly less optimal strategy. The Q-agent, while successful in completing the maze, takes a more winding route, suggesting a less efficient approach and a greater number of steps to reach the goal. This comparison underscores the superior efficiency of DDQN in solving the maze with minimal steps, followed closely by DQN, and identifies areas where Q-agent can improve its path optimization.

For the PPO and AC algorithms, their paths were more complex and less direct, indicating a need for further optimization and learning to enhance their maze navigation efficiency (not shown in the figures).

### 3. Reward History and Distribution

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/reward_history_DDQN.png}
\captionof{figure}{DDQN Reward History (Image created by author)}
\end{minipage}%
\hfill
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/reward_history_DQN.png}
\captionof{figure}{DQN Reward History (Image created by author)}
\end{minipage}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/reward_history_AC.png}
\captionof{figure}{AC Reward History (Image created by author)}
\end{minipage}%
\hfill
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/reward_history_PPO.png}
\captionof{figure}{PPO Reward History (Image created by author)}
\end{minipage}
\end{figure}

\begin{figure}[htbp]
\begin{minipage}{.45\textwidth}
\includegraphics[width=\linewidth]{./images/training_images/reward_history_Qlearning.png}
\captionof{figure}{Q-agent Reward History (Image created by author)}
\end{minipage}
\end{figure}

**Commentary**: The reward history graphs offer insights into the learning stability and efficiency of each algorithm. DDQN and DQN display steady and consistent increases in reward accumulation, indicating stable and reliable learning processes. In contrast, AC and PPO exhibit significant fluctuations in reward history, reflecting instability and inconsistency in their learning curves. This variability suggests these algorithms encounter challenges in maintaining a steady learning trajectory. Q-agent, although stable, shows a slower rate of reward accumulation, indicating a more gradual and less efficient learning process compared to DDQN and DQN. Overall, DDQN and DQN stand out for their consistent and effective reward acquisition, while AC, PPO, and Q-agent highlight areas for potential improvement in learning stability and efficiency.

### 4. Mean Squared Error (MSE) Over Time

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/mse_history_sampled_DDQN.png}
\captionof{figure}{DDQN MSE (Image created by author)}
\end{minipage}%
\hfill
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/mse_history_sampled_DQN.png}
\captionof{figure}{DQN MSE (Image created by author)}
\end{minipage}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/mse_history_AC.png}
\captionof{figure}{AC MSE (Image created by author)}
\end{minipage}%
\hfill
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/mse_history_PPO.png}
\captionof{figure}{PPO Loss (Image created by author)}
\end{minipage}
\end{figure}

**Commentary**: The MSE graphs track the learning accuracy and error management of each algorithm over time. DDQN achieves the lowest and most stable MSE values, signifying its effective learning and strong capability in minimizing prediction errors. DQN also performs well with relatively stable MSE, though slightly higher than DDQN, indicating competent but less optimal error reduction. In contrast, AC and PPO show higher and more variable MSE values, pointing to less effective learning and greater difficulties in managing prediction errors. These fluctuations suggest these algorithms struggle to consistently reduce errors, impacting their overall performance. The comparison highlights DDQN's superior accuracy and error management, with DQN as a strong contender, while AC and PPO require further optimization to enhance their learning precision.

### 5. Moving Average of Rewards

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/steps_per_episode_with_moving_avg_DDQN.png}
\captionof{figure}{DDQN Moving Average (Image created by author)}
\end{minipage}%
\hfill
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/steps_per_episode_with_moving_avg_DQN.png}
\captionof{figure}{DQN Moving Average (Image created by author)}
\end{minipage}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/reward_per_episode_with_moving_avg_AC.png}
\captionof{figure}{AC Moving Average (Image created by author)}
\end{minipage}%
\hfill
\begin{minipage}{.45\textwidth}
\centering
\includegraphics[width=\linewidth]{./images/training_images/reward_per_episode_with_moving_avg_PPO.png}
\captionof{figure}{PPO Moving Average (Image created by author)}
\end{minipage}
\end{figure}

**Commentary**: The moving average of rewards provides a clear view of the long-term performance trends of each algorithm. DDQN and DQN show consistent and progressive improvement in reward accumulation, reflecting their effective learning and adaptation over time. This steady upward trend indicates robust performance enhancement. In contrast, AC and PPO display significant variability in their moving averages, suggesting inconsistent learning and slower performance improvements. The Q-agent, while showing a stable improvement trend, progresses at a slower pace compared to DDQN and DQN, highlighting a need for more efficient learning strategies. Overall, the moving averages affirm DDQN and DQN's effectiveness in continuously enhancing performance, while AC, PPO, and Q-agent demonstrate areas where more consistent and rapid improvements are needed.

## Conclusion and Insights

This comprehensive analysis reveals distinct performance characteristics and efficiencies of various reinforcement learning algorithms in maze navigation. DDQN stands out for its balanced approach, efficiently solving the maze with the fewest steps while demonstrating superior stability and effective error management. DQN, though slightly less efficient in navigation, showcases robust learning stability, making it a reliable choice. Q-agent, despite its simpler approach, competes closely with DDQN in terms of steps required to solve the maze but struggles during the initial learning phases.

AC and PPO exhibit higher fluctuations in their performance metrics, indicating the need for further optimization to achieve better consistency and efficiency. These algorithms show potential but require more refinement to handle the complexities of maze navigation effectively.

Ultimately, this analysis aids in selecting the most suitable reinforcement learning algorithm based on specific task requirements and environmental complexities. It enhances our understanding of their practical applications and helps in optimizing learning outcomes for real-world scenarios.

# Implementation of Real-World Control Algorithms

## Introduction to Real-World Implementation

In this section, I delve into the practical application of control algorithms developed through simulations, now being adapted to control a physical robot. This transition is pivotal for evaluating how simulated behaviors translate into real-world scenarios, thereby assessing the effectiveness and limitations of sim-to-real transfer.

## System Overview

The experimental setup uses an ESP32 microcontroller combined with MPU6050 gyroscopic sensors and ultrasonic sensors for distance measurement, connected to a motor control system. These components enable the robot to perform maneuvers like moving forward, turning left, and turning right. The system's architecture is designed to replicate the simulated environment's dynamics.

## Code Architecture and Integration

**System Initialization**

Understanding the system's initial setup is crucial for ensuring robust and reliable operation. This phase involves preparing the robot by configuring its hardware interfaces, establishing network connectivity, and setting up sensors and actuators.

- **WiFi and OTA Configuration**: This sets up a network connection and facilitates Over-The-Air (OTA) updates, crucial for remote debugging and iterative improvements.
- **Sensor and Display Setup**: Activates ultrasonic sensors for distance monitoring and initializes a display to provide real-time feedback on the robot's status and IP address, enhancing user interaction and debugging capabilities.
- **MPU6050 Setup and Calibration**: Calibrates the gyroscopic sensor for accurate angle measurements, essential for precise navigation.
- **Motor Setup**: Configures motor drivers and establishes initial motor states, prepping the robot for subsequent movement commands.

**Motor Control Mechanism**

This subsection elaborates on how movement functions are implemented, translating simulated navigation algorithms into the real-world robotic system.

- **Variables for Motor Control**

  ```cpp
  int initialSpeed = 125; // Higher initial speed for robust movement
  int minSpeed = 40;      // Minimum speed to maintain control
  int speed = initialSpeed;
  constexpr int TURN_DURATION = 245;
  ```

  These variables dictate the motors' initial and minimum speeds, and the duration for turning, facilitating precise and controlled movements by adjusting the speed dynamically based on the robot's turning angle.

- **Forward Movement**

  The `move_forward` function initiates rapid forward motion, with real-time checks for obstacles to ensure safe stops—mimicking the real-world need for dynamic responsiveness.

- **Left Turn**

  The `move_left` function adjusts motor speeds dynamically, a strategy refined in simulations to accommodate physical and inertia effects during turns, ensuring smooth and controlled navigation.

- **Right Turn**

  The `move_right` function applies similar adjustments and sensor feedback to execute precise right turns. Incorporating `calibrateSensors()` before each movement guarantees accurate gyroscopic data, vital for the precise execution of turns.

- **Stopping Movement**

  The `stop_moving` function is designed to immediately halt all motions, crucial for accident prevention and adaptation to sudden changes in dynamic environments.

**Calibration and Sensor Data Interpretation**

Calibration is crucial for ensuring sensor accuracy and reliability, maintaining the integrity of behaviors developed in simulations when applied in real-world settings. The `calibrateSensors` function periodically recalibrates the gyroscopic sensors to correct any data drift or inaccuracies.

```cpp
void calibrateSensors()
{
    long gyroZAccum = 0;
    Serial.println("Calibrating...");
    for (int i = 0; i < 100; i++)
    {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        gyroZAccum += gz;
        delay(20);
    }
    mpu.setZGyroOffset(-gyroZAccum / 13100); // Calibration based on 100 readings
    Serial.println("Calibration Complete");
}
```

<!-- TODO : Vooral nog wat dieper ingaan op het feit dat jouw sensoren niet voldoende data opleveren om de exacte positie te bepalen binnen de real-world grid. Je doet dat dan wel binnen jouw conclusies, maar hier mag je er ook al wat dieper op ingaan -->
<!-- rewrite this seciont, too many subtitles and too little information to make it a valuable additions -->
## Real-World Application and Limitations

Transitioning from simulated environments to real-world applications introduces unique challenges, particularly in interpreting sensor data and replicating vehicle movements. This section addresses these critical aspects, highlighting both the potential benefits and limitations of applying insights from simulations to actual autonomous vehicle (AV) operations.

### Enhanced Sensor-Based Navigation

Refining sensor-based navigation technologies through simulations can significantly enhance the functionality of autonomous vehicles. These technologies are crucial in real-world applications that demand high precision and adaptability. For instance, in busy urban areas or automated delivery systems, dynamic and accurate navigation is essential for boosting both safety and efficiency. By integrating insights from simulations, sensor-based navigation systems can be better tuned to handle the complex and ever-changing conditions of the real world.

### Informing Autonomous Vehicle Movement

Simulated environments offer controlled settings to study vehicle dynamics and movement responses. Applying these insights to autonomous vehicle development can lead to advanced algorithms capable of managing the unpredictable nature of real-world environments. This knowledge is essential for improving the ability of autonomous systems to safely and efficiently navigate through dynamic and often chaotic traffic conditions, thereby enhancing the overall functionality of autonomous transportation.

### Limitations

**Discrepancies in Sensor Data Interpretation:**
A major challenge in real-world application is the variation in sensor data accuracy between simulated and actual environments. These discrepancies can directly impact the effectiveness of navigational algorithms, potentially compromising the vehicle's decision-making processes, safety, and operational efficiency.

**Insufficient Data for Exact Positioning:**
One critical limitation encountered was the inability of sensors to provide precise data for determining the car’s exact position within the real-world grid. In simulations, sensors provided precise and reliable data, enabling accurate positioning and movement control. However, in real-world conditions, sensor data was often noisy, incomplete, and affected by various environmental factors.

- **Limited Sensor Resolution:** Sensors like ultrasonic distance sensors had limited resolution and range, impacting their ability to detect fine details necessary for precise positioning.
- **Environmental Noise:** Real-world environments introduced noise from various sources, such as electronic devices, surface irregularities, and ambient conditions, which interfered with sensor readings.
- **Dynamic Obstacles:** Unlike controlled simulations, real-world environments had dynamic obstacles that sensors could not always accurately detect or interpret, leading to positioning errors.

These limitations meant that the vehicle often had a rough estimate of its position rather than a precise one, hindering its ability to navigate accurately within a defined grid. Practically, the vehicle primarily relied on distance measurements to avoid crashes and verify its movement capabilities. However, it lacked the ability to determine whether it was centered within the path, which was a significant drawback in maze navigation.

### Challenges in Movement Replication

Replicating vehicle movements from simulations in the real world is challenging due to various factors such as different road surfaces, weather conditions, vehicle weight, and mechanical issues. These factors can cause unexpected changes in vehicle behavior. Therefore, algorithms developed in simulations need to be tweaked and recalibrated to work effectively in real-world conditions. For instance, the vehicle's weight can affect acceleration and deceleration, impacting the timing of movement commands. Similarly, the surface the vehicle moves on can affect traction, influencing its ability to turn or stop.

### Practical Implementation Considerations

To successfully apply simulation insights to the real world, several practical aspects need to be addressed:

- **Sensor Calibration:** Adjust sensors to account for environmental influences.
- **Algorithm Adjustment:** Modify algorithms to fit hardware limitations.
- **Handling Unpredictability:** Ensure the system can cope with real-world unpredictability.

By addressing these considerations, we can bridge the gap between simulation-based research and real-world applications, enhancing the safety, efficiency, and reliability of autonomous vehicle technologies.

## Conclusion for Real-World Application

Transitioning from simulation-based research to real-world applications in autonomous vehicle navigation presents both challenges and opportunities. While using simulation insights for sensor utilization and vehicle movement can revolutionize autonomous vehicle technologies, there is a significant gap between simulation precision and real-world variability that needs to be bridged.

A significant limitation was the imprecision in sensor data for positioning. Sensors prevented crashes by measuring the distance to obstacles so it is able to decide in which direction it should move. However, this did not try to measure the exact position of the car within the grid.

Additionally, real-world factors such as surface variations, environmental conditions, and dynamic obstacles require continuous adjustments and recalibration of algorithms. Practical considerations, including sensor calibration and algorithm adaptation, are essential for effectively deploying autonomous systems.

Ultimately, overcoming these challenges is key to successfully integrating sim2real technologies. Doing so will improve the safety, efficiency, and reliability of autonomous transportation systems, paving the way for broader adoption and use in various real-world scenarios.

# Challenges and Solutions in RL Implementation

## Challenge 1: Choosing the Right Virtual Environment

- **Description**: Picking the best virtual environment for training the RC car.
- **Solution**: I chose **OpenAI Gym** because it’s simple, familiar from previous coursework, and focuses on reinforcement learning.

## Challenge 2: Selecting the Optimal Reinforcement Learning Technique

- **Description**: Finding the most effective RL technique for training the virtual RC car.
- **Solution**: After testing various methods, the Double Deep Q-Network (DDQN) proved to be the best, consistently solving the maze with fewer steps and episodes than other techniques.

## Challenge 3: Addressing Movement Discrepancies in Sim2Real Transfer

- **Description**: Bridging the gap between how the RC car moves in simulations and in the real world.
- **Solution Attempt**: I fine-tuned the action command frequency using an async method, waited for the motor to finish moving, and considered a queued action system. Getting precise movement in the real world turned out to be more critical than in simulations.

## Challenge 4: Alignment Issues and Motor Encoder Implementation

- **Description**: Ensuring the RC car moves in a straight line, as there was a persistent ~3-degree offset.
- **Solution Attempts**:
  - **Attempt 1**: Used motor encoders to improve accuracy but faced precision limits.
  - **Attempt 2**: Switched to a more powerful motor, but the added weight brought back the alignment issue.
  - **Attempt 3**: Added an MPU6050 gyroscope to measure and adjust orientation, which initially helped with 90-degree turns but didn't fix the offset.
  - **Attempt 4**: Removed the Raspberry Pi and used only the ESP32 for all controls, resulting in a lighter, more precise robot, though it still struggled with consistent 90-degree turns.

## Challenge 5: Ensuring Consistent and Effective Training

- **Description**: Maximizing training efficiency and performance while keeping things consistent between simulations and real-world scenarios.
- **Solution**: Training in a simulation was much more efficient due to the difficulties of resetting the RC car, dealing with manual interferences, and limited battery life.

## Challenge 6: Accurate Sensor Data Normalization for Sim2Real Transfer

- **Description**: Aligning sensor data between simulated and real-world environments for accurate model performance.
- **Solution**: Implemented functions to ensure real-world sensor data matched the training data.
  - **Real-World Sensor Data Normalization**:
    $$
    \text{map\_distance}(d) = \begin{cases}
    d & \text{if } d < 25 \\
    25 + (d - 25) \times 0.5 & \text{otherwise}
    \end{cases}
    $$
  - **Simulation Sensor Data Normalization**:
    $$
    \text{normalize\_distance}(d) = \text{max}\left(0, \text{min}\left(\frac{d}{\text{sensor\_max\_range}}, 1\right)\right) \times 1000
    $$

## Challenge 7: Integration of Failsafe Mechanisms

- **Description**: Preventing collisions and ensuring safe navigation in the real world.
- **Solution**: Developed a failsafe system to prevent unwanted forward movement and retrained the model with this feature, which solved the issue of the robot driving into walls and getting stuck.

## Challenge 8: Training Environment and Technique Efficacy

- **Description**: Finding the most effective environment and RL technique for training.
- **Solution**: The DDQN technique was more efficient than DQN, Q-agent, PPO, and ActorCritic approaches, highlighting the importance of selecting the right technique.

## Conclusion for Challenges and Solutions

This section outlines the practical challenges encountered while applying reinforcement learning (RL) techniques to autonomous RC cars. My journey began with selecting OpenAI Gym as the virtual environment due to its simplicity and relevance to RL. The Double Deep Q-Network (DDQN) emerged as the most effective RL technique for navigating complex environments.

However like discussed before , transitioning from simulations to real-world applications revealed significant discrepancies, particularly in movement control and sensor data alignment. I explored various solutions like motor encoders, power adjustments, and gyroscope integration, which partially addressed these issues. Efforts to normalize sensor data and implement failsafe mechanisms also contributed to better alignment with real-world conditions.

A significant advancement was achieved by simplifying the robot's design to use only the ESP32 module, making it lighter and more precise. This change marked a considerable step in overcoming the previous challenges.

Although I made a lot of progress in addressing these challenges, there is still room for improvement in achieving seamless sim-to-real transfer and ensuring consistent performance across different environments.

# Integration of Practical Experiments

Throughout my research, I used various practical experiments to solve the challenges I encountered. These experiments, documented through video demonstrations, provide clear insights into my problem-solving process.

## Addressing Alignment and Orientation Challenges

One of the main challenges was ensuring the RC-car's precise orientation and alignment during movement. To address this, I used the MPU6050 gyroscope to correct alignment issues and achieve accurate 90-degree turns. My efforts focused on using the gyroscope to maintain and correct the car's orientation, crucial for navigating complex mazes with high precision.

### Using the MPU6050 Gyroscope for Precise Orientation

- **Experiment E1 - Gyroscope Calibration**: Testing the MPU6050 gyroscope to correct the car's orientation for accurate navigation, aiming to improve control over the vehicle's movement through maze environments (see Video E1 in the Video References section).
- **Experiment E2 - Navigational Corrections**: Addressing alignment issues for precise 90-degree turns and realigning the car's forward movement to fix a persistent ~3-degree offset (see Video E2 in the Video References section).

## Improving Movement Precision with Encoders

To enhance the RC-car's movement precision, I experimented with rotary encoders. These devices, which accurately measure wheel rotations, were essential for improving straight-line movements and addressing hardware reliability challenges in real-world applications.

- **Experiment E6 - Encoder Implementation**: Adding rotary encoders to the setup to gain more precise control over the car's movements by accurately measuring wheel rotations, thus refining the vehicle's navigation capabilities (see Video E6 in the Video References section).
- **Experiment E7 - Troubleshooting Encoder Malfunction**: Addressing a malfunction with one of the encoders that halted further tests, highlighting the practical challenges of maintaining hardware reliability (see Video E7 in the Video References section).

## Real-World Application Tests

Moving beyond controlled environments, I tested the RC-car in both outdoor and indoor settings to evaluate its performance in real-world conditions. These tests were crucial for assessing the practical application of my research findings and understanding the challenge of accurately translating simulation models to real-world applications.

- **Experiment E9 - Outdoor Navigation Test**: Navigating the RC-car on uneven outdoor surfaces, where variations greatly affected performance, underscoring the importance of environmental factors in autonomous navigation (see Video E9 in the Video References section).
- **Experiment E11 - Indoor Controlled Test**: Conducting controlled indoor tests to closely monitor and adjust the RC-car's navigation strategies, reflecting on the complexities of sim-to-real transfer (see Video E11 in the Video References section).

<!-- \pagebreak -->

<!-- --
  # TODO: Interviews with Sam and Wouter for feedback (have not done these interviews yet)
  • Wat zijn volgens hen de sterke en zwakke punten van het resultaat uit jouw researchproject?   
  • Is ‘het projectresultaat’ (incl. methodiek) bruikbaar in de bedrijfswereld?  
  • Welke alternatieven/suggesties geven bedrijven en/of community?   
  • Wat zijn de mogelijke implementatiehindernissen voor een bedrijf?    
  • Wat is de meerwaarde voor het bedrijf?   
  • Is er een maatschappelijke/economische/socio-economische meerwaarde aanwezig?  

  Rewrite this yourself
-- -->
<!-- 

# Reflection

Reflecting on this research project, I’ve gained essential insights that have profoundly influenced my approach and perspective.

## Openness to New Ideas

One of the key lessons learned is the value of openness. Embracing new ideas and venturing beyond familiar territory was crucial for fostering creativity and discovering innovative solutions. Moving forward, I am committed to maintaining this exploratory spirit, keen to harness the winds of novelty.

## Bridging Theory and Practice

Bridging the gap between theory and practice proved challenging. While the virtual environments were controlled, the real world demanded adaptability to unforeseen complexities. This experience has sharpened my ability to navigate between the elegance of theory and the unpredictability of real-world applications.

## Anticipatory Thinking

Overcoming barriers taught me the importance of anticipatory thinking. From navigating corporate hurdles to adapting to evolving safety standards, looking ahead has become an integral part of my approach, transforming potential obstacles into opportunities for innovation.

## Policy and Regulation

Engaging with policy and regulatory frameworks highlighted the delicate balance between innovation and legislation. Working with policymakers and industry leaders, I learned the importance of crafting regulations that foster innovation while ensuring public safety and accountability.

## Societal Implications

The societal implications of autonomous systems have been profound. From enhancing mobility for the disabled to influencing urban planning, the potential for positive impact is immense. I am more aware of the need to ensure these technologies are accessible and beneficial across societal divides.

## Feedback from the Jury

The feedback from the jury was predominantly positive, expressing admiration for the depth of my research. They highlighted several areas where my work excelled and provided constructive suggestions for improvement.

1. **Depth of Research**: The jury appreciated the comprehensive nature of my research, particularly the detailed exploration of reinforcement learning (RL) and its application in maze navigation. They noted that my work demonstrated a thorough understanding of the theoretical aspects and practical challenges involved in sim-to-real transfer.

2. **Innovative Solutions**: The jury commended my innovative approach to overcoming the challenges of sim-to-real transfer. They were impressed with my use of the Double Deep Q-Network (DDQN) architecture and the development of a custom maze navigation environment. These innovations were seen as significant contributions to the field of AI and robotics.

3. **Practical Implementation**: The jury valued the practical aspects of my project, particularly the real-world testing of the RC car. They recognized the challenges associated with transitioning from simulation to reality and praised my efforts to address these challenges through rigorous testing and calibration.

4. **Suggestions for Improvement**: Hans Ameel brought up some potential solutions for the challenges I encountered, such as increasing the distance to the walls to reduce the impact of deviations. This suggestion was based on the observation that the car's proximity to the walls could lead to errors in navigation due to sensor inaccuracies or physical obstructions.

5. **Use of Camera**: Another suggestion was using a camera to monitor the position of the RC car within the maze. By integrating a camera system, I could obtain real-time visual feedback on the car's location and movements. This would help in fine-tuning the navigation algorithms and ensuring more accurate and reliable performance in real-world scenarios.

6. **Correction on Sensor Use**: Additionally, a correction was made to my presentation: while I had indicated using an accelerometer to measure the rotation of my RC car, I actually used the gyroscope that is part of the MPU6050, which also contains an accelerometer and magnetometer. This correction was essential for accurately conveying the technical aspects of my project and ensuring clarity in the presentation of my methodologies.

## Methodological Insights

## Virtual Environment and Sim-to-Real Transfer

Developing the custom maze navigation environment (RCMazeEnv) and utilizing the DDQN architecture provided significant insights into the complexities of sim-to-real transfer. The virtual environment allowed for controlled experimentation and fine-tuning of the RL algorithms, while real-world testing highlighted the practical challenges of applying these algorithms outside the simulation.

## Real-World Testing and Calibration

The transition from simulation to real-world application involved extensive calibration and testing. Ensuring the RC car's sensors were accurately aligned with their virtual counterparts was critical. The iterative process of testing, calibration, and adjustment helped to refine the system and improve its performance in navigating the physical maze.

## Reward Function and Agent Performance

Designing an effective reward function was crucial for guiding the agent's learning process. The balance between immediate rewards and penalties for inefficient actions helped to optimize the agent's performance. The inclusion of components such as goal achievement bonuses, proximity rewards, progress rewards, exploration penalties, and efficiency penalties provided a comprehensive framework for evaluating and improving the agent's navigation strategies.

## Societal Implications of Autonomous Systems

The potential applications of autonomous systems extend far beyond maze navigation. The insights gained from this research could contribute to the development of autonomous vehicles, drones, and other robotic systems that navigate complex environments. Ensuring these technologies are accessible and beneficial to society requires careful consideration of ethical, regulatory, and practical factors.

## Future Directions

The project can be further developed by integrating more complex sensory systems, applying more sophisticated machine learning models, extending the testing environments to include varying conditions, and fostering community collaboration. These enhancements could not only improve the technology but also advance the field of autonomous vehicle research and development.

## Personal Growth and Future Aspirations

This research project has been a transformative experience, shaping my understanding of AI and robotics and guiding my future aspirations. The lessons learned, from the technical challenges to the broader societal implications, have equipped me with the skills and knowledge to pursue further research and innovation in this field.

## Commitment to Innovation and Societal Stewardship

As I move forward, my ethos will be defined by adaptability, responsiveness, and a commitment to societal stewardship. The lessons learned have prepared me for the next stage of my journey, where I will continue to engage with data, refine methodologies, and embrace the challenges that lie ahead.

## Conclusion

In conclusion, this research project has been an invaluable learning experience, providing deep insights into the challenges and opportunities of sim-to-real transfer in reinforcement learning. The positive feedback from the jury, coupled with the constructive suggestions for improvement, has been instrumental in shaping the direction of my work. Moving forward, I am excited to continue exploring the potential of autonomous systems and their applications, driven by a commitment to innovation and societal impact.

## Personal Note on the Project

Even though this was an incredible project to work on and I loved every second of it, I can't help but feel like I limited myself by not actively researching before starting this project. I went into this (and many other projects before this) headfirst, starting to make a simple maze in Pygame and solving it using a q-agent, even before the three-week period began. However, by doing this, I limited myself by not exploring other methods of completing this maze, like giving the agents more controls than just moving forward, left, and right. For example, as Wouter suggested during one of our evaluations, the use of a camera to see where the car is in the maze. This, along with giving the agent the ability to move the left and right motors completely independent of each other, would have given the agent a lot more control over the car and maybe even eliminated the need for the many corrections I had to make while transferring the model to the real world. 

I doubt I would've been able to get as far as fast as I did if I had taken this alternative route, but I do think it would have been a lot more interesting and maybe even more rewarding as I don't think this project is easily reproducible by others as it is now. But this also might be a good thing, as it shows that there is always room for improvement and that there is always something to learn from every project you do. In essence, I think this project was a great learning experience and I would love to do something like this again in the future, but I would like to take a different approach next time.-->

<!-- ## Reflection

Reflecting on this research project, I’ve gained valuable insights that have significantly influenced my approach and perspective.

## Openness to New Ideas

One of the most important lessons I learned is the value of being open to new ideas. Embracing unfamiliar concepts and stepping out of my comfort zone were crucial for fostering creativity and discovering innovative solutions. Moving forward, I am committed to maintaining this exploratory spirit and staying receptive to novel approaches.

## Bridging Theory and Practice

Transitioning from theory to practice proved more challenging than I expected. While controlled virtual environments were manageable, the real world presented unforeseen complexities. This experience sharpened my ability to balance theoretical knowledge with practical adaptability, preparing me to handle the unpredictability of real-world applications.

## Anticipatory Thinking

Overcoming various obstacles highlighted the importance of anticipatory thinking. Navigating challenges and adapting to evolving conditions made me realize the need to think ahead. This proactive mindset has become integral to my approach, turning potential challenges into opportunities for innovation.

## Policy and Regulation

Engaging with policy and regulatory frameworks was enlightening. It highlighted the delicate balance between fostering innovation and ensuring public safety. Collaborating with policymakers and industry leaders taught me the importance of creating regulations that encourage technological advancement while maintaining accountability.

## Feedback from the Jury

The feedback from the jury was predominantly positive, and their insights were incredibly valuable. They commended some aspects of my work and provided constructive suggestions for improvement.

1. **Practical Implementation**: The jury valued the practical aspects of my project, particularly the real-world testing of the RC car. They acknowledged the difficulties associated with transitioning from simulation to reality and commended my efforts to address these challenges through rigorous testing and calibration.

2. **Suggestions for Improvement**: Hans Ameel suggested increasing the distance to the walls to reduce the impact of deviations. This was based on the observation that the car's proximity to the walls could lead to navigation errors due to sensor inaccuracies or physical obstructions.

3. **Use of Camera**: Another recommendation was to use a camera to monitor the position of the RC car within the maze. Integrating a camera system would provide real-time visual feedback, helping to fine-tune the navigation algorithms and ensure more accurate performance in real-world scenarios.

4. **Correction on Sensor Use**: Additionally, a correction was made to my presentation: while I initially mentioned using an accelerometer to measure the car's rotation, I actually used the gyroscope part of the MPU6050, which also contains an accelerometer and magnetometer. This correction was important for accurately conveying the technical aspects of my project.

## Methodological Insights

Creating the custom maze navigation environment (RCMazeEnv) and using the DDQN architecture provided significant insights into the complexities of sim-to-real transfer. The virtual environment allowed for controlled experimentation and fine-tuning of the RL algorithms, while real-world testing highlighted the practical challenges of applying these algorithms outside the simulation.

## Real-World Testing and Calibration

Moving from simulation to real-world application required extensive calibration and testing. Ensuring the RC car's sensors were aligned with their virtual counterparts was critical. This iterative process of testing, calibration, and adjustment refined the system and improved its performance in navigating the physical maze.

## Reward Function and Agent Performance

Designing an effective reward function was key to guiding the agent's learning process. Balancing immediate rewards and penalties for inefficient actions helped optimize the agent's performance. Components like goal achievement bonuses, proximity rewards, progress rewards, exploration penalties, and efficiency penalties provided a comprehensive framework for evaluating and improving the agent's navigation strategies.

## Societal Implications of Autonomous Systems

The potential applications of autonomous systems extend far beyond maze navigation. Insights from this research could contribute to developing autonomous vehicles, drones, and other robotic systems navigating complex environments. Ensuring these technologies are accessible and beneficial to society requires careful consideration of ethical, regulatory, and practical factors.

## Future Directions

This project can be further developed by integrating more complex sensory systems, applying more sophisticated machine learning models, extending testing environments to include varying conditions, and fostering community collaboration. These enhancements could improve the technology and advance autonomous vehicle research and development.

## Personal Growth and Future Aspirations

This research project has been transformative, shaping my understanding of AI and robotics and guiding my future aspirations. The lessons learned, from technical challenges to broader societal implications, have equipped me with the skills and knowledge to pursue further research and innovation in this field.

## Commitment to Innovation and Societal Stewardship

As I move forward, my ethos will be defined by adaptability, responsiveness, and a commitment to societal stewardship. The lessons learned have prepared me for the next stage of my journey, where I will continue to engage with data, refine methodologies, and embrace future challenges.

## Personal Note

Even though this was an incredible project to work on and I loved every second of it, I can't help but feel like I limited myself by not actively researching before starting this project. I went into this (and many other projects before this) headfirst, starting to make a maze in Pygame even before the three-week period began. However, by doing this, I limited myself by not exploring other methods of completing this maze, like giving the agents more controls than just moving forward, left, and right. For example, as Wouter suggested during one of our evaluations, the use of a camera to see where the car is in the maze. This, along with giving the agent the ability to move the left and right motors completely independent of each other, would have given the agent a lot more control over the car and maybe even eliminated the need for the many corrections I had to make while transferring the model to the real world.

I doubt I would've been able to get as far as fast as I did if I had taken this alternative route, but I do think it would have been a lot more interesting and maybe even more rewarding as I don't think this project is easily reproducible by others as it is now. But this also might be a good thing, as it shows that there is always room for improvement and that there is always something to learn from every project you do. In essence, I think this project was a great learning experience and I would love to do something like this again in the future, but I would like to take a different approach next time.

\pagebreak -->

# Reflection

Reflecting on this research project, I have gained invaluable insights that have significantly influenced my approach and perspective.

## Openness to New Ideas

One of the most important lessons I learned is the value of being open to new ideas. Embracing unfamiliar concepts and stepping out of my comfort zone were crucial for fostering creativity and discovering innovative solutions. For instance, the transition from simulation to real-world testing required novel problem-solving approaches to address discrepancies in sensor data and movement control. This openness not only enriched my technical skills but also enhanced my ability to adapt to new challenges and environments.

## Bridging Theory and Practice

Transitioning from theory to practice proved more challenging than I expected. While controlled virtual environments were manageable, the real world presented unforeseen complexities. This experience sharpened my ability to balance theoretical knowledge with practical adaptability. For example, integrating sensor calibration and adjusting control algorithms were essential steps that highlighted the gap between simulated and real-world conditions. This hands-on experience has prepared me to handle the unpredictability of real-world applications more effectively. Additionally, it underscored the importance of iterative testing and continuous learning in the development process.

## Anticipatory Thinking

Overcoming various obstacles highlighted the importance of anticipatory thinking. Navigating challenges and adapting to evolving conditions made me realize the need to think ahead. This proactive mindset has become integral to my approach, turning potential challenges into opportunities for innovation. For instance, anticipating sensor noise and environmental variability led to the development of robust policies and fine-tuning strategies that improved the system's performance in real-world scenarios. This foresight is crucial for anticipating and mitigating risks, ensuring the robustness and reliability of autonomous systems.

## Policy and Regulation

Engaging with policy and regulatory frameworks was enlightening. It highlighted the delicate balance between fostering innovation and ensuring public safety. Collaborating with policymakers and industry leaders taught me the importance of creating regulations that encourage technological advancement while maintaining accountability. This experience underscored the need for ethical considerations and compliance in the deployment of autonomous systems. Understanding these regulatory landscapes will be essential as I continue to develop and implement AI and robotics solutions in the future.

## Feedback from the Jury

The feedback from the jury was predominantly positive, and their insights were incredibly valuable. They commended the practical aspects of my project, particularly the real-world testing of the RC car. Specific suggestions, such as increasing the distance to the walls to reduce navigation errors and integrating a camera system for real-time visual feedback, were implemented to enhance the accuracy and reliability of the navigation algorithms. This feedback loop was instrumental in refining my approach and improving the overall quality of the project.

## Methodological Insights

Creating the custom maze navigation environment (RCMazeEnv) and using the DDQN architecture provided significant insights into the complexities of sim-to-real transfer. The iterative process of testing, calibration, and adjustment refined the system and improved its performance in navigating the physical maze. Designing an effective reward function was key to guiding the agent’s learning process, balancing immediate rewards and penalties for inefficient actions to optimize performance. This methodological rigor will serve as a foundation for future projects, ensuring a structured and systematic approach to research and development.

## Societal Implications of Autonomous Systems

The potential applications of autonomous systems extend far beyond maze navigation. Insights from this research could contribute to developing autonomous vehicles, drones, and other robotic systems navigating complex environments. Ensuring these technologies are accessible and beneficial to society requires careful consideration of ethical, regulatory, and practical factors. For instance, autonomous systems must be designed to prioritize safety and reliability, minimizing risks to human users and the environment. Moreover, the deployment of such technologies should be accompanied by public education and engagement to build trust and understanding among stakeholders.

## Future Directions

This project can be further developed by integrating more complex sensory systems, applying more sophisticated machine learning models, extending testing environments to include varying conditions, and fostering community collaboration. These enhancements could improve the technology and advance autonomous vehicle research and development. Future research could explore the use of multi-modal sensory inputs, such as combining visual, auditory, and tactile data to enhance the agent's perception and decision-making capabilities. Additionally, collaborations with other researchers and industry partners could provide valuable insights and resources for further advancements.

## Personal Growth and Future Aspirations

This research project has been transformative, shaping my understanding of AI and robotics and guiding my future aspirations. The lessons learned, from technical challenges to broader societal implications, have equipped me with the skills and knowledge to pursue further research and innovation in this field. I am particularly interested in exploring the ethical dimensions of AI and robotics, ensuring that these technologies are developed and deployed in ways that are fair, transparent, and accountable. My future aspirations include contributing to the development of autonomous systems that can positively impact various sectors, from healthcare to transportation, enhancing efficiency and safety.

## Commitment to Innovation and Societal Stewardship

As I move forward, my ethos will be defined by adaptability, responsiveness, and a commitment to societal stewardship. The lessons learned have prepared me for the next stage of my journey, where I will continue to engage with data, refine methodologies, and embrace future challenges. I am committed to fostering a culture of innovation and continuous improvement, where creative problem-solving and interdisciplinary collaboration are encouraged. Furthermore, I will strive to ensure that my work contributes to the greater good, addressing societal challenges and promoting sustainability.

## Personal Note

Even though this was an incredible project to work on and I loved every second of it, I can’t help but feel like I limited myself by not actively researching before starting this project. I went into this (and many other projects before this) headfirst, starting to make a maze in Pygame even before the three-week period began. However, by doing this, I limited myself by not exploring other methods of completing this maze, like giving the agents more controls than just moving forward, left, and right. For example, as Wouter suggested during one of our evaluations, the use of a camera to see where the car is in the maze. This, along with giving the agent the ability to move the left and right motors completely independent of each other, would have given the agent a lot more control over the car and maybe even eliminated the need for the many corrections I had to make while transferring the model to the real world.

I doubt I would have been able to get as far as fast as I did if I had taken this alternative route, but I do think it would have been a lot more interesting and maybe even more rewarding as I don’t think this project is completely finished yet, and I look forward to continuing this journey.

# Advice

## Practical Utilization of Simulations

Simulations are invaluable in research, offering a risk-free, controllable environment for developing and refining algorithms.

- **Cost-Effectiveness:** Simulations allow for substantial cost savings by reducing the need for physical prototypes and extensive real-world trials in the early phases of research.

## Strategies for Effective Transition from Simulation to Reality

Successfully transitioning from simulations to real-world applications is critical for validating the effectiveness of research outcomes.

- **Incremental Testing:** Start with simulations to refine algorithms, then gradually introduce real-world testing to confirm results and adapt to environmental variables.
- **Feedback Loops:** Use continuous feedback mechanisms to enhance simulation models with insights gained from real-world tests, improving their accuracy and applicability.

## Overcoming Common Challenges in Simulation-to-Reality Transitions

Bridging the gap between simulations and actual conditions often requires specific adjustments, especially in terms of sensor data and mechanical operations.

- **Sensor Discrepancy Adjustments:** Regular calibration of real-world sensors is essential to ensure they align with simulation inputs.
- **Movement and Mechanics Alignment:** It's crucial to synchronize physical movements and mechanics with those anticipated by simulations to ensure smooth transitions.

## Insights from My Research

- **Simulation Platforms:** Choosing the right simulation platform, like OpenAI Gym, is critical and may necessitate additional tools for more complex scenarios.
- **DDQN Superiority:** My findings show that Double Deep Q-Network (DDQN) surpasses other models, such as DQN and PPO, by reducing overestimations and enhancing learning stability.

## Methodological Advice

- **Comprehensive Evaluation:** Utilize a mix of qualitative and quantitative approaches to thoroughly evaluate the effectiveness of both simulations and real-world applications.
- **Adaptive Techniques:** Stay flexible and responsive to the results and feedback, which are crucial for effectively tackling unexpected challenges.

## Practical Experiment Integration

- **Prototyping and Iteration:** Use iterative design and prototyping to progressively refine systems, effectively linking theoretical research and practical implementation.
- **Continuous Feedback:** Continuously seek and integrate feedback from stakeholders and peers to improve simulation models and real-world applications.

## Guidelines for Future Research

### Introduction for Future Research

This chapter provides a comprehensive methodology and advice for researchers involved in simulation-based studies, focusing on ensuring a smooth transition from theoretical models to practical applications.

### Step-by-Step Plan

#### Step 1: Selection of Simulation Environments

- **Research and Evaluation:** Investigate and evaluate available simulation tools that fit your study, such as OpenAI Gym, Unity 3D, and CARLA.
- **Criteria Development:** Establish criteria based on fidelity, scalability, ease of use, and integration capabilities with your existing systems.
- **Preliminary Testing:** Perform initial tests to evaluate these environments against your criteria, focusing on how well they replicate real-world conditions and support your research objectives.

#### Step 2: Managing Expectations and Adaptability

- **Expectation Setting:** Set realistic expectations for what simulations can achieve. Understand that simulations can provide valuable insights but may not capture all real-world nuances.
- **Adaptation Strategies:** Be ready to adjust your research approach based on the outcomes of simulations. Address any data discrepancies and be flexible in modifying your methodologies as needed.

#### Step 3: Methodology Flexibility

- **Continuous Evaluation:** Regularly assess the effectiveness of your chosen methodologies. Stay open to feedback and willing to make iterative improvements.
- **Integration of New Technologies:** Keep an eye on emerging technologies and be prepared to incorporate them into your research to enhance your methods and results.

#### Step 4: Addressing Real-World Constraints

- **Practical Implementation:** Consider the practical constraints you'll face when applying your simulation results to the real world. These might include hardware limitations, environmental variability, and regulatory requirements.
- **Pilot Testing:** Conduct small-scale real-world tests to validate your simulation findings. Use these tests to identify and address any gaps between the simulated and real-world performance.

#### Step 5: Enhancing Data Accuracy

- **Sensor Calibration:** Ensure your sensors are properly calibrated both in simulations and real-world tests. Accurate sensor data is crucial for reliable results.
- **Data Normalization:** Implement robust data normalization techniques to align simulated sensor data with real-world conditions. This helps in maintaining the consistency and reliability of your data.

#### Step 6: Fostering Collaboration

- **Interdisciplinary Teams:** Collaborate with experts from different fields to enrich your research. For example, working with specialists in robotics, computer vision, and AI can provide new insights and approaches.
- **Community Engagement:** Engage with the research community through conferences, workshops, and publications to share your findings and gain valuable feedback.

#### Step 7: Future-Proofing Your Research

- **Scalability:** Design your research to be scalable, so it can be adapted to larger and more complex problems in the future.
- **Long-Term Vision:** Keep a long-term vision for your research, anticipating future challenges and opportunities in the field of simulation-to-reality transfer.

### Conclusion for Future Research

Following these guidelines will help researchers navigate the complexities of simulation-based studies and ensure a successful transition from theoretical models to practical applications. By being methodical, adaptable, and collaborative, you can overcome the challenges of sim2real transfer and contribute valuable insights to the field of autonomous systems and beyond.

<!-- \pagebreak -->

<!-- TODO: replace with more natural sounding language -->
<!-- ## Sources of Inspiration and Conceptual Framework

The inspiration for this research draws from a diverse collection of sources, uniquely combining insights from technical documentation, digital platforms, and academic literature. Central to the inspiration were the challenges of micro mouse competitions and the potential of reinforcement learning (RL) in navigating these complex mazes. These initial sparks of interest were further fueled by dynamic demonstrations of RL applications in autonomous vehicle control, particularly through the lens of YouTube and GitHub repositories, alongside influential academic research.

## Micro mouse Competitions and Reinforcement Learning

Micro mouse competitions, which task small robotic mice with the navigation of mazes, served as a foundational inspiration for this study. The direct application of RL in these competitions and related technological showcases provided a compelling narrative on the potential of RL in real-world problem-solving and autonomous control. The exploration of maze traversal algorithms and the strategies for shortest path finding, as detailed in the insightful Medium article by M. A. Dharmasiri \hyperref[ref15]{[15]}, enriched the conceptual foundation by illustrating practical algorithmic approaches in similar contexts.

## Influential YouTube Demonstrations and GitHub Insights

YouTube videos such as "Self Driving and Drifting RC Car using Reinforcement Learning" \hyperref[ref11]{[11]} and "Reinforcement Learning with Multi-Fidelity Simulators -- RC Car" \hyperref[ref16]{[16]} provided vivid demonstrations of RL's applicability in real-world settings, emphasizing the feasibility of sim-to-real transfer. These resources, along with GitHub repositories detailing ventures like the "Sim2Real_autonomous_vehicle" project \hyperref[ref13]{[13]}, highlighted the practical steps and challenges in implementing RL in physical systems.

## Technical Exploration and Academic Foundation

The academic exploration was significantly shaped by articles on autonomous driving decision control by Q. Song et al.\hyperref[ref12]{[12]} and a survey on sim-to-real transfer in deep reinforcement learning for robotics by W. Zhao, J. P. Queralta, and T. Westerlund \hyperref[ref17]{[17]}, which detailed the application of advanced RL algorithms in controlling autonomous vehicles. These articles provided a deep dive into the methodologies and challenges of applying RL in autonomous systems, offering a broad academic perspective on the field.

## Synthesis and Research Direction

These varied sources collectively informed the development of this research, steering the focus towards the feasibility and intricacies of sim2real transfer in the realm of autonomous navigation. The exploration aims to synthesize insights from both digital and academic realms, tackling the nuanced challenges of applying sophisticated RL models in practical, tangible scenarios.

\pagebreak -->

# Sources of Inspiration and Conceptual Framework

The inspiration for this research comes from a mix of technical documentation, digital platforms, and academic literature. Key influences include the challenges of micro mouse competitions and the potential of reinforcement learning (RL) to navigate complex mazes. My interest was further sparked by dynamic RL applications in autonomous vehicle control showcased on YouTube and GitHub, alongside influential academic research.

## Micro Mouse Competitions and Reinforcement Learning

Micro mouse competitions, where small robotic mice navigate mazes, served as a major inspiration. The use of RL in these competitions showed the potential for solving real-world problems and controlling autonomous systems. Insights from a Medium article by M. A. Dharmasiri on maze traversal algorithms and shortest path strategies provided practical algorithmic approaches relevant to this study \hyperref[ref15]{[15]}.

## Influential YouTube Demonstrations and GitHub Insights

YouTube videos like "Self Driving and Drifting RC Car using Reinforcement Learning" \hyperref[ref11]{[11]} and "Reinforcement Learning with Multi-Fidelity Simulators -- RC Car" \hyperref[ref16]{[16]} vividly demonstrated RL's real-world applicability and the feasibility of sim-to-real transfer. GitHub repositories, such as the "Sim2Real_autonomous_vehicle" project \hyperref[ref13]{[13]}, detailed the practical steps and challenges of implementing RL in physical systems.

## Technical Exploration and Academic Foundation

Academic articles also significantly shaped this research. Notable works include Q. Song et al.'s article on autonomous driving decision control \hyperref[ref12]{[12]} and a survey by W. Zhao, J. P. Queralta, and T. Westerlund on sim-to-real transfer in deep RL for robotics \hyperref[ref17]{[17]}. These articles provided in-depth methodologies and highlighted the challenges of applying RL to autonomous systems.

## Synthesis and Research Direction

These diverse sources collectively informed the direction of this research, focusing on the feasibility and complexities of sim-to-real transfer in autonomous navigation. The goal is to combine insights from both digital and academic realms to address the challenges of applying advanced RL models in real-world scenarios.

\pagebreak

# General Conclusion

This thesis has demonstrated the potential of transferring a trained reinforcement learning (RL) agent from a simulated environment to a real-world setting, focusing on navigating a maze using a remote-controlled (RC) car. The detailed experiments and analyses provide a thorough exploration of this transition.

The research shows that such a transfer is not only possible but also comes with significant challenges. The experiments highlighted in **Chapter 7: Analysis and Results: Addressing the Research Questions** emphasize the importance of normalizing sensor data and adapting control algorithms to handle the unpredictable dynamics of the real world. These adaptations were essential for aligning the simulated models with the real-world scenarios encountered during implementation.

The choice of appropriate virtual environments and reinforcement learning techniques, as discussed in **Chapter 4: Methodology**, was crucial in shaping the experimental approach and ensuring effective simulation training. The Double Deep Q-Network (DDQN) proved to be the most suitable technique, providing a robust framework for navigating the complexities of practical applications.

This study confirms the feasibility of sim-to-real transfers and offers a detailed examination of the intricate mechanics involved in this process. This area is of growing importance in AI and robotics research. By integrating theoretical insights with practical applications, this thesis significantly contributes to the ongoing discussion on the viability and challenges of applying reinforcement learning in real-world scenarios.

In conclusion, while transitioning a trained RL agent from simulation to a real environment is feasible, the process requires careful planning, adaptability, and continual refinement. The challenges highlighted throughout this research underscore the need for ongoing efforts to enhance the robustness and reliability of sim-to-real applications, ensuring they can meet the demands of real-world conditions.

\pagebreak
<!-- TODO: Replace with more natural sounding languague -->
<!-- ## Guest Speakers

## Innovations and Best Practices in AI Projects by Jeroen Boeye at Faktion

Jeroen Boeye's talk, delivered on behalf of Faktion, provided valuable insights into the close relationship between software engineering and artificial intelligence in developing AI solutions. He emphasized the importance of not just focusing on AI technology but also on the software engineering principles that support the creation of robust, scalable, and maintainable AI systems. This approach ensures that AI solutions are both technically sound and viable for long-term application.

During his lecture, Jeroen highlighted several aspects of AI application, notably Chatlayer's impact on conversational AI. He explained how Chatlayer improves chatbot interactions through sophisticated conversational flows, enhancing the accuracy and relevance of exchanges with users. Another point of discussion was Metamaze, which he commended for its innovative methods in automating document processing, creating succinct summaries from extensive documents and emails, showcasing the capabilities of supervised machine learning in administrative tasks.

Jeroen outlined a clear roadmap for successful AI project implementation, stressing the need to validate business cases and adopt a problem-first strategy. He discussed the critical role of high-quality data as the foundation for any AI endeavor and offered strategies for creatively overcoming data limitations. The talk also covered the importance of embracing failures as opportunities for innovation and maintaining open communication with stakeholders about challenges and setbacks.

The lecture further presented various practical AI applications across different industries, such as solar panel detection, unauthorized pool identification, air freight container inspection, and early warning systems for wind turbine gearboxes. Jeroen demonstrated how AI could tackle complex challenges through innovative data sourcing, synthetic data generation, and anomaly detection techniques. He also explored case studies on energy analysis in brick ovens and egg incubation processes, emphasizing the importance of data preprocessing and machine learning models in improving efficiency and outcomes.

Key points from Jeroen's talk included the mastery of data preprocessing and treating data as a dynamic asset to better tailor AI models to specific needs. He shared practical tips on enhancing operational efficiency, such as using host mounts for code integration and Streamlit for dashboard creation, to streamline development processes.

In summary, Jeroen Boeye's lecture offered a thorough perspective on integrating AI technologies in real-world settings. His insights into the vital role of software engineering principles, alongside a deep understanding of AI capabilities and constraints, provided valuable guidance for developing effective and sustainable AI solutions. The lecture not only underscored current AI trends and future directions but also shared practical knowledge on navigating the complexities of AI project execution.

## Pioneering AI Solutions at Noest by Toon Vanhoutte

Toon Vanhoutte's engaging lecture, on behalf of Noest from the Cronos Group, shed light on the effective integration of artificial intelligence and software engineering in developing cutting-edge business solutions. With a dedicated team of 56 local experts, Noest has built a reputation for its pragmatic approach to projects, targeting global impact while valuing craftsmanship, partnership, and enjoyment as core principles. This philosophy extends to their diverse services, which include application development, cloud computing, data analytics, AI innovations, low-code platforms, ERP solutions, and comprehensive system integrations, all supported by a strong partnership with Microsoft.

Toon presented a case study on a packaging company that aimed to revolutionize image search capabilities based on product labels. The project faced various challenges, such as inconsistent PDF formats and large file sizes, which were adeptly managed using Azure Blob Storage for data handling and event-driven processing strategies for efficient, cost-effective solutions, showcasing Noest’s skill in utilizing cloud technologies to address complex issues.

Another significant challenge was enhancing image searchability, which involved recognizing text and objects within images. This was tackled using Azure AI Search, supplemented by Large Language Models (LLMs) and vector search techniques. This approach allowed for nuanced search functionalities beyond simple text queries, showing the advanced capabilities of AI in interpreting complex data.

Toon also explored advancements in semantic search, discussing how different search methods—keyword, vector, and hybrid—along with semantic ranking, could significantly improve the accuracy and contextuality of search results. Practical demonstrations, including comparisons between OCR and GPT-4 vision, illustrated the potential of AI to offer deeper insights based on semantic understanding.

A key takeaway from Toon’s lecture was the importance of setting realistic client expectations regarding AI's capabilities and potential inaccuracies, highlighting the experimental nature of these technologies. The discussion on AI’s evolving landscape emphasized the need for prompt engineering, the challenges of navigating a developing field, and the importance of client education in managing expectations about AI technologies like GPT.

In conclusion, Toon Vanhoutte’s presentation not only highlighted Noest’s innovative work in AI and software engineering but also imparted crucial lessons on innovation, adaptable problem-solving, and the necessity for ongoing learning in the dynamic field of AI. This presentation showcased Noest’s commitment to pushing technological boundaries to create impactful, pragmatic solutions that fully utilize AI’s potential.

\pagebreak -->

# Guest Speakers

## Innovations and Best Practices in AI Projects by Jeroen Boeye at Faktion

Jeroen Boeye, representing Faktion, shared valuable insights into the synergy between software engineering and artificial intelligence in developing AI solutions. He emphasized the importance of not only focusing on AI technology but also integrating solid software engineering principles to create robust, scalable, and maintainable AI systems. This holistic approach ensures that AI solutions are both technically sound and viable for long-term application.

During his lecture, Jeroen highlighted various aspects of AI application, particularly Chatlayer's contributions to conversational AI. He explained how Chatlayer enhances chatbot interactions through sophisticated conversational flows, improving the accuracy and relevance of exchanges with users. Another key point was Metamaze, which he praised for its innovative methods in automating document processing, creating concise summaries from extensive documents and emails, and demonstrating the capabilities of supervised machine learning in administrative tasks.

Jeroen outlined a clear roadmap for successful AI project implementation, stressing the need to validate business cases and adopt a problem-first strategy. He discussed the crucial role of high-quality data as the foundation for any AI endeavor and offered strategies for creatively overcoming data limitations. The talk also covered the importance of viewing failures as opportunities for innovation and maintaining open communication with stakeholders about challenges and setbacks.

The lecture further presented various practical AI applications across different industries, such as solar panel detection, unauthorized pool identification, air freight container inspection, and early warning systems for wind turbine gearboxes. Jeroen demonstrated how AI could tackle complex challenges through innovative data sourcing, synthetic data generation, and anomaly detection techniques. He also explored case studies on energy analysis in brick ovens and egg incubation processes, emphasizing the importance of data preprocessing and machine learning models in improving efficiency and outcomes.

Key points from Jeroen's talk included mastering data preprocessing and treating data as a dynamic asset to better tailor AI models to specific needs. He shared practical tips on enhancing operational efficiency, such as using host mounts for code integration and Streamlit for dashboard creation, to streamline development processes.

In summary, Jeroen Boeye's lecture offered a comprehensive perspective on integrating AI technologies in real-world settings. His insights into the vital role of software engineering principles, alongside a deep understanding of AI capabilities and constraints, provided valuable guidance for developing effective and sustainable AI solutions. The lecture not only underscored current AI trends and future directions but also shared practical knowledge on navigating the complexities of AI project execution.

## Pioneering AI Solutions at Noest by Toon Vanhoutte

Toon Vanhoutte, speaking on behalf of Noest from the Cronos Group, delivered an engaging lecture on the effective integration of artificial intelligence and software engineering in developing cutting-edge business solutions. With a dedicated team of 56 local experts, Noest has built a reputation for its pragmatic approach to projects, targeting global impact while valuing craftsmanship, partnership, and enjoyment as core principles. This philosophy extends to their diverse services, which include application development, cloud computing, data analytics, AI innovations, low-code platforms, ERP solutions, and comprehensive system integrations, all supported by a strong partnership with Microsoft.

Toon presented a case study on a packaging company that aimed to revolutionize image search capabilities based on product labels. The project faced various challenges, such as inconsistent PDF formats and large file sizes, which were adeptly managed using Azure Blob Storage for data handling and event-driven processing strategies for efficient, cost-effective solutions, showcasing Noest’s skill in utilizing cloud technologies to address complex issues.

Another significant challenge was enhancing image searchability, which involved recognizing text and objects within images. This was tackled using Azure AI Search, supplemented by Large Language Models (LLMs) and vector search techniques. This approach allowed for nuanced search functionalities beyond simple text queries, demonstrating the advanced capabilities of AI in interpreting complex data.

Toon also explored advancements in semantic search, discussing how different search methods—keyword, vector, and hybrid—along with semantic ranking, could significantly improve the accuracy and contextuality of search results. Practical demonstrations, including comparisons between OCR and GPT-4 vision, illustrated the potential of AI to offer deeper insights based on semantic understanding.

A key takeaway from Toon’s lecture was the importance of setting realistic client expectations regarding AI's capabilities and potential inaccuracies, highlighting the experimental nature of these technologies. The discussion on AI’s evolving landscape emphasized the need for prompt engineering, the challenges of navigating a developing field, and the importance of client education in managing expectations about AI technologies like GPT.

In conclusion, Toon Vanhoutte’s presentation not only highlighted Noest’s innovative work in AI and software engineering but also imparted crucial lessons on innovation, adaptable problem-solving, and the necessity for ongoing learning in the dynamic field of AI. This presentation showcased Noest’s commitment to pushing technological boundaries to create impactful, pragmatic solutions that fully utilize AI’s potential.

\pagebreak

# Installation Steps

This section outlines the required steps to install and set up the project environment. Following these instructions will ensure the successful deployment of the autonomous navigation system.

## Prerequisites

Before starting the setup process, make sure you have the following:

- **Git:** For cloning the project repository.
- **Docker:** To containerize the web application and ensure a consistent runtime environment.
- **Python 3.11 and pip:** If you prefer running the project without Docker, Use python along with the dependencies listed in `/web_app/web/requirements.txt` to get the project running.

## Repository Setup

To clone the repository and navigate to the project directory, use these commands:

```bash
git clone https://github.com/driessenslucas/researchproject.git
cd researchproject
```

## Hardware Setup and Assembly

### Introduction to Hardware Components

Here’s an overview of the hardware components used in the project, including the RC car, sensors, and microcontrollers. Proper integration of these components is essential for the autonomous navigation system to function correctly.

<!-- ![Final RC Car (Image created by author)](./images/final_test/jp_final.jpeg "Final RC Car (Image created by author)") -->
\begin{figure}[htbp]
    \centering
    \begin{minipage}{0.8\textwidth}
        \includegraphics[width=4in]{./images/final_test/jp_final.jpeg}
    \end{minipage}
    \caption{Final RC Car (Image created by author)}
\end{figure}

<!-- #### Components List

- **Core Components**:
  - ESP32-WROOM-32 module (Datasheet available at [Espressif](https://www.espressif.com/sites/default/files/documentation/esp32-wroom-32_datasheet_en.pdf))
  - 3D printed parts from Thingiverse ([hc-sr04](https://www.thingiverse.com/thing:3436448/files), [top plate + alternative for the robot kit](https://www.thingiverse.com/thing:2544002))
  - Motor Driver (Available at [DFRobot](https://www.dfrobot.com/product-66.html))
  - 2WD robot kit (Available at [DFRobot](https://www.dfrobot.com/product-367.html))
  - Mini OLED screen (Available at [Amazon](https://www.amazon.com.be/dp/B0BB1T23LF))
  - Sensors (Available at [Amazon](https://www.amazon.com.be/dp/B07XF4815H))
  - Battery for ESP32 (Available at [Amazon](https://www.amazon.com.be/dp/B09Q4ZMNLW))
- **Supplementary Materials**: Screws, wires, and tools required for assembly.
  - 4mm thick screws 5mm long to hold the wood together (Available at [brico](https://www.brico.be/nl/gereedschap-installatie/ijzerwaren/schroeven/universele-schroeven/sencys-universele-schroeven-torx-staal-gegalvaniseerd-20-x-4-mm-30-stuks/5368208))
  - M3 bolts & nuts (Available at [brico](https://www.brico.be/nl/gereedschap-installatie/ijzerwaren/bouten/sencys-cilinderkop-bout-gegalvaniseerd-staal-m3-x-12-mm-30-stuks/5367637))
  - Wood for the maze (Available at [brico](https://www.brico.be/nl/bouwmaterialen/hout/multiplex-panelen/sencys-vochtwerend-multiplex-paneel-topplex-250x122x1-8cm/5356349)) -->

## Components List

### Core Components

- **ESP32-WROOM-32 module**
  - Available at Amazon.com
- **3D printed parts**
  - Available at Thingiverse.com
    - HC-SR04 holders: <https://www.thingiverse.com/thing:3436448/files>
    - Top plate + alternative for the robot kit: <https://www.thingiverse.com/thing:2544002>
- **Motor Controller (L298N)**
  - Available at DFRobot.com
- **2WD miniQ Robot Chassist**
  - Available at DFRobot.com
- **Mini OLED screen**
  - Available at Amazon.com
- **Sensors (HC-SR04 and MPU6050)**
  - Available at Amazon.com
- **18650 Battery Shield for ESP32**
  - Available at Amazon.com

### Supplementary Materials

- **Screws, wires, and tools required for assembly**
  - 4mm thick screws, 5mm long to hold the wood together
    - Available at most hardware stores
  - M3 bolts & nuts
    - Available at most hardware stores
  - Wood for the maze
    - Available at most hardware stores

### Wiring Guide

In this section, you find how the components are connected to the ESP32 microcontroller. Proper wiring is essential for the system to function correctly.

The pin connections for the ESP32 microcontroller are as follows:

```c
int E1 = 2; //PWM motor 1
int M1 = 17; //GPIO motor 1
int E2 = 19; //PWM motor 2
int M2 = 4; //GPIO motor 2

int sensor0Trig = 27; //GPIO right sensor
int sensor0Echo = 26; //GPIO right sensor

int sensor1Trig = 33; //GPIO left sensor
int sensor1Echo = 32; //GPIO left sensor

int sensor2Trig = 25; //GPIO front sensor
int sensor2Echo = 35; //GPIO front sensor

// OLED display and MPU6050 pins
#define SDA_PIN 21 // this is the default sda pin on the esp32
#define SCL_PIN 22 // this is the default scl pin on the esp32
```

**ESP32 Wiring:**

<!-- ![ESP32 Wiring (Image created by author)](./images/schematics/esp_updated.png "ESP32 Wiring (Image created by author)") -->
\begin{figure}[htbp]
    \centering
    \begin{minipage}{0.8\textwidth}
        \includegraphics[width=4in]{./images/schematics/esp_updated.png}
    \end{minipage}
    \caption{ESP32 Wiring (Image created by author)}
\end{figure}

In the image above, you can see the wiring connections for the ESP32 microcontroller. The pins are connected to the motor driver, sensors, OLED display, and MPU6050 gyroscope.

### Software Configuration

1. **Arduino IDE Setup**: Install the Arduino IDE to program the ESP32 microcontroller. Follow Espressif Systems' instructions to add the ESP32 board to the Arduino IDE.
2. **Library Installation:** Install the [ESP32_SSD1306](https://github.com/lexus2k/ssd1306/tree/master) library for the OLED display functionality.
3. **Code Upload:** Transfer the scripts from the [esp32](./esp32) folder to the ESP32 device. Modify the WiFi settings in the script to match your local network configuration.

## Web Application Setup

### Note

To ensure a smooth setup of the virtual display, it’s recommended to run `docker-compose down` after each session.

### Steps

1. Navigate to the web application's source code directory:

   ```bash
   cd ./web_app/
   ```

2. Launch the Docker containers with:

   ```bash
   docker-compose up -d
   ```

## Usage Instructions

1. Open your web browser and go to <http://localhost:8500> or <http://localhost:5000>.
2. Enter the ESP32's IP address in the web app and select the desired model for deployment.
3. You can also run a virtual demonstration without engaging the physical vehicle.
4. Start the maze navigation by clicking the `Start Maze` button.

A demonstration of the project is available (see Web App Demo in the Video References section).

## Additional Information: Model Training

- You can use a pre-trained model or train a new model using the script in [train](./training/train.py).
- This training script is optimized for efficiency and can be run directly on the Raspberry Pi.
- After training, you will be prompted to save the new model. If saved, it will be stored in the [models](./web_app/models) directory of the `web_app` folder.

By following these steps, you can successfully set up and deploy the autonomous navigation system, ensuring it runs smoothly both in simulations and real-world scenarios.

\pagebreak

# Video References

1. **Video E1 - Gyroscope Calibration**: Testing the MPU6050 gyroscope's ability to correct the car's orientation for accurate navigation, aiming to refine control over the vehicle's movement through maze environments.
   - Click here to go to the video: [Video E1](https://github.com/driessenslucas/researchproject/assets/91117911/32d9e29f-6d5a-4676-b609-2c08923ca1ac)

    \begin{figure}[htbp]
        \centering
        \begin{minipage}{0.2\textwidth}
            \includegraphics[width=1in]{qr_codes/qr_code_1.png}
        \end{minipage}
        \caption{QR code for video Video E1. (Video by author.)}
    \end{figure}

2. **Video E2 - Navigational Corrections**: Addressing alignment issues when attempting precise 90-degree turns and realigning the car's forward movement to rectify a persistent ~3-degree offset.
   - [Video E2](https://github.com/driessenslucas/researchproject/assets/91117911/624b40f2-bee8-49f6-961d-1f72ab18fe13)

    \begin{figure}[htbp]
        \centering
        \begin{minipage}{0.2\textwidth}
            \includegraphics[width=1in]{qr_codes/qr_code_2.png}
        \end{minipage}
        \caption{QR code for video Video E2. (Video by author.)}
    \end{figure}

3. **Video E6 - Encoder Implementation**: Introducing rotary encoders to the setup, hoping to gain more precise control over the car's movements by accurately measuring wheel rotations, thus refining the vehicle's navigation capabilities.
   - Click here to go to the video: [Video E6](https://github.com/driessenslucas/researchproject/assets/91117911/9728e29a-d2fa-48fa-b6e0-e2e1da92228f)

    \begin{figure}[htbp]
        \centering
        \begin{minipage}{0.2\textwidth}
            \includegraphics[width=1in]{qr_codes/qr_code_3.png}
        \end{minipage}
        \caption{QR code for video Video E6. (Video by author.)}
    \end{figure}

4. **Video E7 - Troubleshooting Encoder Malfunction**: Addressing a malfunction with one of the encoders that halted further tests, highlighting the practical challenges of maintaining hardware reliability.
   - Click here to go to the video: [Video E7](https://github.com/driessenslucas/researchproject/assets/91117911/b9ce2cc3-85fd-4136-8670-516c123ba442)

    \begin{figure}[htbp]
        \centering
        \begin{minipage}{0.2\textwidth}
            \includegraphics[width=1in]{qr_codes/qr_code_4.png}
        \end{minipage}
        \caption{QR code for video Video E7. (Video by author.)}
    \end{figure}

5. **Video E9 - Outdoor Navigation Test**: Navigating the RC-car on uneven outdoor surfaces, where variations greatly affected performance, underscoring the importance of environmental factors in autonomous navigation.
   - Click here to go to the video: [Video E9](https://github.com/driessenslucas/researchproject/assets/91117911/02df8a25-b7f0-4061-89b7-414e6d25d31c)

    \begin{figure}[htbp]
        \centering
        \begin{minipage}{0.2\textwidth}
            \includegraphics[width=1in]{qr_codes/qr_code_5.png}
        \end{minipage}
        \caption{QR code for video Video E9. (Video by author.)}
    \end{figure}

6. **Video E11 - Indoor Controlled Test**: Conducting controlled indoor tests to closely monitor and adjust the RC-car's navigation strategies, reflecting on the complexities of sim-to-real transfer.
   - Click here to go to the video: [Video E11](https://github.com/driessenslucas/researchproject/assets/91117911/ce0f47e9-26cd-459e-8b26-ff345d1ee96b)

    \begin{figure}[htbp]
        \centering
        \begin{minipage}{0.2\textwidth}
            \includegraphics[width=1in]{qr_codes/qr_code_6.png}
        \end{minipage}
        \caption{QR code for video Video E11. (Video by author.)}
    \end{figure}

7. **Web App Demo**: A demonstration of the web application's functionality, showcasing the user interface and the autonomous navigation system's control features.
   - Click here to go to the video: [Web App Demo](https://github.com/driessenslucas/researchproject/assets/91117911/b440b295-6430-4401-845a-a94186a9345f)

    \begin{figure}[htbp]
        \centering
        \begin{minipage}{0.2\textwidth}
            \includegraphics[width=1in]{qr_codes/qr_code_7.png}
        \end{minipage}
        \caption{QR code for video Web App Demo. (Video by author.)}
    \end{figure}

8. **DDQN Simulation test**: A simulation test of the DDQN model navigating a maze environment, demonstrating the model's learning capabilities and decision-making processes.
   - Click here to go to the video: [DDQN Simulation](https://github.com/driessenslucas/researchproject/assets/91117911/66539a97-e276-430f-ab93-4a8a5138ee5e)

    \begin{figure}[htbp]
        \centering
        \begin{minipage}{0.2\textwidth}
            \includegraphics[width=1in]{qr_codes/qr_code_8.png}
        \end{minipage}
        \caption{QR code for video DDQN Simulation. (Video by author.)}
    \end{figure}

\pagebreak

# References

\[1\]\label{ref1} G. Brockman et al., "OpenAI Gym," arXiv preprint arXiv:1606.01540, 2016.

\[2\]\label{ref2} A. Dosovitskiy et al., "CARLA: An Open Urban Driving Simulator," in Proceedings of the 1st Annual Conference on Robot Learning, 2017.

\[3\]\label{ref3} H. Van Hasselt, A. Guez, and D. Silver, "Deep Reinforcement Learning with Double Q-learning," in Proceedings of the AAAI Conference on Artificial Intelligence, 2016.

\[4\]\label{ref4} J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv preprint arXiv:1707.06347, 2017.

\[5\]\label{ref5} J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.

\[6\]\label{ref6} K. Bousmalis et al., "Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping," in IEEE International Conference on Robotics and Automation (ICRA), 2018.

\[7\]\label{ref7} Y. Pan and Q. Yang, "A Survey on Transfer Learning," IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, Oct. 2010.

\[8\]\label{ref8} A. A. Rusu et al., "Sim-to-Real Robot Learning from Pixels with Progressive Nets," in Proceedings of the Conference on Robot Learning, 2016.

\[9\]\label{ref9} S. James et al., "Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks," in Proceedings of the 2019 International Conference on Robotics and Automation (ICRA), 2019.

\[10\]\label{ref10} F. Sadeghi and S. Levine, "(CAD)^2RL: Real Single-Image Flight without a Single Real Image," in Proceedings of Robotics: Science and Systems, 2016.

\[11\]\label{ref11} "Self Driving and Drifting RC Car using Reinforcement Learning," YouTube, Aug. 19, 2019. [Online Video]. Available: <https://www.youtube.com/watch?v=U0-Jswwf0hw>. [Accessed: Jan. 29, 2024].

\[12\]\label{ref12} Q. Song et al., "Autonomous Driving Decision Control Based on Improved Proximal Policy Optimization Algorithm," Applied Sciences, vol. 13, no. 11, Art. no. 11, Jan. 2023. [Online]. Available: <https://www.mdpi.com/2076-3417/13/11/6400>. [Accessed: Jan. 29, 2024].

\[13\]\label{ref13} DailyL, "Sim2Real_autonomous_vehicle," GitHub repository, Nov. 14, 2023. [Online]. Available: <https://github.com/DailyL/Sim2Real_autonomous_vehicle>. [Accessed: Jan. 29, 2024].

\[14\]\label{ref14} "OpenGL inside Docker containers, this is how I did it," Reddit, r/docker. [Online]. Available: <https://www.reddit.com/r/docker/comments/8d3qox/opengl_inside_docker_containers_this_is_how_i_did/>. [Accessed: Jan. 29, 2024].

\[15\]\label{ref15} M. A. Dharmasiri, "Micromouse from scratch | Algorithm- Maze traversal | Shortest path | Floodfill," Medium, [Online]. Available: <https://medium.com/@minikiraniamayadharmasiri/micromouse-from-scratch-algorithm-maze-traversal-shortest-path-floodfill-741242e8510>. [Accessed: Jan. 29, 2024].

\[16\]\label{ref16} "Reinforcement Learning with Multi-Fidelity Simulators -- RC Car," YouTube, Dec. 30, 2014. [Online Video]. Available: <https://www.youtube.com/watch?v=c_d0Is3bxXA>. [Accessed: Jan. 29, 2024].

\[17\]\label{ref17} W. Zhao, J. P. Queralta, and T. Westerlund, "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: A Survey," in 2020 IEEE Symposium Series on Computational Intelligence (SSCI), Dec. 2020, pp. 737–744. [Online]. Available: <https://arxiv.org/pdf/2009.13303.pdf>.

\[18\]\label{ref18} R. S. Sutton and A.G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA: The MIT Press, 2018.

\[19\]\label{ref19} H. van Hasselt, A. Guez, D. Silver, et al., "Deep Reinforcement Learning with Double Q-learning," arXiv preprint arXiv:1509.06461, 2015.

\[20\]\label{ref20} Papers With Code, "Double DQN Explained," [Online]. Available: <https://paperswithcode.com/method/double-dqn>.

\[21\]\label{ref21} D. Jayakody, "Double Deep Q-Networks (DDQN) - A Quick Intro (with Code)," 2020. [Online]. Available: <https://dilithjay.com/blog/2020/04/18/double-deep-q-networks-ddqn-a-quick-intro-with-code/>.

\[22\]\label{ref22} H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in _Proc. of AAAI Conf. on Artificial Intelligence_, 2016.

\[23\]\label{ref23} V. Mnih et al., "Human-level control through deep reinforcement learning," _Nature_, vol. 518, no. 7540, pp. 529-533, 2015.

\[24\]\label{ref24} C. J. C. H. Watkins and P. Dayan, "Q-learning," _Machine Learning_, vol. 8, no. 3-4, pp. 279-292, 1992.

\[25\]\label{ref25} J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," _arXiv preprint arXiv:1707.06347_, 2017.

\[26\]\label{ref26} V. R. Konda and J. N. Tsitsiklis, "Actor-critic algorithms," in _Proc. of the 13th International Conf. on Neural Information Processing Systems_, pp. 1008-1014, 2000.

\[27\]\label{ref27} T. Saanum, "Reinforcement Learning with Simple Sequence Priors," arXiv preprint arXiv:2305.17109, 2024.