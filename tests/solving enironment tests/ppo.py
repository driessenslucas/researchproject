import gym
import random
import numpy as np   
import matplotlib.pyplot as plt
import collections
import pygame

# Import Tensorflow libraries

import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model


# disable eager execution (optimization)
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

# ###### Tensorflow-GPU ########
try:
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  print("GPU found")
except:
  print("No GPU found")


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)  # Car color
GREEN = (0, 255, 0)  # Goal



class RCMazeEnv(gym.Env):
    def __init__(self, maze_size_x=12, maze_size_y=12):
        self.maze_size_x = maze_size_x
        self.maze_size_y = maze_size_y
        self.maze = self.generate_maze()
        self.car_position = (1, 1)
        self.possible_actions = range(3)
        self.car_orientation = 'E'
        self.sensor_readings = {'front': 0, 'left': 0, 'right': 0}
        self.steps = 0
        self.previous_steps = 0
        self.previous_distance = 0
        self.goal = (10, 10)
        self.visited_positions = set()
        self.reset()

        # Pygame initialization
        pygame.init()
        self.window_width = 600
        self.window_height = 600
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.cell_width = self.window_width / maze_size_x
        self.cell_height = self.window_height / maze_size_y
        # pygame clock
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("RC Maze Game")

        # Load assets
        self.car_image = self.load_image(
            "../textures/car.png",
            int(self.cell_width),
            int(self.cell_height),
        )

        self.wall_image = self.load_image(
            "../textures/wall_center.png",
            int(self.cell_width),
            int(self.cell_height),
        )

        self.goal_image = self.load_image(
            "../textures/door_closed.png",
            int(self.cell_width),
            int(self.cell_height),
        )
        self.floor_image = self.load_image(
            "../textures/floor_mud_e.png",
            int(self.cell_width),
            int(self.cell_height),
        )
        self.top_of_wall = self.load_image(
            "../textures/gargoyle_top_1.png",
            int(self.cell_width),
            int(self.cell_height),
        )
        self.top_of_wall = pygame.transform.rotate(self.top_of_wall, 180)

    def generate_maze(self):
        # For simplicity, create a static maze with walls
        # '1' represents a wall, and '0' represents an open path
        maze = np.zeros((self.maze_size_y, self.maze_size_x), dtype=int)
        # Add walls to the maze (this can be customized)

        layout = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

        maze = np.array(layout)

        return maze

    def reset(self):
        self.car_position = (1, 1)
        self.car_orientation = 'E'
        self.update_sensor_readings()
        self.previous_steps = self.steps
        self.steps = 0
        self.previous_distance = 0
        self.visited_positions.clear()  # Clear the visited positions
        self.visited_positions.add(self.car_position)
        return self.get_state()

    def step(self, action):

        if action == 0:
            self.move_forward()
        elif action == 1:  # Turn left
            self.turn_left()
        elif action == 2:  # Turn right
            self.turn_right()

        self.update_sensor_readings()
        self.visited_positions.add(self.car_position)
        reward = self.compute_reward()
        self.steps += 1
        done = self.is_done()

        return self.get_state(), reward, done

    def move_forward(self):
        x, y = self.car_position

        # Check sensor reading in the direction of car's orientation
        if self.sensor_readings['front'] <= 4:
            # If the sensor reading is 4 or less, do not move forward
            return

        if self.car_orientation == 'N' and y > 0 and self.maze[y - 1][x] != 1:
            self.car_position = (x, y - 1)
        elif self.car_orientation == 'S' and y < self.maze_size_y - 1 and self.maze[y + 1][x] != 1:
            self.car_position = (x, y + 1)
        elif self.car_orientation == 'E' and x < self.maze_size_x - 1 and self.maze[y][x + 1] != 1:
            self.car_position = (x + 1, y)
        elif self.car_orientation == 'W' and x > 0 and self.maze[y][x - 1] != 1:
            self.car_position = (x - 1, y)

    def turn_left(self):
        orientations = ['N', 'W', 'S', 'E']
        idx = orientations.index(self.car_orientation)
        self.car_orientation = orientations[(idx + 1) % 4]

    def turn_right(self):
        orientations = ['N', 'E', 'S', 'W']
        idx = orientations.index(self.car_orientation)
        self.car_orientation = orientations[(idx + 1) % 4]

    def update_sensor_readings(self):
        # Simple sensor implementation: counts steps to the nearest wall
        self.sensor_readings['front'] = self.distance_to_wall('front')
        self.sensor_readings['left'] = self.distance_to_wall('left')
        self.sensor_readings['right'] = self.distance_to_wall('right')

    def distance_to_wall(self, direction):
        x, y = self.car_position
        sensor_max_range = 255  # Maximum range of the ultrasonic sensor

        def calculate_distance(dx, dy):
            distance = 0
            while 0 <= x + distance * dx < self.maze_size_x and \
                0 <= y + distance * dy < self.maze_size_y and \
                self.maze[y + distance * dy][x + distance * dx] != 1:
                distance += 1
                if distance > sensor_max_range:  # Limiting the sensor range
                    break
            return distance

        if direction == 'front':
            if self.car_orientation == 'N':
                distance = calculate_distance(0, -1)
            elif self.car_orientation == 'S':
                distance = calculate_distance(0, 1)
            elif self.car_orientation == 'E':
                distance = calculate_distance(1, 0)
            elif self.car_orientation == 'W':
                distance = calculate_distance(-1, 0)

        elif direction == 'left':
            if self.car_orientation == 'N':
                distance = calculate_distance(-1, 0)
            elif self.car_orientation == 'S':
                distance = calculate_distance(1, 0)
            elif self.car_orientation == 'E':
                distance = calculate_distance(0, -1)
            elif self.car_orientation == 'W':
                distance = calculate_distance(0, 1)

        elif direction == 'right':
            if self.car_orientation == 'N':
                distance = calculate_distance(1, 0)
            elif self.car_orientation == 'S':
                distance = calculate_distance(-1, 0)
            elif self.car_orientation == 'E':
                distance = calculate_distance(0, 1)
            elif self.car_orientation == 'W':
                distance = calculate_distance(0, -1)

        # Normalize the distance to a range of 0-1
        normalized_distance = distance / sensor_max_range
        normalized_distance = max(0, min(normalized_distance, 1))

        return normalized_distance * 1000

    def compute_reward(self):
        """
        Compute the reward based on the current state of the environment


        @return The reward to be added to the step function
        """
        # Initialize reward
        reward = 0

        # Check for collision or out of bounds
        # If the sensor is on the front left right or front side of the sensor is not on the board.
        if any(
            self.sensor_readings[direction] == 0
            for direction in ["front", "left", "right"]
        ):
            reward -= 20

        # Check if goal is reached
        # The reward of the goal.
        if self.car_position == self.goal:
            reward += 500
            # Additional penalty if it takes too many steps to reach the goal
            # If the reward is more than 1000 steps then the reward is decremented by 200.
            if self.steps > 1000:
                reward -= 200
            

        # Calculate the Euclidean distance to the goal
        distance_to_goal = (
            (self.car_position[0] - self.goal[0]) ** 2
            + (self.car_position[1] - self.goal[1]) ** 2
        ) ** 0.5

        # Define a maximum reward when the car is at the goal
        max_reward_at_goal = 50

        # Reward based on proximity to the goal
        reward += max_reward_at_goal / (
            distance_to_goal + 1
        )  # Adding 1 to avoid division by zero

        # # Reward or penalize based on movement towards or away from the goal
        # Move the reward to the goal
        if distance_to_goal < self.previous_distance:
            reward += 50  # Positive reward for moving closer to the goal
        elif distance_to_goal > self.previous_distance:
            reward -= 25  # Negative reward for moving farther from the goal

        # Apply a penalty to revisit the same position
        if self.car_position in self.visited_positions:
            # Apply a penalty for revisiting the same position
            reward -= 10

        # Penalize for each step taken to encourage efficiency
        reward -= 5

        # Update the previous_distance for the next step
        self.previous_distance = distance_to_goal
        return reward

    def is_done(self):
        # is done if it reaches the goal or goes out of bounds or takes more than 3000 steps
        return self.car_position == self.goal or self.steps > 3000 or self.car_position[0] < 0 or self.car_position[1] < 0 or self.car_position[0] > 11 or self.car_position[1] > 11

    def get_state(self):
        car_position = [float(coord) for coord in self.car_position]
        sensor_readings = [float(value) for value in self.sensor_readings.values()]

        state = car_position + [self.car_orientation] + sensor_readings

        # cast state to this ['1.0' '1.0' 'N' '1.0' '1.0' '10.0']
        state = np.array(state, dtype=str)

        # get the orientation and convert do label encoding
        if state[2] == 'N':
            state[2] = 0
        elif state[2] == 'E':
            state[2] = 1
        elif state[2] == 'S':
            state[2] = 2
        elif state[2] == 'W':
            state[2] = 3

        state = np.array(state, dtype=float)

        return state

    def load_image(self, image_path, width, height):
        image = pygame.image.load(image_path)
        image = pygame.transform.scale(image, (width, height))
        return image

    def draw_maze(self):
        for y in range(self.maze_size_y):
            for x in range(self.maze_size_x):
                if self.maze[y][x] == 1:
                    self.screen.blit(
                        self.wall_image, (x * self.cell_width, y * self.cell_height)
                    )

                    self.screen.blit(
                        self.top_of_wall, (x * self.cell_width, y * self.cell_height)
                    )
                    # add top of wall
                if self.maze[y][x] == 0:
                    self.screen.blit(
                        self.floor_image, (x * self.cell_width, y * self.cell_height)
                    )

    def draw_car(self):
        if self.car_orientation == "N":
            car_image = pygame.transform.rotate(self.car_image, 180)
        elif self.car_orientation == "E":
            car_image = pygame.transform.rotate(self.car_image, 90)
        elif self.car_orientation == "S":
            car_image = self.car_image
        elif self.car_orientation == "W":
            car_image = pygame.transform.rotate(self.car_image, 270)

        self.screen.blit(
            car_image,
            (
                self.car_position[0] * self.cell_width,
                self.car_position[1] * self.cell_height,
            ),
        )

    def draw_goal(self):
        self.screen.blit(
            self.goal_image,
            (self.goal[0] * self.cell_width, self.goal[1] * self.cell_height),
        )

    def render(self,render_mode='human', framerate=60, delay=0):
        if render_mode == 'human':
            self.draw_maze()
            self.draw_car()
            self.draw_goal()
            pygame.display.flip()
            self.clock.tick(framerate)  
        elif render_mode == 'rgb_array':
            rendered_maze = np.array(self.maze, dtype=str)
            x, y = self.car_position
            rendered_maze[y][x] = 'C'
            # print array
            print(rendered_maze, '\n')

    def close_pygame(self):
        # Close the Pygame window
        pygame.quit()


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as mse


class PPOAgent:
    def __init__(self, env, action_dim):
        self.env = env
        self.clip_epsilon = 0.2
        self.optimizer = Adam(learning_rate=0.001)
    
        self.action_dim = action_dim
        self.policy_network = self.build_policy_network()
        self.value_network = self.build_value_network()

    def build_policy_network(self):
    # Create a policy network appropriate for your custom environment
        policy_network = Sequential([
            Input(shape=(6,)),  # Adjust the input shape to (None, 6)
            Dense(32, activation='tanh'),
            Dense(64, activation='tanh'),
            Dense(32, activation='tanh'),
            Dense(self.action_dim, activation='softmax')
        ])
        return policy_network

    def build_value_network(self):
        # Create a value network appropriate for your custom environment
        value_network = Sequential([
            Input(shape=(6,)),  # Adjust the input shape to (None, 6)
            Dense(32, activation='tanh'),
            Dense(64, activation='tanh'),
            Dense(32, activation='tanh'),
            Dense(1, activation='linear')
        ])
        return value_network

    def process_observation(self, observation):
        """Flatten observation array and check its validity."""
        flattened_observation = np.hstack(observation)
        if flattened_observation.shape[0] != 6:
            raise ValueError("Invalid observation shape.")
        return flattened_observation

    def compute_discounted_rewards(self, rewards, gamma=0.99):
        """Compute discounted rewards for the episode."""
        discounted_rewards = np.zeros_like(rewards)
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            discounted_rewards[i] = R
        return discounted_rewards

    def compute_advantages(self, discounted_rewards, observations):
        """Compute advantages based on discounted rewards and observations."""
        baseline = np.mean(discounted_rewards)
        advantages = discounted_rewards - baseline
        return advantages

    def compute_loss(self, observations, actions, advantages, old_probabilities):
        """Compute the PPO loss."""
        new_probabilities = self.policy_network(observations)
        action_masks = tf.one_hot(actions, self.action_dim, dtype=tf.float32)
        new_action_probabilities = tf.reduce_sum(action_masks * new_probabilities, axis=1)
        old_action_probabilities = tf.reduce_sum(action_masks * old_probabilities, axis=1)

        ratio = new_action_probabilities / (old_action_probabilities + 1e-10)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate_objective = tf.minimum(ratio * advantages, clipped_ratio * advantages)
        loss = -tf.reduce_mean(surrogate_objective)
        return loss

    def train_step(self, observations, actions, advantages, old_probabilities):
        """Train the network with one step of samples."""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(observations, actions, advantages, old_probabilities)
        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))
        return loss

    def train_episode(self,episode, max_steps):
        """Run one episode of training, collecting necessary data and updating the network."""
        observations, actions, rewards, probabilities = [], [], [], []
        total_reward = 0
        observation = self.env.reset()
        done = False

        path_history = []
        while not done:

            path_history.append((episode, self.env.car_position))
            flattened_observation = self.process_observation(observation)
            action_probabilities = self.policy_network.predict(np.expand_dims(flattened_observation, axis=0))
            action = np.random.choice(self.action_dim, p=action_probabilities.ravel())
            
            # render the environment
            # self.env.render(framerate=720, delay=0)
            observation, reward, done = self.env.step(action)
            total_reward += reward
            observations.append(flattened_observation)

            if done:
                reward += 10000

            actions.append(action)
            rewards.append(reward)
            probabilities.append(action_probabilities.ravel())

        discounted_rewards = self.compute_discounted_rewards(rewards)
        advantages = self.compute_advantages(discounted_rewards, observations)

        loss = self.train_step(np.vstack(observations), np.array(actions), advantages, np.vstack(probabilities))
        return total_reward, loss, path_history

    def plot_and_save_results(self, reward_history, loss_history,episode_path_history):
        # Reward history plot
        plt.plot(reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title("Reward History for PPO")
        plt.savefig('../images/training_images/reward_history_PPO.png')
        #plt.show()()

        # check -
        self.plot_visit_heatmap(episode_path_history, title="Visit Heatmap for PPO")

        # print reward distribution
        self.plot_reward_distribution(reward_history, title="Reward Distribution for PPO")

        try:
            self.plot_move_avg_reward(reward_history)
        except:
            pass

    def plot_move_avg_reward(self, reward_history, window_size=10):
                # moving average
        window_size = 10
        moving_avg = np.convolve(reward_history, np.ones(window_size) / window_size, mode="valid")

        low_point = np.argmin(moving_avg)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history, label="Reward per Episode", color="lightgray")
        plt.plot( range(window_size - 1, len(reward_history)), moving_avg, label="Moving Average", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Average Reward per Episode with Moving Average for PPO")
        plt.legend()
        plt.savefig("../images/training_images/reward_per_episode_with_moving_avg_PPO.png")
        #plt.show()()

    def plot_reward_distribution(self, rewards, title="Reward Distribution"):
        plt.hist(rewards, bins=30, alpha=0.7, color="blue")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.savefig("../images/training_images/reward_distribution_PPO.png")
        #plt.show()()
        

    def plot_mse_history(self, mse_history):
        plt.plot(mse_history)
        plt.xlabel('Episode')
        plt.ylabel('loss')
        plt.title("loss over time for PPO")
        plt.savefig("../images/training_images/mse_history_PPO.png")
        #plt.show()()

    def plot_visit_heatmap(self, episode_path_history, title="Visit Heatmap for PPO"):

        from math import nan
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap, LogNorm

        # convert to dictionary
        episode_path_dict = {}
        for episode, position in episode_path_history:
            if episode in episode_path_dict:
                episode_path_dict[episode].append(position)
            else:
                episode_path_dict[episode] = [position]

        maze = self.env.maze

        visit_counts = np.zeros((maze.shape[0], maze.shape[1]), dtype=int)



        for episode, path in episode_path_dict.items():
            for position in path:
                visit_counts[position] += 1

        # visit_counts[env.maze == 1] = -1
        print(visit_counts)

        # Transpose visit_counts to match the expected orientation
        visit_counts_transposed = visit_counts.T  # Transpose the matrix

        # Filter out the wall cells by setting their count to NaN for visualization
        # filtered_counts = np.where(
        #     visit_counts_transposed == -1, np.nan, visit_counts_transposed
        # )
        filtered_counts = visit_counts_transposed
        # set the walls to -1
        filtered_counts[env.maze == 1] = -1
        

        # Define a continuous colormap (you can choose any colormap you like)
        cmap = plt.cm.plasma
        cmap.set_bad("white")  # Use gray for NaN (walls)
        plt.figure(figsize=(12, 8))
        # Use LogNorm for logarithmic normalization; set vmin to a small value > 0 to handle cells with 0 visits
        plt.imshow(
            filtered_counts,
            cmap=cmap,
            norm=LogNorm(vmin=0.1, vmax=np.nanmax(filtered_counts)),
            interpolation="nearest",
        )
        # add the nr of visits to the cells
        for i in range(visit_counts_transposed.shape[0]):
            for j in range(visit_counts_transposed.shape[1]):
                if visit_counts_transposed[i, j] != -1 or visit_counts_transposed[i, j] != nan:
                    plt.text(j, i, visit_counts_transposed[i, j], ha="center", va="center")

        plt.colorbar(label="Number of Visits")
        plt.title(title)
        plt.savefig("../images/training_images/visit_heatmap_PPO.png")
        #plt.show()()
            # save the image

    def train(self, num_episodes, max_steps_per_episode):
        """Main training loop."""
        reward_history = []
        loss_history = []
        episode_path_history = []
        for episode in range(num_episodes):
            self.env.reset()
            total_reward, loss, path_history = self.train_episode(episode ,max_steps_per_episode)
            reward_history.append(total_reward)
            loss_history.append(loss)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Loss: {loss}")
            for path in path_history:
                episode_path_history.append(path)
        
        print('reward_history:',reward_history)
        print('loss_history:',loss_history)
        print('episode_path_history:',episode_path_history)
        self.plot_and_save_results(reward_history, loss_history, episode_path_history)


if __name__ == "__main__":

    env = RCMazeEnv()  # Create your custom environment

    observation_dim = env.reset().shape[0]  # Adjust this based on your custom environment's state space
    action_dim = 3  # Adjust this based on your custom environment's action space
    ppo_agent = PPOAgent(env, action_dim=action_dim)

    ppo_agent.train(num_episodes=175, max_steps_per_episode=3000)
