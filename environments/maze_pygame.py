import gym
import numpy as np
from gym import spaces
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import pygame
import random 
import matplotlib.pyplot as plt
import collections



class RCMazeEnv(gym.Env):
   def __init__(self, maze_size_x=12, maze_size_y=12):
      self.maze_size_x = maze_size_x
      self.maze_size_y = maze_size_y
      self.maze = self.generate_maze()
      self.car_position = (1, 1)
      self.possible_actions = range(3)
      self.car_orientation = 'N'
      self.sensor_readings = {'front': 0, 'left': 0, 'right': 0}
      self.steps = 0
      self.previous_distance = 0
      self.goal = (10, 10)
      self.previous_steps = 0
      self.visited_positions = set()
      self.reset()

         
   def generate_maze(self):
      # For simplicity, create a static maze with walls
      # '1' represents a wall, and '0' represents an open path
      # maze = np.zeros((self.maze_size_y, self.maze_size_x), dtype=int)
      # Add walls to the maze (this can be customized)

      
      layout = [
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
         [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
         [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
         [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
      
   
      maze = np.array(layout)

      return maze

   def reset(self):
      self.car_position = (1, 1)
      self.car_orientation = 'N'
      self.update_sensor_readings()
      self.steps = 0
      self.previous_distance = 0
      self.previous_steps = 0
      self.visited_positions.clear()  # Clear the visited positions
      self.visited_positions.add(self.car_position)
      return self.get_state()

   def step(self, action):
      if action == 0:
         self.move_forward()
      elif action == 1:
         self.turn_left()
      elif action == 2:
         self.turn_right()
      self.update_sensor_readings()
      self.visited_positions.add(self.car_position)
      reward = self.compute_reward()
      self.steps += 1
      done = self.is_done()
      #print each sensor reading and the car orientation
      # print('sensor readings: ', self.sensor_readings)
      # print('car orientation: ', self.car_orientation)
      print('car position: ', self.car_position)
      return self.get_state(), reward, done

   
   def move_forward(self):
      x, y = self.car_position
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

    return normalized_distance
 
   def compute_reward(self):
      # Initialize reward
      reward = 0

      # Check for collision or out of bounds
      if any(self.sensor_readings[direction] == 0 for direction in ['front', 'left', 'right']):
         reward -= 20

      # Check if goal is reached
      if self.car_position == self.goal:
         reward += 500
         # Additional penalty if it takes too many steps to reach the goal
         if self.steps > 1000:
               reward -= 200
         return reward  # Return immediately as this is the terminal state

      # Calculate the Euclidean distance to the goal
      distance_to_goal = ((self.car_position[0] - self.goal[0]) ** 2 + (self.car_position[1] - self.goal[1]) ** 2) ** 0.5

      # Define a maximum reward when the car is at the goal
      max_reward_at_goal = 50

      # Reward based on proximity to the goal
      reward += max_reward_at_goal / (distance_to_goal + 1)  # Adding 1 to avoid division by zero

      # # Reward or penalize based on movement towards or away from the goal
      if distance_to_goal < self.previous_distance:
         reward += 50  # Positive reward for moving closer to the goal
      elif distance_to_goal > self.previous_distance:
         reward -= 25  # Negative reward for moving farther from the goal

      if self.car_position in self.visited_positions:
         # Apply a penalty for revisiting the same position
         reward -= 10
         
      # Penalize for each step taken to encourage efficiency
      reward -= 2
      
      # Update the previous_distance for the next step
      self.previous_distance = distance_to_goal
      return reward

   def is_done(self):
      #is done if it reaches the goal or goes out of bounds or takes more than 3000 steps
      return self.car_position == self.goal or self.steps > 3000 or self.car_position[0] < 0 or self.car_position[1] < 0 or self.car_position[0] > 11 or self.car_position[1] > 11
      
   def get_state(self):
      car_position = [float(coord) for coord in self.car_position]
      sensor_readings = [float(value) for value in self.sensor_readings.values()]
      
      state = car_position + [self.car_orientation] + sensor_readings
      
      # cast state to this ['1.0' '1.0' 'N' '1.0' '1.0' '10.0']
      state = np.array(state, dtype=str)
      
      #get the orientation and convert do label encoding
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

   def init_pygame(self):
        # Initialize Pygame
        pygame.init()
        self.screen_size = (600, 600)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("RC Maze Environment")
      
        

       def render(self, render_mode='human'):
        if render_mode == 'human':
            # Fill the background
            self.screen.fill((255, 255, 255))

            # Render the maze
            for y in range(self.maze_size_y):
                for x in range(self.maze_size_x):
                    if self.maze[y][x] == 1:
                        self.draw_block(x, y, color=(128, 128, 128))  # Draw walls
                    elif (x, y) == self.goal:
                        self.draw_block(x, y, color=(0, 255, 0))  # Draw goal

            # Draw the car
            car_x, car_y = self.car_position
            self.draw_car(car_x, car_y, color=(255, 0, 255))

            # Update the display
            pygame.display.flip()

        elif render_mode == 'rgb_array':
            rendered_maze = np.array(self.maze, dtype=str)
            x, y = self.car_position
            rendered_maze[y][x] = 'C'  # Representing the car
         
            #print array
            print(rendered_maze, '\n') # Print the maze

    def draw_block(self, x, y, color):
        block_size = self.screen_size[0] / self.maze_size_x
        pygame.draw.rect(self.screen, color, (x * block_size, y * block_size, block_size, block_size))

    def draw_car(self, x, y, color):
        block_size = self.screen_size[0] / self.maze_size_x
        car_rect = (x * block_size, y * block_size, block_size, block_size)
        pygame.draw.ellipse(self.screen, color, car_rect)

    def close_pygame(self):
        # Close Pygame
        pygame.quit()

   
