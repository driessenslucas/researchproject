import numpy as np

import gym
from gym import spaces

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import pygame


import gym
import random
import numpy as np   
import matplotlib.pyplot as plt
import collections

# Import Tensorflow libraries


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from IPython.display import HTML

from tqdm import tqdm

tqdm.pandas()

# disable eager execution (optimization)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

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
      maze = np.zeros((self.maze_size_y, self.maze_size_x), dtype=int)
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
      distance = 0
      max_distance = self.maze_size_x if direction in ['left', 'right'] else self.maze_size_y
      
      if direction == 'front':
         if self.car_orientation == 'N':
               while y - distance >= 0 and self.maze[y - distance][x] != 1:
                  distance += 1
         elif self.car_orientation == 'S':
               while y + distance < self.maze_size_y and self.maze[y + distance][x] != 1:
                  distance += 1
         elif self.car_orientation == 'E':
               while x + distance < self.maze_size_x and self.maze[y][x + distance] != 1:     
                  distance += 1
         elif self.car_orientation == 'W':
               while x - distance >= 0 and self.maze[y][x - distance] != 1:
                  distance += 1
      elif direction == 'left':
         if self.car_orientation == 'N':
               while x - distance >= 0 and self.maze[y][x - distance] != 1:
                  distance += 1
         elif self.car_orientation == 'S':
               while x + distance < self.maze_size_x and self.maze[y][x + distance] != 1:
                  distance += 1
         elif self.car_orientation == 'E':
               while y - distance >= 0 and self.maze[y - distance][x] != 1:
                  distance += 1
         elif self.car_orientation == 'W':
               while y + distance < self.maze_size_y and self.maze[y + distance][x] != 1:
                  distance += 1
      elif direction == 'right':
         if self.car_orientation == 'N':
               while x + distance < self.maze_size_x and self.maze[y][x + distance] != 1:
                  distance += 1
         elif self.car_orientation == 'S':
               while x - distance >= 0 and self.maze[y][x - distance] != 1:
                  distance += 1
         elif self.car_orientation == 'E':
               while y + distance < self.maze_size_y and self.maze[y + distance][x] != 1:
                  distance += 1
         elif self.car_orientation == 'W':
               while y - distance >= 0 and self.maze[y - distance][x] != 1:
                  distance += 1
      
         # Normalize the measured distance
      normalized_distance = (max_distance - distance - 1) / (max_distance - 1)

      # Ensure the value is within the range [0, 1]
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

   def init_opengl(self):
      # Initialize OpenGL context
      glutInit()
      glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
      glutInitWindowSize(1200, 1200)
      glutCreateWindow("RC Maze Environment")

      # Set up OpenGL environment
      glEnable(GL_DEPTH_TEST)
      glClearColor(0.0, 0.0, 0.0, 0.0)  # Clear to a grey color

      # Set up lighting (optional)
      glEnable(GL_LIGHTING)
      glEnable(GL_LIGHT0)
      glLightfv(GL_LIGHT0, GL_POSITION, [0, 10, 10, 1])
      glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1])
      glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
      
      glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
      glEnable(GL_COLOR_MATERIAL)

      # Set up camera (you may want to make this adjustable)
      gluLookAt(self.maze_size_x / 2, self.maze_size_y / 2, 10,  # Camera position (above the center of the maze)
          self.maze_size_x / 2, self.maze_size_y / 2, 0,  # Look at point (center of the maze)
          0, 1, 0)  # Up vector
      
      glMatrixMode(GL_PROJECTION)
      glLoadIdentity()
      gluPerspective(60, 1, 0.1, 100)  # Adjust field of view angle, aspect ratio, near and far planes
      glMatrixMode(GL_MODELVIEW)
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      
      
      
      
        # Set the rendering function
      glutDisplayFunc(self.render)
      
   def run_opengl(self):
        # Set up the rendering context and callbacks
        # but do NOT call glutMainLoop()
        glutDisplayFunc(self.render)
        glutIdleFunc(self.render)  # Update rendering in idle time

   def render(self):
      
      # Clear buffers
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      
      camera_distance = 5.0  # Distance behind the car
      camera_height = 4.0  # Height above the car

      # # Calculate the camera position based on the car's orientation
      # if self.car_orientation == 'N':  # Facing North
      #    camera_x = self.car_position[0]
      #    camera_y = self.car_position[1] + camera_distance
      #    camera_z = camera_height
      # elif self.car_orientation == 'S':  # Facing South
      #    camera_x = self.car_position[0]
      #    camera_y = self.car_position[1] - camera_distance
      #    camera_z = camera_height
      # elif self.car_orientation == 'E':  # Facing East
      #    camera_x = self.car_position[0] - camera_distance
      #    camera_y = self.car_position[1]
      #    camera_z = camera_height
      # elif self.car_orientation == 'W':  # Facing West
      #    camera_x = self.car_position[0] + camera_distance
      #    camera_y = self.car_position[1]
      #    camera_z = camera_height

      # # The point where the camera should be pointed: the car's position
      # look_at_x = self.car_position[0]
      # look_at_y = self.car_position[1]
      # look_at_z = 0  # Assuming the car is at ground level (z=0)

      # # Set up the camera
      # glMatrixMode(GL_MODELVIEW)
      # glLoadIdentity()
      # gluLookAt(camera_x, camera_y, camera_z,  # Camera position (x, y, z)
      #          look_at_x, look_at_y, look_at_z,  # Look at position (x, y, z)
      #          0, 0, 1)  # Up vector (x, y, z), assuming Z is up

      # Render the maze
      for y in range(self.maze_size_y):
         for x in range(self.maze_size_x):
               if self.maze[y][x] == 1:
                  self.draw_cube(x, y, color=(0.5, 0.5, 0.5))
               elif (x, y) == self.goal:
                  #set color to green
                  self.draw_cube(x, y, color=(0.0, 1.0, 0.0))
                  
      # Render the car's sensor readings
      car_x, car_y = self.car_position
      sensor_colors = (0.0, 1.0, 1.0)  # Cyan color for sensor lines
      if self.car_orientation == 'N':
         # For North, 'left' is to the West, 'right' is to the East
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['front'], sensor_colors, orientation='front')
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['left'], sensor_colors, orientation='left')
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['right'], sensor_colors, orientation='right')
      elif self.car_orientation == 'S':
         # For South, 'left' is to the East, 'right' is to the West
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['front'], sensor_colors, orientation='back')
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['left'], sensor_colors, orientation='right')
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['right'], sensor_colors, orientation='left')
      elif self.car_orientation == 'E':
         # For East, 'left' is to the North, 'right' is to the South
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['front'], sensor_colors, orientation='right')
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['left'], sensor_colors, orientation='front')
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['right'], sensor_colors, orientation='back')
      elif self.car_orientation == 'W':
         # For West, 'left' is to the South, 'right' is to the North
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['front'], sensor_colors, orientation='left')
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['left'], sensor_colors, orientation='back')
         self.draw_sensor_line(car_x, car_y, self.sensor_readings['right'], sensor_colors, orientation='front')


      # Draw the car
      car_x, car_y = self.car_position
      self.draw_car(car_x, car_y, color=(1.0, 0.0, 0.0))
      
      # Swap buffers
      glutSwapBuffers()
      

   def draw_cube(self, x, y, color):
      # Set the color
      glColor3fv(color)

      # Draw a cube at position (x, y)
      glPushMatrix()
      glTranslate(x, y, 0)  # Adjust this to position your cube correctly in 3D
      glScalef(1, 1, 1)  # Adjust the size of your cube
      glutSolidCube(1)  # You may want to adjust the size
      glPopMatrix()
      
   def draw_sensor_line(self, x, y, distance, color, orientation):
      close_threshold = 0.5  # Set the threshold for being 'too close' to the wall
      # print('distance', distance, 'orientation', orientation)
      
      if distance <= close_threshold:
        glColor3fv((1.0, 0.0, 0.0))  # Too close, set color to red
      else:
        glColor3fv(color)  # Not too close, use the specified color
      
      glPushMatrix()  # Save the current transformation matrix
      
      # Translate to the car's position
      glTranslate(x, y, 0.5)  # 0.5 to raise the line to the middle of the wall's height
      
      # Rotate the line based on the car's orientation
      if orientation == 'left':
         glRotatef(90, 0, 0, 1)  # Rotate 90 degrees around the z-axis for left
      elif orientation == 'right':
         glRotatef(-90, 0, 0, 1)  # Rotate -90 degrees for right
      elif orientation == 'front':
         pass  # No rotation needed for front
      elif orientation == 'back':
         glRotatef(180, 0, 0, 1)  # Rotate 180 degrees for back
      
      # Rotate the line to lay flat on the x-axis
      glRotatef(90, 0, 1, 0)  # Rotate 90 degrees around the y-axis to lay flat
      
      # Draw the cylinder representing the sensor line
      
      #make length of the sensor dependent on the distance to the wall with a max distance of 0.5
      if distance > 0.5:
         distance = 0.5
         
      glutSolidCylinder(0.05, distance, 5, 5)  # Draw cylinder with a smaller radius and length = sensor reading
      
      glPopMatrix()  # Restore the transformation matrix

      
      
   def draw_car(self, x, y, color):
            # Set the color
      glColor3fv(color)

      # Draw a cube at position (x, y)
      glPushMatrix()
      glTranslate(x, y, 0)  # Adjust this to position your cube correctly in 3D
      glScalef(1, 1, 1)  # Adjust the size of your cube
      glutSolidCube(0.5)  # You may want to adjust the size
      glPopMatrix()

   def close_opengl(self):
      # Close the OpenGL context
      glutLeaveMainLoop()

        
 

class DQAgent:
    def __init__(self, replayCapacity, inputShape, outputShape):
        ## Initialize replay memory
        self.capacity = replayCapacity
        self.memory = collections.deque(maxlen=self.capacity)
        self.populated = False
        ## Policiy model
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.policy_model = self.buildNetwork()

        ## Target model
        self.target_model = self.buildNetwork()
        self.target_model.set_weights(self.policy_model.get_weights())

    def addToReplayMemory(self, step):
        self.step = step
        self.memory.append(self.step)

    def sampleFromReplayMemory(self, batchSize):
        self.batchSize = batchSize
        if self.batchSize > len(self.memory):
            self.populated = False
            return self.populated
        else:
            return random.sample(self.memory, self.batchSize)


    def buildNetwork(self):
        model = Sequential()
        model.add(Dense(32, input_shape=self.inputShape, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.outputShape, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['MeanSquaredError'])
        return model

    def policy_network_fit(self,batch, batchSize):
        self.batchSize = batchSize
        self.batch = batch


    def policy_network_predict(self, state):
        self.state = state
        self.qPolicy = self.policy_model.predict(self.state)
        return self.qPolicy

    def target_network_predict(self, state):
        self.state = state
        self.qTarget = self.target_model.predict(self.state)
        return self.qTarget

    def update_target_network(self):
        self.target_model.set_weights(self.policy_model.get_weights()) 
        
# set main
if __name__ == "__main__":
   env = RCMazeEnv()
   state = env.reset()

   env.init_opengl()
   env.run_opengl()

   # Model parameters
   REPLAY_MEMORY_CAPACITY = 20000
   # MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
   POSSIBLE_ACTIONS = env.possible_actions

   # state = state[0]
   # create DQN agent
   agent = DQAgent(replayCapacity=REPLAY_MEMORY_CAPACITY, inputShape=state.shape, outputShape=len(POSSIBLE_ACTIONS))


   # reset the parameters
   DISCOUNT = 0.90
   BATCH_SIZE = 64  # How many steps (samples) to use for training
   UPDATE_TARGET_INTERVAL = 500
   EPSILON = 0.99 # Exploration percentage
   MIN_EPSILON = 0.01
   DECAY = 0.9999
   EPISODE_AMOUNT = 100

   # Fill the replay memory with the first batch of samples
   updateCounter = 0
   rewardHistory = []
   epsilonHistory = []

   np.set_printoptions(precision=3, suppress=True)

   for episode in range(EPISODE_AMOUNT):
      episodeReward = 0
      stepCounter = 0  # count the number of successful steps within the episode

      state = env.reset()
      done = False
      epsilonHistory.append(EPSILON)

      while not done:
         glutMainLoopEvent()
         

         if random.random() <= EPSILON:
               action = random.sample(POSSIBLE_ACTIONS, 1)[0]
         else:
               qValues = agent.policy_network_predict(state.reshape(1,-1))
               action = np.argmax(qValues[0])

         newState, reward, done = env.step(action)

         stepCounter +=1

         # store step in replay memory
         step = (state, action, reward, newState, done)
         agent.addToReplayMemory(step)
         state = newState
         episodeReward += reward
         env.render()
         # When enough steps in replay memory -> train policy network
         if len(agent.memory) >= (BATCH_SIZE):
               EPSILON = DECAY * EPSILON
               if EPSILON < MIN_EPSILON:
                  EPSILON = MIN_EPSILON
               # sample minibatch from replay memory
               
               miniBatch = agent.sampleFromReplayMemory(BATCH_SIZE)
               miniBatch_states = np.asarray(list(zip(*miniBatch))[0],dtype=float)
               miniBatch_actions = np.asarray(list(zip(*miniBatch))[1], dtype = int)
               miniBatch_rewards = np.asarray(list(zip(*miniBatch))[2], dtype = float)
               miniBatch_next_state = np.asarray(list(zip(*miniBatch))[3],dtype=float)
               miniBatch_done = np.asarray(list(zip(*miniBatch))[4],dtype=bool)
               
               # current state q values1tch_states)
               y = agent.policy_network_predict(miniBatch_states)

               next_state_q_values = agent.target_network_predict(miniBatch_next_state)
               max_q_next_state = np.max(next_state_q_values,axis=1)

               for i in range(BATCH_SIZE):
                  if miniBatch_done[i]:
                     y[i,miniBatch_actions[i]] = miniBatch_rewards[i]
                  else:
                     y[i,miniBatch_actions[i]] = miniBatch_rewards[i] + DISCOUNT *  max_q_next_state[i]

               agent.policy_model.fit(miniBatch_states, y, batch_size=2048, verbose = 0)
               
               #todo: add early stopping
            
         else:
               continue
         if updateCounter == UPDATE_TARGET_INTERVAL:
               agent.update_target_network()
               updateCounter = 0
         updateCounter += 1
      print('episodeReward for episode ', episode, '= ', episodeReward, 'with epsilon = ', EPSILON)
      rewardHistory.append(episodeReward)
   env.close_pygame()
   env.close()

