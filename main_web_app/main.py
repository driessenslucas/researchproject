import numpy as np
import gym
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import random 
import matplotlib.pyplot as plt
import collections
# Import Tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
# disable eager execution (optimization)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from flask import Flask, send_file, Response, render_template,jsonify, request
import threading
import time
import io
from PIL import Image
import queue
import cv2
import requests
import json
import aiohttp
import asyncio
from threading import Lock
from gpiozero import DistanceSensor
from signal import pause


class RCMazeEnv(gym.Env):
   def __init__(self, maze_size_x=12, maze_size_y=12, esp_ip='192.168.0.7', use_virtual_sensors=True):
      """
       Initialize the maze.
       
       @param maze_size_x - The size of the maze in x - direction
       @param maze_size_y - The size of the maze in y - direction
       @param esp_ip - The IP address of the ESP for motor control
      """
      self.maze_size_x = maze_size_x
      self.maze_size_y = maze_size_y
      self.maze = self.generate_maze()
      self.car_position = (1, 1)
      self.possible_actions = range(3)
      self.car_orientation = 'E'
      self.sensor_readings = {'front': 0, 'left': 0, 'right': 0}
      self.steps = 0
      self.previous_distance = 0
      self.goal = (10, 10)
      self.esp_ip = esp_ip
      self.previous_steps = 0
      self.visited_positions = set()
      self.opelgl_window = (1200,1200)
      self.use_virtual_sensors = use_virtual_sensors
      # self.reset()
      
   def generate_maze(self):
      """
       Generate maze. This is a function that will generate the maze
       
       
       @return A tuple containing the number of rows and the number of columns
      """
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
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
      
      maze = np.array(layout)

      return maze

   async def reset(self):
      """
       Reset the state of the environment.
       
       
       @return the current state of the environment ==> ['1.0' '1.0' 'N' '1.0' '1.0' '10.0'] // car position, car direction and the sensor readings
      """
      self.car_position = (1, 1)
      self.car_orientation = 'E'
      await self.update_sensor_readings()
      self.steps = 0
      self.previous_distance = 0
      self.previous_steps = 0
      self.visited_positions.clear() 
      self.visited_positions.add(self.car_position)
      return self.get_state()

   async def step(self, action):
      """
       Move car one step. This is a coroutine. It returns the state, reward and if the environment is done or not after this 'step' is done
       
       @param action - The action to take.
       
       @return Tuple of ( state reward done )
      """
      # Move the car to the right or forward
      if action == 0:
         # dont allow to move forward if the car is too close to a wall
         # Move forward or car if sensor readings front 4
         if self.sensor_readings['front'] >= 4:
            self.move_forward()
            if self.use_virtual_sensors == False:
               # self.move_car('forward')
               pass
      elif action == 1:
         self.turn_left()
         if self.use_virtual_sensors == False:
               # self.move_car('left')
               pass
      elif action == 2:
         self.turn_right()
         if self.use_virtual_sensors == False:
               # self.move_car('right')
               pass
         
      await self.update_sensor_readings()
      
      self.visited_positions.add(self.car_position)
      reward = self.compute_reward()
      self.steps += 1
      done = self.is_done()
         
      print('sensor readings: ', self.sensor_readings)
      return self.get_state(), reward, done
   
   def move_forward(self):
      """
       Move car forward in maze.
      """
      x, y = self.car_position
      # position of the car in the maze
      if self.car_orientation == 'N' and y > 0 and self.maze[y - 1][x] != 1:
         self.car_position = (x, y - 1)
      elif self.car_orientation == 'S' and y < self.maze_size_y - 1 and self.maze[y + 1][x] != 1:
         self.car_position = (x, y + 1)
      elif self.car_orientation == 'E' and x < self.maze_size_x - 1 and self.maze[y][x + 1] != 1:
         self.car_position = (x + 1, y)
      elif self.car_orientation == 'W' and x > 0 and self.maze[y][x - 1] != 1:
         self.car_position = (x - 1, y)
      
   def turn_left(self):
      """
       Turns the car to the left
      """
      orientations = ['N', 'W', 'S', 'E']
      idx = orientations.index(self.car_orientation)
      self.car_orientation = orientations[(idx + 1) % 4]

   def turn_right(self):
      """
       Turns the car to the right
      """
      orientations = ['N', 'E', 'S', 'W']
      idx = orientations.index(self.car_orientation)
      self.car_orientation = orientations[(idx + 1) % 4]
      
   def move_car(self, direction):
      """
       Make a call to the esp so it knows in which direction to move the motors
       
       @param direction - 'left'or'right'or'forward'
      """
      # Get the request to the ESP server.
      if direction == 'forward':
        esp_forward = f'http://{self.esp_ip}/forward'
        requests.get(esp_forward)
      elif direction == 'left':
        esp_left = f'http://{self.esp_ip}/left'
        requests.get(esp_left)
      elif direction == 'right':
        esp_right = f'http://{self.esp_ip}/right'
        requests.get(esp_right)
        
   async def update_sensor_readings(self):
      if self.use_virtual_sensors:
         await self.update_virtual_sensor_readings()
      else:
         await self.update_real_sensor_readings()
        
   ## for actual sensors
   async def update_real_sensor_readings(self):
        """
         Fetch sensor readings and update the sensor_readings
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_sensor_data(session, 'front'),
                self.fetch_sensor_data(session, 'left'),
                self.fetch_sensor_data(session, 'right')
            ]
            results = await asyncio.gather(*tasks)
            
            self.sensor_readings['front'], self.sensor_readings['left'], self.sensor_readings['right'] = results
                    # Update shared sensor data structure
            global sensor_data, sensor_data_lock
            with sensor_data_lock:
               sensor_data.update(self.sensor_readings)

   async def fetch_sensor_data(self, session, direction, retry_delay=0.5, max_retries=10):
      """
       Fetch sensor data from sensor. This is a function to be used in order to get sensor data from sensor
       
       @param session - connect to the session
       @param direction - Direction of sensor ( front / back ).
       @param retry_delay - Delay between retries in case of failure
       @param max_retries - Maximum number of retries. Default is 10
       
       @return Float value of sensor data from the HC-SR04
      """
      time.sleep(0.1)
      #   url = f'http://sensors:5500/sensor/{direction}'

      #   while True:  # Continue indefinitely until successful
      #       try:
      #           async with session.get(url, timeout=5) as response:
      #               if response.status == 200:
      #                   data = await response.text()
      #                   return float(data)
      #               else:
      #                   print(f"Error: Received status code {response.status} from sensor.")
      #       except Exception as e:
      #           print(f"Error: Failed to get sensor data from {url}. Exception: {e}")

      #       await asyncio.sleep(retry_delay)  # Wait before retrying
         
      try:
         sensor_front = DistanceSensor(echo=5, trigger=6)
         sensor_left = DistanceSensor(echo=17, trigger=27)
         sensor_right = DistanceSensor(echo=23, trigger=24)
      except:
         pass
      try:
         # distance between sensor and front direction
         if direction == "front":
               return float(sensor_front.distance * 100)
         elif direction == "left":
               return float(sensor_left.distance * 100)
         elif direction == "right":
               return float(sensor_right.distance * 100)
      except Exception as e:
         print(f"Error: {e}")
         return "Error reading sensor"


   async def update_virtual_sensor_readings(self):
      """
       Update the readings from the virtual sensors.
      """
      # Simple sensor implementation: counts steps to the nearest wall
      self.sensor_readings['front'] = self.distance_to_wall('front')
      self.sensor_readings['left'] = self.distance_to_wall('left')
      self.sensor_readings['right'] = self.distance_to_wall('right')
      
      global sensor_data, sensor_data_lock
      with sensor_data_lock:
         sensor_data.update(self.sensor_readings)

   def distance_to_wall(self, direction):
        """
         Returns the distance to the wall. It is based on the car orientation and the position of the car
         
         @param direction - 'front'or'right'or'front'
         
         @return The distance to the wall
        """
        x, y = self.car_position
        sensor_max_range = 255  # Maximum range of the ultrasonic sensor

        def calculate_distance(dx, dy):
            """
             Calculates the distance from the sensor. This is used to calculate the distance between the maze and the sensor
             
             @param dx - The distance in the x direction
             @param dy - The distance in the y direction ( 0 is top 1 is bottom )
             
             @return The distance between the sensor and the maze
            """
            distance = 0
            # This function is used to determine the distance between the sensor range and the sensor range.
            while 0 <= x + distance * dx < self.maze_size_x and \
                0 <= y + distance * dy < self.maze_size_y and \
                self.maze[y + distance * dy][x + distance * dx] != 1:
                distance += 1
                # This method is used to limit the sensor range.
                if distance > sensor_max_range:  # Limiting the sensor range
                    break
            return distance

        # Calculates the distance between the two directions.
        if direction == 'front':
            # Calculates the distance between the car_orientation and the orientation of the car.
            if self.car_orientation == 'N':
                distance = calculate_distance(0, -1)
            elif self.car_orientation == 'S':
                distance = calculate_distance(0, 1)
            elif self.car_orientation == 'E':
                distance = calculate_distance(1, 0)
            elif self.car_orientation == 'W':
                distance = calculate_distance(-1, 0)

        elif direction == 'left':
            # Calculates the distance between the car_orientation and the orientation of the car.
            if self.car_orientation == 'N':
                distance = calculate_distance(-1, 0)
            elif self.car_orientation == 'S':
                distance = calculate_distance(1, 0)
            elif self.car_orientation == 'E':
                distance = calculate_distance(0, -1)
            elif self.car_orientation == 'W':
                distance = calculate_distance(0, 1)

        elif direction == 'right':
            # Calculates the distance between the car_orientation and the orientation of the car.
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
        if any(self.sensor_readings[direction] == 0 for direction in ['front', 'left', 'right']):
            reward -= 20

        # Check if goal is reached
        # The reward of the goal.
        if self.car_position == self.goal:
            reward += 500
            # Additional penalty if it takes too many steps to reach the goal
            # If the reward is less than 1000 steps then the reward is decremented by 200.
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
        reward -= 2
        
        # Update the previous_distance for the next step
        self.previous_distance = distance_to_goal
        return reward

   def is_done(self):
      """
       Checks if the environment is done. It is done if it reaches the goal or takes more than 3000 steps or the car is out of bounds
      """
      #is done if it reaches the goal or goes out of bounds or takes more than 3000 steps
      return self.car_position == self.goal or self.steps > 3000 or self.car_position[0] < 0 or self.car_position[1] < 0 or self.car_position[0] > 11 or self.car_position[1] > 11
      
   def get_state(self):
      """
       Get the state of the car. This is a list of position sensor readings and orientation
       
       
       @return a combination of the car position, orientation, sensor readings in an array ['1.0' '1.0' 'N' '1.0' '1.0' '10.0'] // car position, car direction and the sensor readings
      """
      car_position = [float(coord) for coord in self.car_position]
      # self.update_sensor_readings()
      sensor_readings = [float(value) for value in self.sensor_readings.values()]
      
      state = car_position + [self.car_orientation] + sensor_readings
      
      # cast state to this ['1.0' '1.0' 'N' '1.0' '1.0' '10.0']
      state = np.array(state, dtype=str)
      
      #get the orientation and convert do label encoding
      # Set the state of the state to the next character in the state string.
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

   async def run_maze_env(self, replayCapacity=2000000):
      """
       Run RCMaze environment and return rewards.
       
       @param esp_ip - IP address of ESP to use //outdates use self.esp_ip
       @param replayCapacity - Size of replay memory
      """
      global maze_running, frame_queue
      
      
      # env = env
      state = await self.reset()

      self.init_opengl()
      self.run_opengl()
      
      REPLAY_MEMORY_CAPACITY = replayCapacity
      POSSIBLE_ACTIONS = self.possible_actions

      # create DQN agent
      test_agent = DQNAgent(replayCapacity=REPLAY_MEMORY_CAPACITY, input_shape=state.shape, output_shape=len(POSSIBLE_ACTIONS))


      from keras.models import load_model
      test_agent.policy_model = load_model('./models/DDQN_RCmaze_v2.h5')


      done = False
      rewards = []
      
      desired_fps = 2.0
      frame_duration = 1.0 / desired_fps

      last_time = time.time()
      done = False

      while maze_running:  # Use the maze_running flag to control the loop
         # This is a loop that is run in a loop.
         while not done:
            if not maze_running:
                print("Stopping maze environment...")
                break  # Exit the loop if maze_running is False
             
            current_time = time.time()
            elapsed = current_time - last_time
            # If the frame is full or full we can t wait for the frame to be processed.
            if elapsed >= frame_duration:
               
               glutMainLoopEvent()
               qValues = test_agent.policy_network_predict(np.array([state]))
               action = np.argmax(qValues[0])
               self.render()
               frame = self.capture_frame()
               try:
                  frame_queue.put_nowait(frame)  # Non-blocking put
               except queue.Full:
                  pass  # Skip if the queue is full
               
               state, reward, done = await self.step(action)
               rewards.append(reward)
               
               self.render()
               frame = self.capture_frame()
               try:
                  frame_queue.put_nowait(frame)  # Non-blocking put
               except queue.Full:
                  pass  # Skip if the queue is full
               
               last_time = current_time
            
               # prints the number of steps in the list
               if done:
                  print('done in ', len(rewards), 'steps')
                  maze_running = False
                  break
            
         print(sum(rewards))
         self.close_opengl()
      self.close_opengl()
      print("Maze environment stopped.")
      

   def init_opengl(self):
      """
       Initialize OpenGL and set up the RC Maze environment.
      """
      # Initialize OpenGL context
      glutInit()
      glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
      glutInitWindowSize(self.opelgl_window[0], self.opelgl_window[1])
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
      gluPerspective(90, 1, 0.1, 100)  # Adjust field of view angle, aspect ratio, near and far planes
      glMatrixMode(GL_MODELVIEW)
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      
      # Set the rendering function
      glutDisplayFunc(self.render)
      glutHideWindow()
      
   def run_opengl(self):
      """
       Run OpenGL.
      """
      # Set up the rendering context and callbacks
      # but do NOT call glutMainLoop()
      glutDisplayFunc(self.render)
      glutIdleFunc(self.render)  # Update rendering in idle time
      
   def third_person_view(self, CAMERA_DISTANCE=2.5,CAMERA_HEIGHT=1.5):
      """
       Returns the position of the third person view. It is possible to specify the distance and height of the camera
       
       @param CAMERA_DISTANCE - Distance from the camera to the car
       @param CAMERA_HEIGHT - Height of the camera above the car
      """
      camera_distance = CAMERA_DISTANCE # Distance from the camera to the car
      camera_height = CAMERA_HEIGHT  # Height of the camera above the car
      
      # Assuming self.car_orientation is 'N' and you want to be behind the car (to the 'S')
      # The camera orientation is N S S W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W W
      if self.car_orientation == 'N':  # Car is facing North
         camera_x = self.car_position[0]
         camera_y = (self.maze_size_y - self.car_position[1] - 1) - camera_distance  # Move camera to South
         camera_z = camera_height
      elif self.car_orientation == 'S':  # Car is facing South
         camera_x = self.car_position[0]
         camera_y = (self.maze_size_y - self.car_position[1] - 1) + camera_distance  # Move camera to North
         camera_z = camera_height
      elif self.car_orientation == 'E':  # Car is facing East
         camera_x = self.car_position[0] - camera_distance  # Move camera to West
         camera_y = self.maze_size_y - self.car_position[1] - 1
         camera_z = camera_height
      elif self.car_orientation == 'W':  # Car is facing West
         camera_x = self.car_position[0] + camera_distance  # Move camera to East
         camera_y = self.maze_size_y - self.car_position[1] - 1
         camera_z = camera_height

      # The point where the camera should be pointed: the car's position
      look_at_x = self.car_position[0]
      look_at_y = self.maze_size_y - self.car_position[1] - 1
      look_at_z = 1  # Assuming the car is at ground level (z=0)

      # Set up the camera
      glMatrixMode(GL_MODELVIEW)
      glLoadIdentity()
      gluLookAt(camera_x, camera_y, camera_z,  # Camera position (x, y, z)
               look_at_x, look_at_y, look_at_z,  # Look at position (x, y, z)
               0, 0, 2)  # Up vector (x, y, z), assuming Z is up
      
   def capture_frame(self):
      """
       Captures and returns the OpenGL frame. This is used to display the openGL window on the web app
       
       
       @return The PNG image of the current frame as a bytes
      """
      # Capture the OpenGL frame
      width, height = self.opelgl_window[0], self.opelgl_window[1]
      glPixelStorei(GL_PACK_ALIGNMENT, 1)
      data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
      image = Image.frombytes("RGB", (width, height), data)
      image = image.transpose(Image.FLIP_TOP_BOTTOM)

      buffer = io.BytesIO()
      image.save(buffer, format="PNG")
      buffer.seek(0)

      return buffer.getvalue()
   
   def render(self):
      """
       Render the maze and the car's readings to the OpenGL
      """
      # Clear buffers
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      # set third person view over the car (you can commit this for a top down view)
      self.third_person_view( CAMERA_DISTANCE=2.5,CAMERA_HEIGHT=3.5 )

      # Render the maze
      # draws the cube at the maze_size_y maze_size_x maze_size_y maze_size_x maze_size_y maze_size_x maze_size_y maze_size_x maze_size_y maze_size_x maze_size_y maze_size_x maze_size_x maze_size_y maze_size_x maze_size_x maze_size_y maze_size_x maze_size_y maze_size_y maze_size_y maze_size_y maze_size_y
      for y in range(self.maze_size_y):
         # draw the cube at the given x y
         for x in range(self.maze_size_x):
            # draw the cube at x y
            if self.maze[y][x] == 1:
               self.draw_cube(x, y, color=(0.5, 0.5, 0.5))
            elif (x, y) == self.goal:
               #set color to green
               self.draw_cube(x, y, color=(0.0, 1.0, 0.0))
                  
      # Render the car's sensor readings
      car_x, car_y = self.car_position
      #set sensor_color_directon with front being light blue, left being yellow and right being green
      sensor_colors = {'front': (0.0, 1.0, 1.0), 'left': (1.0, 1.0, 0.0), 'right': (0.0, 1.0, 0.0)}
      
      # Render the sensors
      # draws the sensor line for each sensor
      for sensor in ['front', 'left', 'right']:
         self.draw_sensor_line(car_x, car_y, self.sensor_readings[sensor], 
                                 sensor_colors[sensor], sensor)
         
      # Draw the car
      car_x, car_y = self.car_position
      # self.draw_car(car_x, car_y, color=(1.0, 0.0, 1.0))
      self.draw_car(car_x, car_y, [1, 0, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])  # pink body, grey wheels, dark grey window

      
      # Swap buffers
      glutSwapBuffers()
      
   def draw_cube(self, x, y, color):
      """
       Draw a cube at x y with the given color.
       
       @param x - x coordinate of the cube
       @param y - y coordinate of the cube ( 0 0 )
       @param color - color of the cube
      """
      # Set the color
      glColor3fv(color)
      # Draw a cube at position (x, y), flipping y coordinate
      glPushMatrix()
      glTranslate(x, self.maze_size_y - y - 1, 0)  # Adjust for vertical flipping
      glScalef(2, 2, 1)  # Adjust the size of your cube
      glutSolidCube(0.5)  # Adjust the size if needed
      glPopMatrix()
      
   def get_sensor_rotation_angle(self, sensor_orientation):
      """
       Get rotation angle based on sensor orientation and car orientation.
       
       @param sensor_orientation - Sensor orientation ( N S E W )
       
       @return Rotation angle ( degrees )
      """
      # print('direction: ', self.car_orientation)
      # Rotation logic based on car's orientation and sensor's relative position
      rotation_mapping = {
         'N': {'front': 90, 'left': 180, 'right': 0},
         'S': {'front': -90, 'left': 0, 'right': 180},
         'E': {'front': 0, 'left': 90, 'right': -90},
         'W': {'front': 180, 'left': -90, 'right': 90}
      }
      # Calculate total rotation angle
      return rotation_mapping[self.car_orientation][sensor_orientation]

   def draw_sensor_line(self, car_x, car_y, distance, color, sensor_orientation):
      """
       Draw a line on the maze that is acts like a sensor coming out of the car.
       
       @param car_x - x coordinate of the car's position
       @param car_y - y coordinate of the car's position
       @param distance - distance to the car from the walls
       @param color - color of the line 
       @param sensor_orientation - orientation of the sensor ( left / right / front )
      """
      close_threshold = 4
      glColor3fv((1.0, 0.0, 0.0) if distance <= close_threshold else color)

      # Calculate rotation based on car's and sensor's orientation
      rotation_angle = self.get_sensor_rotation_angle(sensor_orientation)

      glPushMatrix()
      glTranslate(car_x, self.maze_size_y - car_y - 1, 0.5)  # Adjust for vertical flipping
      glRotatef(rotation_angle, 0, 0, 1)
      glRotatef(90, 0, 1, 0)

      # Draw sensor line
      # distance = min(distance, 0.5)  # Cap distance
      glutSolidCylinder(0.05, 0.5, 5, 5)

      glPopMatrix()

   # def draw_car(self, x, y, color):
   #    """
   #    Draw a car at x y.
      
   #    @param x - x coordinate of the car
   #    @param y - y coordinate of the car 
   #    @param color - color of the car
   #    """
   #    # Set the color
   #    glColor3fv(color)

   #    # Adjust for vertical flipping
   #    car_y = self.maze_size_y - y - 1

   #    # Draw the main body of the car as a rectangle
   #    glPushMatrix()
   #    glTranslate(x, car_y, 0)
   #    glScalef(1.5, 0.8, 1)  # Adjust for the size of the car body
   #    glutSolidCube(0.5)
   #    glPopMatrix()

   #    # Draw the wheels with rotation based on car's orientation
   #    wheel_radius = 0.1
   #    wheel_width = 0.1

   #    # Define wheel positions relative to the car
   #    wheel_positions = [(x-0.5, car_y-0.3), (x+0.5, car_y-0.3), (x-0.5, car_y+0.3), (x+0.5, car_y+0.3)]

   #    # Apply rotation to each wheel based on car orientation
   #    for wheel_x, wheel_y in wheel_positions:
   #       glPushMatrix()
   #       glTranslate(wheel_x, wheel_y, 0)

   #       # Get rotation angle for the wheels
   #       rotation_angle = self.get_sensor_rotation_angle('front')  # Assuming you want the wheels to align with the front of the car
   #       glRotatef(rotation_angle, 0, 0, 1)
         
   #       glRotatef(90, 1, 0, 0)  # Orient the wheel correctly
   #       glutSolidTorus(wheel_radius, wheel_width, 10, 10)
   #       glPopMatrix()
   def draw_car(self, x, y, body_color, wheel_color, window_color):
      """
      Draw a car with a given orientation and different colors for body, wheels, and window.
      
      @param x - x coordinate of the car
      @param y - y coordinate of the car
      @param body_color - color of the car's body
      @param wheel_color - color of the wheels
      @param window_color - color of the window
      """
      
      #car orientation
      car_orientation = self.car_orientation
      
      # Calculate rotation based on car's orientation
      rotation_angle = self.get_sensor_rotation_angle('front')

      # Set the color for the car body and draw it with the correct orientation
      glColor3fv(body_color)
      glPushMatrix()
      glTranslate(x, self.maze_size_y - y - 1, 0)
      glRotatef(rotation_angle, 0, 0, 1)  # Rotate the car body
      glScalef(1.0, 0.5, 0.3)  # Car body size
      glutSolidCube(1)
      glPopMatrix()

      # Draw the wheels in a different color
      glColor3fv(wheel_color)
      wheel_radius = 0.1
      wheel_width = 0.05
      wheel_positions = [(x-0.55, y-0.25), (x+0.55, y-0.25),
                        (x-0.55, y+0.25), (x+0.55, y+0.25)]
      for wheel_x, wheel_y in wheel_positions:
         glPushMatrix()
         glTranslate(wheel_x, self.maze_size_y - wheel_y - 1, -0.1)
         glRotatef(rotation_angle, 0, 0, 1)  # Rotate the wheel with the car
         glRotatef(90, 1, 0, 0)  # Orient the wheel correctly
         glutSolidTorus(wheel_width, wheel_radius, 10, 10)
         glPopMatrix()

      # Add a smaller rectangle on top as a window, in a different color
      glColor3fv(window_color)
      glPushMatrix()
      glTranslate(x, self.maze_size_y - y - 1, 0.15)  # Adjust for the height of the window
      glRotatef(rotation_angle, 0, 0, 1)  # Rotate the window with the car
      glScalef(0.4, 0.2, 0.1)  # Window size
      glutSolidCube(1)
      glPopMatrix()



   def close_opengl(self):
      """
       Close the OpenGL context and clean up the GLUT window. This is called when the window is no longer needed
      """
      # Close the OpenGL context
      glutLeaveMainLoop()
      # glutDestroyWindow(glutGetWindow())

class DQNAgent:
   def __init__(self, replayCapacity, input_shape, output_shape, learning_rate=0.001, discount_factor=0.90):
      """
       Initialize the policy and target network.
       
       @param replayCapacity - The capacity of the replay network
       @param input_shape - The shape of the input to the policy
       @param output_shape - The shape of the output to the policy
       @param learning_rate - The learning rate to be used for learning
       @param discount_factor - The discount factor to be
      """
      self.capacity = replayCapacity
      self.memory = collections.deque(maxlen=self.capacity)
      self.learning_rate = learning_rate
      self.discount_factor = discount_factor
      self.input_shape = input_shape
      self.output_shape = output_shape
      self.policy_model = self.buildNetwork()
      self.target_model = self.buildNetwork()
      self.target_model.set_weights(self.policy_model.get_weights())


   def addToReplayMemory(self, step):
      """
       Add a step to the replay memory. This is used to track the progress of the step that was replayed.
       
       @param step - The step to add to the replay memory. Note that it is up to the caller to make sure this is indeed the last step
      """
      self.step = step
      self.memory.append(self.step)

   def sampleFromReplayMemory(self, batchSize):
      """
       Sample from replay memory. This is a re - implementation of L { sampleFromReplayMemory } to avoid having to do a lot of random sampling in the case of large batches.
       
       @param batchSize - The number of items to sample from the replay memory.
       
       @return A batch of items from the replay memory that has been filled
      """
      self.batchSize = batchSize
      # Sample a random number of samples from the memory.
      if self.batchSize > len(self.memory):
         self.populated = False
         return self.populated
      else:
         return random.sample(self.memory, self.batchSize)

   def buildNetwork(self):
      """
       Builds and compiles the neural network. This is a helper to allow subclasses to override the build method in order to add more features such as learning rate and loss.
       
       
       @return A Sequential object that can be used to train the
      """
      model = Sequential()
      model.add(Dense(32, input_shape=self.input_shape, activation='relu'))
      model.add(Dense(64, activation='relu'))
      model.add(Dense(32, activation='relu'))
      model.add(Dense(self.output_shape, activation='linear'))
      model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['MeanSquaredError'])
      return model

   def policy_network_fit(self, batch, batch_size):
      """
       Fit the policy network to the batch. Predict Q - values for starting state using the policy network Select the best action for the next state using the target network Update Q - values for actions taken
       
       @param batch - List of ( states actions rewards next_states dones )
       @param batch_size - Size of the batch to be used
      """
      states, actions, rewards, next_states, dones = zip(*batch)
      states = np.array(states)
      next_states = np.array(next_states)
      # Predict Q-values for starting state using the policy network
      q_values = self.policy_model.predict(states)
      # Predict Q-values for next state using the policy network
      q_values_next_state_policy = self.policy_model.predict(next_states)
      # Select the best action for the next state using the policy network
      best_actions = np.argmax(q_values_next_state_policy, axis=1)
      # Predict Q-values for next state using the target network
      q_values_next_state_target = self.target_model.predict(next_states)
      # Update Q-values for actions taken
      # Update the q_values and actions for each batch.
      for i in range(batch_size):
         # Update the q_values for the i th dones.
         if dones[i]:
            q_values[i, actions[i]] = rewards[i]
         else:
            # Double DQN update rule
            q_values[i, actions[i]] = rewards[i] + self.discount_factor * q_values_next_state_target[i, best_actions[i]]
      # Train the policy network
      self.policy_model.fit(states, q_values, batch_size=batch_size, verbose=0)

   def policy_network_predict(self, state):
      """
       Predict Q - Policy for a given state. This is a wrapper around policy_model. predict
       
       @param state - A dictionary of state to predict Q - Policy for
       
       @return A numpy array of shape ( N_samples N
      """
      self.state = state
      self.qPolicy = self.policy_model.predict(self.state)
      return self.qPolicy

   def target_network_predict(self, state):
      """
       Predict the target for a given state using the target network. This is a low - level method and should not be called directly.
       
       @param state - A dictionary containing the state to predict. Must be an instance of : py : class : ` qutip. states. ChainState `
       
       @return A
      """
      self.state = state
      self.qTarget = self.target_model.predict(self.state)
      return self.qTarget

   def update_target_network(self):
      """
       Update the target network based on the policy model. This is called when the policy is changed
      """
      self.target_model.set_weights(self.policy_model.get_weights())



app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/*": {"origins": "*"}}


frame_queue = queue.Queue(maxsize=34)  # maxsize=1 to avoid holding too many frames

maze_thread = None
maze_running = False

# Shared data structure and lock
sensor_data = {"front": 0, "left": 0, "right": 0}
sensor_data_lock = Lock()

def generate_frames():
   """
    Generate frames from mjpegstreamer and return them as a list of lines
   """
   cap = cv2.VideoCapture("http://mjpgstreamer:8080/?action=stream") # this is for the ESP32-CAM when using docker
   #cap = cv2.VideoCapture(0) # for webcam access
   # Yields frame content from the current buffer.
   while True:
      success, frame = cap.read()
      # Yields frame frame in bytes.
      if not success:
         break
      else:
         ret, buffer = cv2.imencode('.jpg', frame)
         frame = buffer.tobytes()
         yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_flask_app():
   app.run(debug=True, use_reloader=False, threaded=True, host='0.0.0.0', port=5000)
   
def run_async_function(func):
   """
    Run a function in an asyncio event loop. This is for tests that need to run asynchronous functions.
    
    @param func - The function to run in the event loop. It must take no arguments
   """
   loop = asyncio.new_event_loop()
   asyncio.set_event_loop(loop)
   loop.run_until_complete(func)
   loop.close()
        
@app.route('/frame')
def frame():
    """
     Get a frame from the queue and return it as a PNG. This is a blocking call so it will block until something is available to be returned.
     
     
     @return Response object with status 200 or 503 if queue is full
    """
    try:
      global frame_queue
      image_data = frame_queue.get_nowait()  # Non-blocking get
      print("Sending image data with status 200")
   
      return Response(image_data, mimetype='image/png', status=200)
    except queue.Empty:
      print("Queue empty, sending status 503")
      return Response(status=503)  # Service Unavailable

@app.route("/get_my_ip", methods=["GET"])
def get_my_ip():
   """
    Get IP address of user. This is used to check if user is running host camera or not.
    
    
    @return JSON with ip address of user or 404 if not
   """
   ## add something to get host camera??? 
   return jsonify({'ip': request.remote_addr}), 200

@app.route("/get_sensor_readings")
def get_sensor_readings():
   """
    Get a copy of the sensor data and return it as a JSON object.
    
    
    @return A JSON object containing the readings of the sensor
   """
   global sensor_data, sensor_data_lock
   with sensor_data_lock:
      # Make a copy of the data to avoid holding the lock while JSONifying
      data_copy = sensor_data.copy()
   return jsonify(data_copy)

@app.route('/start-maze/<use_virtual>/<esp_ip>')
def start_maze(use_virtual,esp_ip):
   """
    Starts maze if it is not already running. This is a blocking call
    
    @param use_virtual - choose if you want to run the simulation or the real rc car // true for virtual and false for real
    @param esp_ip - ESP IP to use for the maze
    
    @return String describing success or failure of the maze startup
   """
   global maze_thread, maze_running
   # This method is used to start the maze thread
   if maze_thread is None or not maze_thread.is_alive():
      
      #convert js true / false to python true false
      if use_virtual == 'true':
         use_virtual = True
      elif use_virtual == 'false':
         use_virtual = False
      
      env = RCMazeEnv(esp_ip=esp_ip, use_virtual_sensors=use_virtual)
      # Use a lambda function to pass arguments to the async function wrapper
      maze_running = True
      maze_thread = threading.Thread(target=lambda: run_async_function(env.run_maze_env()))
      maze_thread.start()
      return "Maze started with ESP IP: " + esp_ip
   else:
      return "Maze is already running"
   
@app.route('/close-maze')
def close_maze():
    """
    Stops the maze if it is running.
    
    @return String describing success or failure of the maze closure
    """
    global maze_thread
    if maze_thread is not None and maze_thread.is_alive():
        global maze_running
        maze_running = False  # This flag should be checked within the maze thread to stop its operation
        
        maze_thread.join()  # Wait for the thread to finish
        maze_thread = None  # Reset the thread variable
        return "Maze stopped successfully"
    else:
        return "Maze is not running"


@app.route('/')
def index():
   """
    The index page of the web application. This is used to display the home page of the web application.
    
    
    @return The template that renders the index page of the web application
   """
   return render_template('index.html')        

@app.route('/video')
def video():
   """
    Generate and stream a video. This is a wrapper around : func : ` generate_frames ` that creates a thread to stream the video.
    
    
    @return A : class : ` twisted. web. server. Response ` object
   """
   # Create and start the video thread
   video_thread = threading.Thread(target=generate_frames)
   video_thread.start()

   # Return the response to stream the video
   return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# This is a wrapper around run_main__ to create a new thread and start the Flask app in its own thread.
if __name__ == "__main__":
   # maze_thread = threading.Thread(target=run_maze_env)
   # maze_thread.start()

   # Create and start the Flask app in its own thread
   flask_thread = threading.Thread(target=run_flask_app)
   flask_thread.daemon = True
   flask_thread.start()

   flask_thread.join()
   # maze_thread.join()
   
   
   

