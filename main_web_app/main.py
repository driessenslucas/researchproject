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
   def __init__(self, maze_size_x=12, maze_size_y=12, esp_ip='192.168.0.27'):
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
      self.reset()

         
   def generate_maze(self):
      # For simplicity, create a static maze with walls
      # '1' represents a wall, and '0' represents an open path
      # maze = np.zeros((self.maze_size_y, self.maze_size_x), dtype=int)
      # Add walls to the maze (this can be customized)

      
      layout = [
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
         [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
         [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
         [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
         [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
         [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
      
   
      maze = np.array(layout)

      return maze

   def reset(self):
      self.car_position = (1, 1)
      self.car_orientation = 'E'
      self.update_sensor_readings()
      self.steps = 0
      self.previous_distance = 0
      self.previous_steps = 0
      self.visited_positions.clear()  # Clear the visited positions
      self.visited_positions.add(self.car_position)
      return self.get_state()

   async def step(self, action):
      # time.sleep(1)
      if action == 0:
         self.move_forward()
         self.move_car('forward')
      elif action == 1:
         self.turn_left()
         self.move_car('left')
      elif action == 2:
         self.turn_right()
         self.move_car('right')
      
      await self.update_sensor_readings()
      
      self.visited_positions.add(self.car_position)
      reward = self.compute_reward()
      self.steps += 1
      done = self.is_done()
         
      print('sensor readings: ', self.sensor_readings)
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
      
   def move_car(self, direction):
      if direction == 'forward':
        #call to esp
        url = f'http://{self.esp_ip}/forward'
        print(url)
        requests.get(url)
      elif direction == 'left':
        # do call to http://{self.esp_ip}/left
        #print('left')
        #call to esp
        url = f'http://{self.esp_ip}/left'
        requests.get(url)
      elif direction == 'right':
        # do call to http://{self.esp_ip}/right
        #print('right')
        #call to esp
        url = f'http://{self.esp_ip}/right'
        requests.get(url)
        
   async def update_sensor_readings(self):
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

   async def fetch_sensor_data(self, session, direction, retry_delay=1):
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
      
         sensor_front = DistanceSensor(echo=5, trigger=6)
         sensor_left = DistanceSensor(echo=17, trigger=27)
         sensor_right = DistanceSensor(echo=23, trigger=24)
         
         if direction == "front":
               return f"{float(sensor_front.distance * 100)}"
         elif direction == "left":
               return f"{float(sensor_left.distance * 100)}"
         elif direction == "right":
               return f"{float(sensor_right.distance * 100)}"
         pause()


   # def update_sensor_readings(self):
   #    # Simple sensor implementation: counts steps to the nearest wall
   #    self.sensor_readings['front'] = self.distance_to_wall('front')
   #    self.sensor_readings['left'] = self.distance_to_wall('left')
   #    self.sensor_readings['right'] = self.distance_to_wall('right')

   # def distance_to_wall(self, direction):
   #      x, y = self.car_position
   #      sensor_max_range = 255  # Maximum range of the ultrasonic sensor

   #      def calculate_distance(dx, dy):
   #          distance = 0
   #          while 0 <= x + distance * dx < self.maze_size_x and \
   #              0 <= y + distance * dy < self.maze_size_y and \
   #              self.maze[y + distance * dy][x + distance * dx] != 1:
   #              distance += 1
   #              if distance > sensor_max_range:  # Limiting the sensor range
   #                  break
   #          return distance

   #      if direction == 'front':
   #          if self.car_orientation == 'N':
   #              distance = calculate_distance(0, -1)
   #          elif self.car_orientation == 'S':
   #              distance = calculate_distance(0, 1)
   #          elif self.car_orientation == 'E':
   #              distance = calculate_distance(1, 0)
   #          elif self.car_orientation == 'W':
   #              distance = calculate_distance(-1, 0)

   #      elif direction == 'left':
   #          if self.car_orientation == 'N':
   #              distance = calculate_distance(-1, 0)
   #          elif self.car_orientation == 'S':
   #              distance = calculate_distance(1, 0)
   #          elif self.car_orientation == 'E':
   #              distance = calculate_distance(0, -1)
   #          elif self.car_orientation == 'W':
   #              distance = calculate_distance(0, 1)

   #      elif direction == 'right':
   #          if self.car_orientation == 'N':
   #              distance = calculate_distance(1, 0)
   #          elif self.car_orientation == 'S':
   #              distance = calculate_distance(-1, 0)
   #          elif self.car_orientation == 'E':
   #              distance = calculate_distance(0, 1)
   #          elif self.car_orientation == 'W':
   #              distance = calculate_distance(0, -1)

   #      # Normalize the distance to a range of 0-1
   #      normalized_distance = distance / sensor_max_range
   #      normalized_distance = max(0, min(normalized_distance, 1))

   #      return normalized_distance * 1000
   
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
      # self.update_sensor_readings()
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
      gluPerspective(90, 1, 0.1, 100)  # Adjust field of view angle, aspect ratio, near and far planes
      glMatrixMode(GL_MODELVIEW)
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      
      # Set the rendering function
      glutDisplayFunc(self.render)
      glutHideWindow()
      
   def run_opengl(self):
      # Set up the rendering context and callbacks
      # but do NOT call glutMainLoop()
      glutDisplayFunc(self.render)
      glutIdleFunc(self.render)  # Update rendering in idle time
      
        
   def third_person_view(self):
      camera_distance = 2.5 # Distance from the camera to the car
      camera_height = 1.5  # Height of the camera above the car
      
      # Assuming self.car_orientation is 'N' and you want to be behind the car (to the 'S')
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
        # Capture the OpenGL frame
        width, height = 1200, 1200
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (width, height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return buffer.getvalue()
   

   def render(self):
      # Clear buffers
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

      # self.third_person_view()

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
      #set sensor_color_directon with front being light blue, left being yellow and right being green
      sensor_colors = {'front': (0.0, 1.0, 1.0), 'left': (1.0, 1.0, 0.0), 'right': (0.0, 1.0, 0.0)}
      
      # Render the sensors
      for sensor in ['front', 'left', 'right']:
         self.draw_sensor_line(car_x, car_y, self.sensor_readings[sensor], 
                                 sensor_colors[sensor], sensor)
         
      # Draw the car
      car_x, car_y = self.car_position
      self.draw_car(car_x, car_y, color=(1.0, 0.0, 1.0))
      
      # Swap buffers
      glutSwapBuffers()
      

   def draw_cube(self, x, y, color):
      # Set the color
      glColor3fv(color)

      # Draw a cube at position (x, y), flipping y coordinate
      glPushMatrix()
      glTranslate(x, self.maze_size_y - y - 1, 0)  # Adjust for vertical flipping
      glScalef(2, 2, 1)  # Adjust the size of your cube
      glutSolidCube(0.5)  # Adjust the size if needed
      glPopMatrix()
      
   def get_sensor_rotation_angle(self, sensor_orientation):
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

   def draw_car(self, x, y, color):
      # Set the color
      glColor3fv(color)

      # Draw a cube at position (x, y), flipping y coordinate
      glPushMatrix()
      glTranslate(x, self.maze_size_y - y - 1, 0)  # Adjust for vertical flipping
      glScalef(1, 1, 1)  # Adjust the size of your cube
      glutSolidCube(0.5)  # Adjust the size if needed
      glPopMatrix()

   def close_opengl(self):
      # Close the OpenGL context
      glutLeaveMainLoop()
      glutDestroyWindow(glutGetWindow())

class DQNAgent:
    def __init__(self, replayCapacity, input_shape, output_shape, learning_rate=0.001, discount_factor=0.90):
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
      model.add(Dense(32, input_shape=self.input_shape, activation='relu'))
      model.add(Dense(64, activation='relu'))
      model.add(Dense(32, activation='relu'))
      model.add(Dense(self.output_shape, activation='linear'))
      model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['MeanSquaredError'])
      return model

    def policy_network_fit(self, batch, batch_size):
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
            for i in range(batch_size):
                if dones[i]:
                    q_values[i, actions[i]] = rewards[i]
                else:
                    # Double DQN update rule
                    q_values[i, actions[i]] = rewards[i] + self.discount_factor * q_values_next_state_target[i, best_actions[i]]

            # Train the policy network
            self.policy_model.fit(states, q_values, batch_size=batch_size, verbose=0)

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



def generate_frames():
    cap = cv2.VideoCapture("http://mjpgstreamer:8080/?action=stream") # this is for the ESP32-CAM when using docker
   #  cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)
# allow origin from anywhere
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/*": {"origins": "*"}}


frame_queue = queue.Queue(maxsize=34)  # maxsize=1 to avoid holding too many frames

def run_flask_app():
   app.run(debug=True, use_reloader=False, threaded=True, host='0.0.0.0', port=5000)
        
        
@app.route('/frame')
def frame():
    try:
        image_data = frame_queue.get_nowait()  # Non-blocking get
        print("Sending image data with status 200")
    
        return Response(image_data, mimetype='image/png', status=200)
    except queue.Empty:
        print("Queue empty, sending status 503")
        return Response(status=503)  # Service Unavailable


@app.route("/get_my_ip", methods=["GET"])
def get_my_ip():
   ## add something to get host camera??? 
   return jsonify({'ip': request.remote_addr}), 200

maze_thread = None

# Shared data structure and lock
sensor_data = {"front": 0, "left": 0, "right": 0}
sensor_data_lock = Lock()

# @app.route("/get_sensor_readings")
# def get_sensor_readings():
#     global sensor_data, sensor_data_lock
#     with sensor_data_lock:
#         # Make a copy of the data to avoid holding the lock while JSONifying
#         data_copy = sensor_data.copy()
#     return jsonify(data_copy)


@app.route('/start-maze/<esp_ip>')
def start_maze(esp_ip):
    global maze_thread
    if maze_thread is None or not maze_thread.is_alive():
        # Use a lambda function to pass arguments to the target function
        maze_thread = threading.Thread(target=lambda: run_maze_env(esp_ip))
        maze_thread.start()
        return "Maze started with ESP IP: " + esp_ip
    else:
        return "Maze is already running"


@app.route('/')
def index():
   return render_template('index.html')        


@app.route('/video')
def video():
    # Create and start the video thread
    video_thread = threading.Thread(target=generate_frames)
    video_thread.start()

    # Return the response to stream the video
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_maze_env(esp_ip):
   env = RCMazeEnv(esp_ip=esp_ip)
   state = env.reset()

   env.init_opengl()
   env.run_opengl()
   
   REPLAY_MEMORY_CAPACITY = 20000
   POSSIBLE_ACTIONS = env.possible_actions

   # create DQN agent
   test_agent = DQNAgent(replayCapacity=REPLAY_MEMORY_CAPACITY, input_shape=state.shape, output_shape=len(POSSIBLE_ACTIONS))


   from keras.models import load_model
   test_agent.policy_model = load_model('./models/DDQN_RCmaze.h5')


   done = False
   rewards = []
   
   desired_fps = 4.0
   frame_duration = 1.0 / desired_fps

   last_time = time.time()
   done = False


   while not done:
      current_time = time.time()
      elapsed = current_time - last_time
      if elapsed >= frame_duration:
         
         glutMainLoopEvent()
         qValues = test_agent.policy_network_predict(np.array([state]))
         action = np.argmax(qValues[0])
         env.render()
         frame = env.capture_frame()
         try:
            frame_queue.put_nowait(frame)  # Non-blocking put
         except queue.Full:
            pass  # Skip if the queue is full
         
         state, reward, done = asyncio.run(env.step(action))
         rewards.append(reward)
         
         env.render()
         frame = env.capture_frame()
         try:
            frame_queue.put_nowait(frame)  # Non-blocking put
         except queue.Full:
            pass  # Skip if the queue is full
         
         # last_time = current_time
       
         if done:
            print('done in ', len(rewards), 'steps')
            break
   # env.close()
   print(sum(rewards))
   env.close_opengl()
   # env.close_pygame()



# set main
if __name__ == "__main__":
   # maze_thread = threading.Thread(target=run_maze_env)
   # maze_thread.start()

   # Create and start the Flask app in its own thread
   flask_thread = threading.Thread(target=run_flask_app)
   flask_thread.daemon = True
   flask_thread.start()

   flask_thread.join()
   # maze_thread.join()
   
   
   

