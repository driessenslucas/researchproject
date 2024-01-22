from flask import Flask, Response, render_template
import cv2
from gpiozero import DistanceSensor
from time import sleep
import os
from signal import pause

app = Flask(__name__)

@app.route('/')
def index():
   return "hello_world"

def map_distance(distance):
    if distance < 25:
        # No change for distances less than 20 cm
        return distance
    else:
        return 25 + (distance - 25) * 0.5



@app.route('/sensor/<direction>')
def get_sensor_value(direction):
      try:
         sensor_front = DistanceSensor(echo=5, trigger=6)
         sensor_left = DistanceSensor(echo=17, trigger=27)
         sensor_right = DistanceSensor(echo=23, trigger=24)
      
         sleep(0.1)
         try:
            distance = 0
            if direction == "front":
               distance = sensor_front.distance * 100
            elif direction == "left":
               distance = sensor_left.distance * 100
            elif direction == "right":
               distance = sensor_right.distance * 100
               
            mapped_distance = map_distance(distance)

            return f"{mapped_distance:.2f}"
         except Exception as e:
            print(f"Error: {e}")
            return "Error reading sensor"

      except Exception as e:
         print(f"Error: {e}")
         return "Error reading sensor"
         
      pause()



if __name__ == '__main__':

   app.run(debug=True, host="0.0.0.0", port='5500')
   
   
   
