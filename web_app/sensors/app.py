from flask import Flask, Response, render_template
import cv2
from gpiozero import DistanceSensor
# from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import os
from signal import pause

app = Flask(__name__)

@app.route('/')
def index():
   return "hello_world"


@app.route('/sensor/<direction>')
def get_sensor_value(direction):
      try:
         sensor_front = DistanceSensor(echo=5, trigger=6)
         sensor_left = DistanceSensor(echo=17, trigger=27)
         sensor_right = DistanceSensor(echo=23, trigger=24)
      except:
         pass
      try:
         if direction == "front":
            distance = float(sensor_front.distance * 100)
            # pause()
            return f"{distance}"
         elif direction == "left":
            distance = float(sensor_left.distance * 100)
            # pause()
            return f"{distance}"
         elif direction == "right":
            distance = float(sensor_right.distance * 100)
            # pause()
            return f"{distance}"
      except Exception as e:
         print(f"Error: {e}")
         return "Error reading sensor"
      pause()



if __name__ == '__main__':
   app.run(debug=True, host="0.0.0.0", port='5500')
   
   
   
