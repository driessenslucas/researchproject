from flask import Flask, Response, render_template
import cv2
from gpiozero import DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import os

app = Flask(__name__)

#get env variable factory = PiGPIOFactory(host='${env.host_IP}')
host = os.getenv("HOST_IP", "192.168.0.25") 
factory = PiGPIOFactory(host=host)

@app.route('/')
def index():
   return "hello_world"

@app.route('/sensor/<direction>')
def get_sensor_value(direction):
   if direction == "front":
      sensor_front = DistanceSensor(echo=5, trigger=6, pin_factory=factory)
      print('Distance: ', sensor_front.distance )
      return f"{sensor_front.distance * 100 }"
   elif direction == "left":
      sensor_left = DistanceSensor(echo=17, trigger=27, pin_factory=factory)
      print('Distance: ', sensor_left .distance )
      return f"{sensor_left.distance  * 100}"
   elif direction == "right":
      sensor_right = DistanceSensor(echo=24, trigger=23, pin_factory=factory)
      print('Distance: ', sensor_right.distance )
      return f"{sensor_right.distance * 100 }"


if __name__ == '__main__':
   app.run(debug=True, host="0.0.0.0", port='5500')