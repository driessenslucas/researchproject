# from gpiozero import DistanceSensor
# from time import sleep

# sensor = DistanceSensor(echo=5, trigger=6)
# while True:
#    print('Distance: ', sensor.distance )
#    sleep(1)
   
   
from flask import Flask, Response, render_template
import cv2
from gpiozero import DistanceSensor
from time import sleep

app = Flask(__name__)


@app.route('/')
def index():
   return "hello_world"

@app.route('/sensor/<direction>')
def get_sensor_value(direction):
   if direction == "front":
      sensor_front = DistanceSensor(echo=5, trigger=6)
      print('Distance: ', sensor_front.distance )
      return f"{sensor_front.distance * 100 }"
   elif direction == "left":
      sensor_left = DistanceSensor(echo=17, trigger=27)
      print('Distance: ', sensor_left .distance )
      return f"{sensor_left.distance  * 100}"
   elif direction == "right":
      sensor_right = DistanceSensor(echo=24, trigger=23)
      print('Distance: ', sensor_right.distance )
      return f"{sensor_right.distance * 100 }"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")