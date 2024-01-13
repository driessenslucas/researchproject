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

@app.route('/sensor/{direction}')
def get_sensor_value(direction):
   if direction == "front":
      sensor = DistanceSensor(echo=5, trigger=6)
      print('Distance: ', sensor.distance )
      return f"{sensor.distance}"
   elif direction == "left":
      sensor = DistanceSensor(echo=17, trigger=27)
      print('Distance: ', sensor.distance )
      return f"{sensor.distance}"
   elif direction == "right":
      sensor = DistanceSensor(echo=23, trigger=24)
      print('Distance: ', sensor.distance )
      return f"{sensor.distance}"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")