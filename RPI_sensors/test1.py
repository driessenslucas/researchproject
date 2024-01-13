# from gpiozero import DistanceSensor
# from time import sleep

# sensor = DistanceSensor(echo=5, trigger=6)
# while True:
#    print('Distance: ', sensor.distance )
#    sleep(1)
   
   
from flask import Flask, Response, render_template
import cv2
from gpiozero import DistanceSensor

app = Flask(__name__)


@app.route('/')
def index():
   return "hello_world"

@app.route('/sensor')
def get_sensor_value():
   from gpiozero import DistanceSensor
   from time import sleep

   sensor = DistanceSensor(echo=5, trigger=6)
   print('Distance: ', sensor.distance )
   
   return f"{sensor.distance}"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")