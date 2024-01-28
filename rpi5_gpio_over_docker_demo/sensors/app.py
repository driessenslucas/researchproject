from flask import Flask, Response, render_template
from gpiozero import DistanceSensor
from time import sleep
from signal import pause

app = Flask(__name__)


@app.route('/')
def index():
   return "hello_world 2"

@app.route('/sensor/')
def get_sensor_value():
      try:
        hc_sr04 = DistanceSensor(echo=27, trigger=17)
        distance = hc_sr04.distance * 100
        sleep(0.1)
        return f"distance: {distance:.2f} cm"

      except Exception as e:
         print(f"Error: {e}")
         return "Error reading sensor"

if __name__ == '__main__':
  app.run(debug=True, host="0.0.0.0", port='5500')
