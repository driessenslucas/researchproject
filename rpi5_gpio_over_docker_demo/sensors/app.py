from flask import Flask, Response, render_template
from gpiozero import DistanceSensor
from time import sleep
from signal import pause
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas

app = Flask(__name__)

hc_sr04 = DistanceSensor(echo=5, trigger=6)
device = ssd1306(i2c(port=1, address=0x3c), width=128, height=64, rotate=0)

# set the contrast to minimum.
device.contrast(1)

def show_distance_on_screen(distance):
   print(f"Distance: {distance:.2f} cm")
   message = 'Raspberry Pi'
   text_size = draw.textsize(message)
   draw.text((device.width - text_size[0], (device.height - text_size[1]) // 2), message, fill='white')
   print(f"Distance: {distance:.2f} cm")

@app.route('/')
def index():
   return "hello_world"

@app.route('/sensor')
def get_sensor_value():
      try:
         distance = hc_sr04.distance * 100
         show_distance_on_screen(distance)
         sleep(0.1)
         return distance

      except Exception as e:
         print(f"Error: {e}")
         return "Error reading sensor"

if __name__ == '__main__':
   app.run(debug=True, host="0.0.0.0", port='5500')
   
   
   
