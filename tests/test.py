import requests

import json

url = "http://192.168.0.7/sensors/"
response = requests.get(url)
if response.status_code == 200:
    try:
        sensor_data = response.text
        #GET FIRST LINE
        sensor_data = sensor_data.split('\n', 1)[0]
        #parse to json
        sensor_data = json.loads(sensor_data)
        print(sensor_data)  # Print or process your sensor data
        
        #sensor left:
        print(sensor_data['left'])
        #sensor right:
        print(sensor_data['right'])
        #sensor front:
        print(sensor_data['front'])
    except ValueError:
        print("Error: Received non-JSON response")
else:
    print(f"Error: Received status code {response.status_code}")