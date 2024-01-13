import requests
response = requests.get('http://192.168.0.25:5000/sensor')
print(response.text)