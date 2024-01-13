from flask import Flask, send_file, Response

app = Flask(__name__)
env = RCMazeEnv()
env.init_opengl()

@app.route('/frame')
def frame():
    image_data = env.capture_frame()
    return Response(image_data, mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
