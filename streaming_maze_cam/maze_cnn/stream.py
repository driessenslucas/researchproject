from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture("http://mjpgstreamer:8080/?action=stream")

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Route to render the HTML template
    return render_template('index.html')

## add something for pygame display

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")


## this is for later:
# from flask import Flask, Response
# import cv2
# import numpy as np
# import your_cnn_model  # Replace with your actual CNN model import

# app = Flask(__name__)

# # Load your CNN model
# model = your_cnn_model.load_model('path_to_your_model')

# def process_frame(frame):
#     # Preprocess the frame for your CNN model
#     # This depends on your model's requirements
#     processed_frame = preprocess_frame_for_your_model(frame)

#     # Perform classification
#     prediction = model.predict(processed_frame)

#     # Overlay the prediction on the frame
#     # Customize this based on how you want to display the result
#     overlay_text = f"Prediction: {prediction}"
#     cv2.putText(frame, overlay_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     return frame

# def generate_frames():
#     cap = cv2.VideoCapture("http://192.168.0.25:8500/?action=stream")

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             # Process each frame through the CNN model
#             frame = process_frame(frame)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)


## possible pygame integration
# import pygame
# import cv2
# import numpy as np
# from flask import Flask, Response, render_template

# # Initialize Pygame
# pygame.init()
# window_size = (640, 480)
# screen = pygame.display.set_mode(window_size)

# app = Flask(__name__)

# def generate_frames():
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         # Your Pygame logic here
#         # For example, fill screen with a color
#         screen.fill((255, 0, 0))
#         pygame.display.flip()

#         # Capture Pygame frame
#         view = pygame.surfarray.array3d(screen)
#         view = view.transpose([1, 0, 2])  # Transpose it to the correct format
#         frame = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0")
