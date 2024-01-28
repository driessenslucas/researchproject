


#include "I2Cdev.h"
#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

#include <WiFi.h>
#include <ArduinoOTA.h>
// or for ESP32: #include <WiFi.h>

const char* ssid = "telenet-799DCED";
const char* password = "m7cnypsHjxhp";
WiFiServer server(80);

// Variable to store the HTTP request
String header;

int E1 = 2;
int M1 = 17;
int E2 = 19;
int M2 = 4;

int turnDuration = 350;

//mpu stuff

MPU6050 mpu;

// Global variables to keep track of the angle and the initial angle
float angleZ = 0;
float initialAngleZ = 0.0;
bool initialAngleSet = false;
unsigned long lastTime = 0;

bool isTurning = false;



void setup() {
  pinMode(M1, OUTPUT);
  pinMode(M2, OUTPUT);
  analogWrite(E2, 0); // Set speed
  digitalWrite(M2, HIGH); // Move forward
  Wire.begin(5,4);
  Serial.begin(9600);
  WiFi.begin(ssid, password);


  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  


  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();

    ArduinoOTA.onStart([]() {
    String type;
    if (ArduinoOTA.getCommand() == U_FLASH)
      type = "sketch";
    else // U_SPIFFS
      type = "filesystem";

    // NOTE: if updating SPIFFS this would be the place to unmount SPIFFS using SPIFFS.end()
    Serial.println("Start updating " + type);
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nEnd");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR) Serial.println("End Failed");
  });
  ArduinoOTA.begin();

  mpu.initialize();

  if (!mpu.testConnection()) {
      Serial.println("MPU6050 connection failed");
      while (1);
  }

  // Use this code to calibrate the MPU6050 and set offsets
  // mpu.setXGyroOffset();
//   mpu.setYGyroOffset();
//   mpu.setZGyroOffset();
  Serial.println("done init MPU6050");
}


void loop() {

  ArduinoOTA.handle();
  WiFiClient client = server.available(); // Listen for incoming clients

  if (client) {
    String currentLine = "";
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (currentLine.length() == 0) {
            // Send a standard HTTP response header
            client.println("HTTP/1.1 200 OK");
            client.println("Content-type:text/html");
            client.println("Connection: close");
            client.println();
            // Send the HTML content
            client.println("<!DOCTYPE html>");
            client.println("<html>");
            client.println("<body>");
            client.println("<h2>ESP32 Motor Control</h2>");
            client.println("<p><a href=\"/forward\"><button class=\"button\">Forward</button></a></p>");
            client.println("<p><a href=\"/left\"><button class=\"button\">Left</button></a></p>");
            client.println("<p><a href=\"/right\"><button class=\"button\">Right</button></a></p>");
            client.println("<p><a href=\"/stop\"><button class=\"button button2\">Stop</button></a></p>");
            client.println("</body>");
            client.println("</html>");

            break;
          } else {
            currentLine = "";
          }
        } else if (c != '\r') {
          currentLine += c;
        }


        // Check the request route
        if (currentLine.endsWith("GET /forward")) {
          Serial.println("moving forward");
          move_forward();
        } else if (currentLine.endsWith("GET /left")) {
          Serial.println("moving left");
          move_left();
        } else if (currentLine.endsWith("GET /right")) {
          Serial.println("moving right");
          move_right();
        }
        else{
          stop_moving(); 
        }

        // Read and handle MPU6050 data
        if (isTurning) {
         // Read raw gyro values
            int16_t ax, ay, az;
            int16_t gx, gy, gz;
            mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        
            // Convert gyroscope values to degrees/sec
            float gyroZ = gz / 131.0;
        
            // Current time in milliseconds
            unsigned long currentTime = millis();
            if (lastTime == 0) {
                lastTime = currentTime; // Initialize lastTime
            }
        
            // Time difference in seconds
            float deltaTime = (currentTime - lastTime) / 1000.0;
            lastTime = currentTime;
        
            // Integrate the gyroscope data
            angleZ += gyroZ * deltaTime;
      
          // Check if the turn is completed
          if (checkTurnCompleted()) {
              // Stop the motors
              stop_moving();
              isTurning = false;
              Serial.println("Turn completed");
          }
        }
      }
    }
    client.stop();
  }
}

void startTurning() {
  // Set the initial angle to the current angle
  initialAngleZ = angleZ;
  isTurning = true;
}

bool checkTurnCompleted() {
  // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
  float angleDifference = abs(angleZ - initialAngleZ);
  return (angleDifference > 88 && angleDifference < 93);
}

void move_forward() {
  // Forward movement code
  analogWrite(E1, 255); // Set speed
  analogWrite(E2, 255); // Set speed
  digitalWrite(M1, HIGH); // Move forward
  digitalWrite(M2, LOW); // Move forward

  delay(500);

  // Stop motors after turning
  analogWrite(E2, 0);
  analogWrite(E1, 0);
  digitalWrite(M2, HIGH);
  digitalWrite(M1, LOW);
}

void move_left() {
  // Left turn code for 90-degree stationary turn
  analogWrite(E1, 255); // Set speed for motor 1
  analogWrite(E2, 255); // Set speed for motor 2
  digitalWrite(M1, HIGH); // Run motor 1 forward
  digitalWrite(M2, HIGH); // Run motor 2 backward

  startTurning(); // Start turning and initialize the gyroscope measurement

//  delay(turnDuration); // Wait for the duration of the turn
//
//  // Stop motors after turning
//  analogWrite(E1, 0);
//  analogWrite(E2, 0);
//  digitalWrite(M1, LOW);
//  digitalWrite(M2, HIGH);
}

void move_right() {
  // Right turn code for 90-degree stationary turn
  analogWrite(E1, 255); // Set speed for motor 1
  analogWrite(E2, 255); // Set speed for motor 2
  digitalWrite(M1, LOW); // Run motor 1 backward
  digitalWrite(M2, LOW); // Run motor 2 forward

  startTurning();

//  delay(turnDuration); // Wait for the duration of the turn
//
//  // Stop motors after turning
//  analogWrite(E1, 0);
//  analogWrite(E2, 0);
//  digitalWrite(M1, LOW);
//  digitalWrite(M2, HIGH);
}
void stop_moving() {
  analogWrite(E1, 0); // Stop one motor for turning
  analogWrite(E2, 0); // Set speed
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}
