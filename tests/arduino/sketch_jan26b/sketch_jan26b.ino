#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

MPU6050 mpu;
// Global variables to keep track of the angle and the initial angle
float angleZ = 0;
float initialAngleZ = 0.0;
bool initialAngleSet = false;
unsigned long lastTime = 0;

bool isTurning = false;


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


// Encoder pins
const byte encoder0pinA = 33;
const byte encoder0pinB = 32;
const byte encoder1pinA = 25;
const byte encoder1pinB = 26;

int turnDuration = 350;


// Encoder step counters
volatile int stepsWheel1 = 0;
volatile int stepsWheel2 = 0;

// Last state of encoder signals
int lastStateA1 = 0;
int lastStateB1 = 0;
int lastStateA2 = 0;
int lastStateB2 = 0;


void setup() {
    Serial.begin(9600);
    Wire.begin();
    Serial.println("Initialize MPU6050");
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed");
        while (1);
    }

    pinMode(M1, OUTPUT);
    pinMode(M2, OUTPUT);
    analogWrite(E2, 0); // Set speed
    digitalWrite(M2, HIGH); // Move forward
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

        // Set up the encoder pins
    pinMode(encoder0pinA, INPUT);
    pinMode(encoder0pinB, INPUT);
    pinMode(encoder1pinA, INPUT);
    pinMode(encoder1pinB, INPUT);
  
    // Initialize encoder states
    lastStateA1 = digitalRead(encoder0pinA);
    lastStateB1 = digitalRead(encoder0pinB);
    lastStateA2 = digitalRead(encoder1pinA);
    lastStateB2 = digitalRead(encoder1pinB);
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
          isTurning= true;
          move_left();
        } else if (currentLine.endsWith("GET /right")) {
          Serial.println("moving right");
          isTurning= true;
          move_right();
        }
        else if (currentLine.endsWith("GET /stop") ){
          stop_moving(); 
          isTurning = false;
          initialAngleSet = false;
        }
      }
    }
    client.stop();
  }
   
}

float cumulativeOffset = 0.0;

float getYawFromMPU6050()
{
    
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


    cumulativeOffset += gyroZ * deltaTime;

    Serial.print("inital value: ");
    Serial.println(angleZ);
    return angleZ;
}

void get_steps() {
  int a1 = digitalRead(encoder0pinA);
  int b1 = digitalRead(encoder0pinB);
  int a2 = digitalRead(encoder1pinA);
  int b2 = digitalRead(encoder1pinB);

  // if a1 or b1 changes from 0 to 1, that means the encoder is count it as a step
  // if it changes states 
  if (a1 != lastStateA1 || b1 != lastStateB1) {
    if (a1 == 1 && b1 == 0) {
      stepsWheel1--;
    }
    if (a1 == 0 && b1 == 1) {
      stepsWheel1--;
    }
  }
  if (a2 != lastStateA2 || b2 != lastStateB2) {
    if (a2 == 1 && b2 == 0) {
      stepsWheel2--;
    }
    if (a2 == 0 && b2 == 1) {
      stepsWheel2--;
    }
  }
  lastStateA1 = a1;
  lastStateB1 = b1;
  lastStateA2 = a2;
  lastStateB2 = b2;
  Serial.print(stepsWheel1);
  Serial.print(" - ");
  Serial.print(stepsWheel2);
  Serial.println();

  delay(10);
}


void startTurning() {
  isTurning = true;
}


void move_forward() {
    int targetSteps = 8; // Define the target steps based on your requirements
    stepsWheel1 = 0;
    stepsWheel2 = 0;

    cumulativeOffset = 0.0;
    angleZ = 0.0;

    int rampUpTime = 333; // milliseconds for ramping up speed
    int steps = 255; // Number of steps for ramp-up
    int rampUpDelay = rampUpTime / steps; // Delay between each speed increment
    float initialYaw = getYawFromMPU6050();


    float kp = 1.0; // Proportional constant
    float ki = 0.1; // Integral constant
    float kd = 0.05; // Derivative constant

    float integral = 0.0;
    float lastError = 0.0;
    float derivative = 0.0;


    while (abs(stepsWheel1) < targetSteps || abs(stepsWheel2) < targetSteps ) {
        get_steps(); // Update steps from encoders      
        if(abs(stepsWheel1) >= targetSteps ){
          break;
        }

        float currentYaw = getYawFromMPU6050();
        float yawDifference = currentYaw - initialYaw;
        Serial.print("yawdifference:");
        Serial.println(yawDifference);
//
//        float currentYaw = getYawFromMPU6050();
        float yawError = currentYaw - initialYaw;

        integral += yawError;
        derivative = yawError - lastError;
        lastError = yawError;

        int adjustment = kp * yawError + ki * integral + kd * derivative; // PID adjustment

        int leftWheelSpeed = constrain(100 - adjustment, 0, 255);
        int rightWheelSpeed = constrain(100 + adjustment, 0, 255);

        analogWrite(E1, 200);
        analogWrite(E2, 200);
        digitalWrite(M1, HIGH);
        digitalWrite(M2, LOW);


      }
        
    stop_moving();
    stepsWheel1=0;
    stepsWheel2=0;

    // Final yaw check and correction if needed
    float finalYaw = getYawFromMPU6050(); 
    float yawDifferencefinal = initialYaw - finalYaw;
    float yawThreshold = 1.0;
    Serial.print("final difference: ");
    Serial.println(yawDifferencefinal);

    if (abs(yawDifferencefinal) > yawThreshold) {
        Serial.println("adjustment needed: ");
        if (yawDifferencefinal > 0) {
            isTurning = true;
            
            correct_left(yawDifferencefinal);
        } else {
            isTurning = true;
            correct_right(yawDifferencefinal);
        }
    }
}


void correct_left(float yawDifference) {

  yawDifference -= cumulativeOffset;

  int speed = 100;
  // Read and handle MPU6050 data
  while(isTurning) {
    // Left turn code for 90-degree stationary turn
    analogWrite(E1, speed); // Set speed for motor 1
    analogWrite(E2, speed); // Set speed for motor 2
    digitalWrite(M1, HIGH); // Run motor 1 forward
    digitalWrite(M2, HIGH); // Run motor 2 backward

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

    // Set the initial angle when a certain condition is met (e.g., a button press)
    if (!initialAngleSet) {
        initialAngleZ = angleZ;
        initialAngleSet = true;
        Serial.print("Initial angle set: ");
        Serial.println(initialAngleZ);
    }

    // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
    float angleDifference = angleZ - initialAngleZ;
    Serial.print("Current Angle Difference: ");
    Serial.println(angleDifference);
    angleDifference = abs(angleDifference);

    if (initialAngleSet && (angleDifference >= yawDifference)) {
        Serial.println("Rotated approximately 90 degrees from the initial position");
        // Reset the initial angle to start measuring again
        initialAngleSet = false;
        stop_moving();
        isTurning = false;
    }
    delay(100);
    speed++;
  }
  cumulativeOffset = 0.0;
}

void correct_right(float yawDifference ) {
  yawDifference -= cumulativeOffset;

  int speed = 100;
    // Read and handle MPU6050 data
  while(isTurning) {
  // Right turn code for 90-degree stationary turn
    analogWrite(E1, speed); // Set speed for motor 1
    analogWrite(E2, speed); // Set speed for motor 2
    digitalWrite(M1, LOW); // Run motor 1 backward
    digitalWrite(M2, LOW); // Run motor 2 forward

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

    // Set the initial angle when a certain condition is met (e.g., a button press)
    if (!initialAngleSet) {
        initialAngleZ = angleZ;
        initialAngleSet = true;
        Serial.print("Initial angle set: ");
        Serial.println(initialAngleZ);
    }

    // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
    float angleDifference = angleZ - initialAngleZ;
    Serial.print("Current Angle Difference: ");
    Serial.println(angleDifference);
    angleDifference = abs(angleDifference);

    if (initialAngleSet && (angleDifference >= yawDifference)) {
        Serial.println("Rotated approximately 90 degrees from the initial position");
        // Reset the initial angle to start measuring again
        initialAngleSet = false;
        stop_moving();
        isTurning = false;
    }
    delay(100);
    speed++;
  }
  cumulativeOffset = 0.0;
}



void move_left() {
  int speed = 150;
  // Read and handle MPU6050 data
  while(isTurning) {
    // Left turn code for 90-degree stationary turn
    analogWrite(E1, speed); // Set speed for motor 1
    analogWrite(E2, speed); // Set speed for motor 2
    digitalWrite(M1, HIGH); // Run motor 1 forward
    digitalWrite(M2, HIGH); // Run motor 2 backward

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

    // Set the initial angle when a certain condition is met (e.g., a button press)
    if (!initialAngleSet) {
        initialAngleZ = angleZ;
        initialAngleSet = true;
        Serial.print("Initial angle set: ");
        Serial.println(initialAngleZ);
    }

    // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
    float angleDifference = angleZ - initialAngleZ;
    Serial.print("Current Angle Difference: ");
    Serial.println(angleDifference);
    angleDifference = abs(angleDifference);

    if (initialAngleSet && (angleDifference > 85)) {
        Serial.println("Rotated approximately 90 degrees from the initial position");
        // Reset the initial angle to start measuring again
        initialAngleSet = false;
        stop_moving();
        isTurning = false;
    }
    delay(100);
    speed++;
  }
}

void move_right() {

  int speed = 150;
    // Read and handle MPU6050 data
  while(isTurning) {
  // Right turn code for 90-degree stationary turn
    analogWrite(E1, speed); // Set speed for motor 1
    analogWrite(E2, speed); // Set speed for motor 2
    digitalWrite(M1, LOW); // Run motor 1 backward
    digitalWrite(M2, LOW); // Run motor 2 forward

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

    // Set the initial angle when a certain condition is met (e.g., a button press)
    if (!initialAngleSet) {
        initialAngleZ = angleZ;
        initialAngleSet = true;
        Serial.print("Initial angle set: ");
        Serial.println(initialAngleZ);
    }

    // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
    float angleDifference = angleZ - initialAngleZ;
    Serial.print("Current Angle Difference: ");
    Serial.println(angleDifference);
    angleDifference = abs(angleDifference);

    if (initialAngleSet && (angleDifference > 85)) {
        Serial.println("Rotated approximately 90 degrees from the initial position");
        // Reset the initial angle to start measuring again
        initialAngleSet = false;
        stop_moving();
        isTurning = false;
    }
    delay(100);
    speed++;
  }
}
void stop_moving() {
  isTurning = false;
  analogWrite(E1, 0); // Stop one motor for turning
  analogWrite(E2, 0); // Set speed
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}
