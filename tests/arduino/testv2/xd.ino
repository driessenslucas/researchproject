#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"
#include <ssd1306.h>

#include <WiFi.h>
#include <ArduinoOTA.h>

const char* ssid = "telenet-799DCED"; // Enter your WiFi SSID
const char* password = ""; // Enter your WiFi password

WiFiServer server(80);

// Variable to store the HTTP request
String header;

int sensor0Trig = 27;
int sensor0Echo = 26;

int sensor1Trig = 33;
int sensor1Echo = 32;

int sensor2Trig = 25;
int sensor2Echo = 35;

//define sound speed in cm/uS
#define SOUND_SPEED 0.034
#define CM_TO_INCH 0.393701

int E1 = 2;
int M1 = 17;
int E2 = 19;
int M2 = 4;

int turnDuration = 245;

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
  analogWrite(E2, 0);      // Set speed
  digitalWrite(M2, HIGH);  // Move forward

  Serial.begin(9600);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Wire.begin();
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed");
    while (1)
      ;
  }

   ssd1306_setFixedFont(ssd1306xled_font6x8);
   ssd1306_128x64_i2c_init();
   //  ssd1306_128x64_spi_init(22, 5, 21); // Use this line for ESP32 (VSPI)  (gpio22=RST, gpio5=CE for VSPI, gpio21=D/C)
   ssd1306_clearScreen();
   ssd1306_printFixed(0,  8, "ESP IP address:", STYLE_NORMAL);
   String ip = WiFi.localIP().toString();
   ssd1306_printFixed(0, 16, ip.c_str(), STYLE_BOLD);

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();

  ArduinoOTA.onStart([]() {
    String type;
    if (ArduinoOTA.getCommand() == U_FLASH)
      type = "sketch";
    else  // U_SPIFFS
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

  pinMode(sensor0Trig, OUTPUT);  // Sets the trigPin as an Output
  pinMode(sensor0Echo, INPUT);   // Sets the echoPin as an Input

  pinMode(sensor1Trig, OUTPUT);  // Sets the trigPin as an Output
  pinMode(sensor1Echo, INPUT);   // Sets the echoPin as an Input

  pinMode(sensor2Trig, OUTPUT);  // Sets the trigPin as an Output
  pinMode(sensor2Echo, INPUT);   // Sets the echoPin as an Input

  mpu.setZGyroOffset(-18);
}

void loop() {

  ArduinoOTA.handle();
  WiFiClient client = server.available();  // Listen for incoming clients

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
        } else if (currentLine.endsWith("GET /stop")) {
          stop_moving();
        } else if (currentLine.endsWith("GET /sensors")) {

          client.println("HTTP/1.1 200 OK");
          client.println("Content-type:text/plain");
          client.println("Connection: close");
          client.println();
          client.println(createJsonResponse());
          client.println("got sensors");
        }
      }
    }
    client.stop();
  }
}


String createJsonResponse() {
  String jsonResponse = "{";
  jsonResponse += "\"right\": " + get_sensor1() + ",";
  jsonResponse += "\"left\": " + get_sensor2() + ",";
  jsonResponse += "\"front\": " + get_sensor3();
  jsonResponse += "}";
  return jsonResponse;
}

String get_sensor1() {

  // Clears the trigPin
  digitalWrite(sensor0Trig, LOW);
  delayMicroseconds(2);
  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(sensor0Trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(sensor0Trig, LOW);

  // Reads the echoPin, returns the sound wave travel time in microseconds
  int duration = pulseIn(sensor0Echo, HIGH);

  // Calculate the distance
  float distanceCm = duration * SOUND_SPEED / 2;


  // Prints the distance in the Serial Monitor
  Serial.print("Distance (cm): ");
  Serial.println(distanceCm);

  if(distanceCm > 100.0){
    distanceCm = 100;
  }

  return String(distanceCm);
}
String get_sensor2() {

  // Clears the trigPin
  digitalWrite(sensor1Trig, LOW);
  delayMicroseconds(2);
  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(sensor1Trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(sensor1Trig, LOW);

  // Reads the echoPin, returns the sound wave travel time in microseconds
  int duration = pulseIn(sensor1Echo, HIGH);

  // Calculate the distance
  float distanceCm = duration * SOUND_SPEED / 2;


  // Prints the distance in the Serial Monitor
  Serial.print("Distance (cm): ");
  Serial.println(distanceCm);

  if(distanceCm > 100.0){
    distanceCm = 100;
  }

  return String(distanceCm);
}
String get_sensor3() {

  // Clears the trigPin
  digitalWrite(sensor2Trig, LOW);
  delayMicroseconds(2);
  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(sensor2Trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(sensor2Trig, LOW);

  // Reads the echoPin, returns the sound wave travel time in microseconds
  int duration = pulseIn(sensor2Echo, HIGH);

  // Calculate the distance
  float distanceCm = duration * SOUND_SPEED / 2;


  // Prints the distance in the Serial Monitor
  Serial.print("Distance (cm): ");
  Serial.println(distanceCm);

  if(distanceCm > 100.0){
    distanceCm = 100;
  }

  return String(distanceCm);
}

void move_forward() {
  // Forward movement code
  analogWrite(E1, 255);    // Set speed
  analogWrite(E2, 255);    // Set speed
  digitalWrite(M1, LOW);   // Move forward
  digitalWrite(M2, HIGH);  // Move forward

  delay(700);

  // Stop motors after turning

  analogWrite(E2, 0);

  delay(20);
  analogWrite(E1, 0);

  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}


void move_left() {
  isTurning = true;
  int initialSpeed = 100; // Set a higher initial speed
  int minSpeed = 50;      // Set a minimum speed
  int speed = initialSpeed;

  // Right turn code for 90-degree stationary turn
  while(isTurning) {
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

    // Calculate the angle difference
    float angleDifference = abs(angleZ - initialAngleZ);

    // Dynamically adjust speed based on angle difference
    speed = initialSpeed - (int)((angleDifference / 90) * (initialSpeed - minSpeed));
    speed = max(speed, minSpeed); // Ensure speed doesn't fall below minimum

        // Increase speed slightly if within 4 degrees of target angle
    if (angleDifference >= 84 && angleDifference < 90) {
        speed += 5; // Increase the speed by a small amount, e.g., 20
        speed = min(speed, initialSpeed); // Ensure speed doesn't exceed initial speed
    }

    // Set the motor speeds
    analogWrite(E1, speed); // Set speed for motor 1
    analogWrite(E2, speed); // Set speed for motor 2
    digitalWrite(M1, HIGH); // Run motor 1 backward
    digitalWrite(M2, HIGH); // Run motor 2 forward

    Serial.print("Current Angle Difference: ");
    Serial.println(angleDifference);

    // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
    if (initialAngleSet && angleDifference >= 87) {
        Serial.println("Rotated approximately 90 degrees from the initial position");
        // Reset the initial angle to start measuring again
        initialAngleSet = false;
        stop_moving();
        isTurning = false;
    }
    delay(100);
  }
}

void move_right() {
  isTurning = true;
  int initialSpeed = 100; // Set a higher initial speed
  int minSpeed = 50;      // Set a minimum speed
  int speed = initialSpeed;

  // Right turn code for 90-degree stationary turn
  while(isTurning) {
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

    // Calculate the angle difference
    float angleDifference = abs(angleZ - initialAngleZ);

    // Dynamically adjust speed based on angle difference
    speed = initialSpeed - (int)((angleDifference / 90) * (initialSpeed - minSpeed));
    speed = max(speed, minSpeed); // Ensure speed doesn't fall below minimum

    // Increase speed slightly if within 4 degrees of target angle
    if (angleDifference >= 84 && angleDifference < 90) {
        speed += 10; // Increase the speed by a small amount, e.g., 20
        speed = min(speed, initialSpeed); // Ensure speed doesn't exceed initial speed
    }

    // Set the motor speeds
    analogWrite(E1, speed); // Set speed for motor 1
    analogWrite(E2, speed); // Set speed for motor 2
    digitalWrite(M1, LOW); // Run motor 1 backward
    digitalWrite(M2, LOW); // Run motor 2 forward

    Serial.print("Current Angle Difference: ");
    Serial.println(angleDifference);

    // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
    if (initialAngleSet && angleDifference >= 88) {
        Serial.println("Rotated approximately 90 degrees from the initial position");
        // Reset the initial angle to start measuring again
        initialAngleSet = false;
        stop_moving();
        isTurning = false;
    }
    delay(100);
  }
}

void stop_moving() {
  isTurning = false;
  analogWrite(E1, 0);  // Stop one motor for turning
  analogWrite(E2, 0);  // Set speed
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}