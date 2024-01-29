#include <WiFi.h>
#include <ArduinoOTA.h>
// or for ESP32: #include <WiFi.h>

const char* ssid = "telenet-799DCED";
const char* password = "";

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

int turnDuration = 265;


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

  pinMode(sensor0Trig, OUTPUT); // Sets the trigPin as an Output
  pinMode(sensor0Echo, INPUT); // Sets the echoPin as an Input

  pinMode(sensor1Trig, OUTPUT); // Sets the trigPin as an Output
  pinMode(sensor1Echo, INPUT); // Sets the echoPin as an Input

  pinMode(sensor2Trig, OUTPUT); // Sets the trigPin as an Output
  pinMode(sensor2Echo, INPUT); // Sets the echoPin as an Input
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

          client.println("HTTP/1.1 200 OK");
          client.println("Content-type:text/plain");
          client.println("Connection: close");

          client.println(createJsonResponse());
          client.println("moving forward");
        } else if (currentLine.endsWith("GET /left")) {
          Serial.println("moving left");
          move_left();

          client.println("HTTP/1.1 200 OK");
          client.println("Content-type:text/plain");
          client.println("Connection: close");

          client.println(createJsonResponse());
          client.println("left moving");
        } else if (currentLine.endsWith("GET /right")) {
          Serial.println("moving right");
          move_right();

          client.println("HTTP/1.1 200 OK");
          client.println("Content-type:text/plain");
          client.println("Connection: close");

          client.println(createJsonResponse());
          client.println("right moving");
        } else if (currentLine.endsWith("GET /stop")) {
          stop_moving();
          
          client.println("HTTP/1.1 200 OK");
          client.println("Content-type:text/plain");
          client.println("Connection: close");

          client.println(createJsonResponse());
          client.println("stopped moving");
        }
        else if (currentLine.endsWith("GET /sensors")) {
          
          client.println("HTTP/1.1 200 OK");
          client.println("Content-type:text/plain");
          client.println("Connection: close");
          client.println(); // Empty line to separate headers and body
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
  int duration1 = pulseIn(sensor0Echo, HIGH);
  
  // Calculate the distance
  float distanceCm1 = duration1 * SOUND_SPEED/2;
  

  // Prints the distance in the Serial Monitor
  Serial.print("Distance (cm): ");
  Serial.println(distanceCm1);

  
  return String(distanceCm1);
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
  int duration2 = pulseIn(sensor1Echo, HIGH);
  
  // Calculate the distance
  float distanceCm2 = duration2 * SOUND_SPEED/2;
  

  // Prints the distance in the Serial Monitor
  Serial.print("Distance (cm): ");
  Serial.println(distanceCm2);

  return String(distanceCm2);
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
  int duration3 = pulseIn(sensor2Echo, HIGH);
  
  // Calculate the distance
  float distanceCm3 = duration3 * SOUND_SPEED/2;
  

  // Prints the distance in the Serial Monitor
  Serial.print("Distance (cm): ");
  Serial.println(distanceCm3);

  return String(distanceCm3);
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
  digitalWrite(M2, HIGH);
  digitalWrite(M1, LOW);
}

void move_left() {
  // Left turn code for 90-degree stationary turn
  analogWrite(E1, 200);    // Set speed for motor 1
  analogWrite(E2, 200);    // Set speed for motor 2
  digitalWrite(M1, HIGH);  // Run motor 1 forward
  digitalWrite(M2, HIGH);  // Run motor 2 backward

  delay(turnDuration);  // Wait for the duration of the turn

  // Stop motors after turning
  analogWrite(E1, 0);
  analogWrite(E2, 0);
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}

void move_right() {
  // Right turn code for 90-degree stationary turn
  analogWrite(E1, 200);   // Set speed for motor 1
  analogWrite(E2, 200);   // Set speed for motor 2
  digitalWrite(M1, LOW);  // Run motor 1 backward
  digitalWrite(M2, LOW);  // Run motor 2 forward

  delay(turnDuration);  // Wait for the duration of the turn

  // Stop motors after turning
  analogWrite(E1, 0);
  analogWrite(E2, 0);
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}
void stop_moving() {
  analogWrite(E1, 0);  // Stop one motor for turning
  analogWrite(E2, 0);  // Set speed
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}