int encoder0pinA = 25;
int encoder0pinB = 26;

int E1 = 2;
int M1 = 17;
int E2 = 19;
int M2 = 4;

int encoderPosCount = 0;
int rotationCount = 0; // Counts the full rotations
int targetRotations = 0; // Target number of rotations
int pinALast;
int aVal;
bool targetReached = false;
unsigned long lastEncoderMoveTime = 0;
const unsigned long timeout = 3000; // 3 seconds timeout
const int positionsPerRotation = 3; // Change this based on your encoder's specification

void setup() {

  
  pinMode(M1, OUTPUT);
  pinMode(M2, OUTPUT);
  analogWrite(E2, 0); // Set speed
  digitalWrite(M2, HIGH); // Move forward
  Serial.begin(9600);
  pinMode(encoder0pinA, INPUT);
  pinMode(encoder0pinB, INPUT);
  pinALast = digitalRead(encoder0pinA);
  
  Serial.println("Enter the target number of rotations:");
}

void loop() {
  if (Serial.available() > 0) {
    targetRotations = Serial.parseInt(); // Read the target number of rotations
    Serial.print("Target rotations set to: ");
    Serial.println(targetRotations);
    targetReached = false;
  }

  aVal = digitalRead(encoder0pinA);

  if (!targetReached && aVal != pinALast) {
   
    if (digitalRead(encoder0pinB) != aVal) {
      encoderPosCount++;
    } else {
      encoderPosCount--;
    }

  }

    // Check for a full rotation
    if (abs(encoderPosCount) >= positionsPerRotation) {
      rotationCount++;
      encoderPosCount = 0; // Reset position count after a full rotation
      Serial.print("Full Rotations: ");
      Serial.println(rotationCount);

      if (rotationCount >= targetRotations) {
        Serial.println("Target number of rotations reached!");
        targetReached = true;
        // Optional: reset the rotation count
        stop_moving();
        rotationCount = 0;
      }
    }

    lastEncoderMoveTime = millis();
  

  if (millis() - lastEncoderMoveTime > timeout) {
    if (encoderPosCount != 0 || rotationCount != 0) {
      encoderPosCount = 0;
      rotationCount = 0;
      Serial.println("Position and Rotation reset to 0 due to inactivity.");
    }
    lastEncoderMoveTime = millis();
  }

  pinALast = aVal;
  delay(100);
}

void move_forward() {

  analogWrite(E1, 255); // Gradually increase speed
  analogWrite(E2, 255); 
  digitalWrite(M1, HIGH); // Move forward
  digitalWrite(M2, LOW);  


}
void stop_moving() {
  analogWrite(E1, 0); // Stop one motor for turning
  analogWrite(E2, 0); // Set speed
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}
