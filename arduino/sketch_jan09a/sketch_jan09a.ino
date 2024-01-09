// Editor     : Lauren from DFRobot
// Date       : 17.02.2012

// Product name: L298N motor driver module DF-MD v1.3
// Product SKU : DRI0002
// Version     : 1.0

// Description:
// The sketch for using the motor driver L298N
// Run with the PWM mode

// Connection:
// M1 pin -> Digital pin 4
// E1 pin -> Digital pin 5
// M2 pin -> Digital pin 7
// E2 pin -> Digital pin 6
// Motor Power Supply -> Center blue screw connector (5.08mm 3p connector)
// Motor A -> Screw terminal close to E1 driver pin
// Motor B -> Screw terminal close to E2 driver pin
//
// Note: You should connect the GND pin from the DF-MD v1.3 to your MCU controller. They should share the GND pins.

int E1 = 2;
int M1 = 17;
int E2 = 5;
int M2 = 4;

void setup() {
    pinMode(M1, OUTPUT);
    pinMode(M2, OUTPUT);
    Serial.begin(9600); // Initialize serial communication at 9600 bits per second
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    switch(command) {
      case 'F':
        move_forward();
        break;
      case 'L':
        move_left();
        break;
      case 'R':
        move_right();
        break;
    }
  }
}

void move_forward() {
  // Forward movement code
  analogWrite(E1, 255); // Set speed
  analogWrite(E2, 255); // Set speed
  digitalWrite(M1, HIGH); // Move forward
  digitalWrite(M2, LOW); // Move forward
}

void move_left() {
  // Left turn code
  analogWrite(E1, 255); // Set speed
  analogWrite(E2, 0); // Stop one motor for turning
  digitalWrite(M1, HIGH);
  digitalWrite(M2, HIGH);
}

void move_right() {
  // Right turn code
  analogWrite(E1, 0); // Stop one motor for turning
  analogWrite(E2, 255); // Set speed
  digitalWrite(M1, LOW);
  digitalWrite(M2, LOW);
}
