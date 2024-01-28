
//Wiring method:  
//black: motor power "-"
//Red: Motor Power "+"
//Brown: Encode Power "+"
//Green: Encode Power "-"
//Blue: Encode Output A phase
//White: encode phase B of output


const byte encoder0pinA = 33; // 25 or 33
const byte encoder0pinB = 32; // 26 or 32
const byte encoder1pinA = 25;
const byte encoder1pinB = 26; // 26 or 32



void setup() {
  Serial.begin(9600);
  pinMode(encoder0pinA,INPUT);
  pinMode(encoder0pinB,INPUT);
  pinMode(encoder1pinA,INPUT);
  pinMode(encoder1pinB,INPUT);
}

void loop() {
  int a1 = digitalRead(encoder0pinA);
  int b1 = digitalRead(encoder0pinB);
  int a2 = digitalRead(encoder1pinA);
  int b2 = digitalRead(encoder1pinB);
  Serial.print(a1); 
  Serial.print(" - ");
  Serial.print(b1);
  Serial.print(" //// ");
  Serial.print(a2); 
  Serial.print(" - ");
  Serial.print(b2);
  Serial.println();
  delay(100);
}


//
//const byte encoder0pinA = 33; // 25 or 33
//const byte encoder0pinB = 32; // 26 or 32
//const byte encoder1pinA = 25;
//const byte encoder1pinB = 26; // 26 or 32
//
//int stepsWheel1 =0;
//int stepsWheel2 =0;
//
//
//int lastStateA1 = 0;
//int lastStateB1 = 0;
//int lastStateA2 = 0;
//int lastStateB2 = 0;
//
//void setup() {
//  Serial.begin(9600);
//  pinMode(encoder0pinA,INPUT);
//  pinMode(encoder0pinB,INPUT);
//  pinMode(encoder1pinA,INPUT);
//  pinMode(encoder1pinB,INPUT);
//}
//
//void loop() {
//  int a1 = digitalRead(encoder0pinA);
//  int b1 = digitalRead(encoder0pinB);
//  int a2 = digitalRead(encoder1pinA);
//  int b2 = digitalRead(encoder1pinB);
////  Serial.print(a1); 
////  Serial.print(" - ");
////  Serial.print(b1);
////  Serial.print(" //// ");
////  Serial.print(a2); 
////  Serial.print(" - ");
////  Serial.print(b2);
////  Serial.println();
////  delay(100);
//
//
//
//  // if a1 or b1 changes from 0 to 1, that means the encoder is count it as a step
//  // if it changes states 
//  if (a1 != lastStateA1 || b1 != lastStateB1) {
//    if (a1 == 1 && b1 == 0) {
//      stepsWheel1++;
//    }
//    if (a1 == 0 && b1 == 1) {
//      stepsWheel1--;
//    }
//  }
//  if (a2 != lastStateA2 || b2 != lastStateB2) {
//    if (a2 == 1 && b2 == 0) {
//      stepsWheel2++;
//    }
//    if (a2 == 0 && b2 == 1) {
//      stepsWheel2--;
//    }
//  }
//  lastStateA1 = a1;
//  lastStateB1 = b1;
//  lastStateA2 = a2;
//  lastStateB2 = b2;
//  Serial.print(stepsWheel1);
//  Serial.print(" - ");
//  Serial.print(stepsWheel2);
//  Serial.println();
//  delay(10);
//
//}


//
//int E1 = 2;
//int M1 = 17;
//int E2 = 19;
//int M2 = 4;
//
//const byte encoder0pinA = 33; // 25 or 33
//const byte encoder0pinB = 32; // 26 or 32
//const byte encoder1pinA = 25;
//const byte encoder1pinB = 26; // 26 or 32
//
//int stepsWheel1 =0;
//int stepsWheel2 =0;
//
//int steps;
//
//bool stepsInit = false;
//
//int lastStateA1 = 0;
//int lastStateB1 = 0;
//int lastStateA2 = 0;
//int lastStateB2 = 0;
//
//
//// PID parameters
//float Kp = 1.0; // Proportional gain
//float Ki = 0.1; // Integral gain
//float Kd = 0.05; // Derivative gain
//
//// PID variables
//float integral = 0;
//float lastError = 0;
//unsigned long lastTime = millis();
//
//
//void setup() {
//  Serial.begin(9600);
//  pinMode(encoder0pinA,INPUT);
//  pinMode(encoder0pinB,INPUT);
//  pinMode(encoder1pinA,INPUT);
//  pinMode(encoder1pinB,INPUT);
//
//  pinMode(M1, OUTPUT);
//  pinMode(M2, OUTPUT);
//  analogWrite(E2, 0); // Set speed
//  digitalWrite(M2, HIGH); // Move forward
//
//  lastTime = millis();
//}
//
//void loop() {
//  if (Serial.available() > 0 && !stepsInit) {
//    steps = Serial.parseInt();
//    stepsInit = true;
//  }
//
//  if (stepsInit) {
//    Serial.print("steps wanted: ");
//    Serial.println(steps);
//
//    if (abs(stepsWheel1) >= steps && abs(stepsWheel2) >= steps) {
//      stopMotors();
//      stepsInit = false; // Reset the flag after stopping motors
//      stepsWheel1 = 0;
//      stepsWheel2 = 0;
//    } else {
//      moveMotors();
//    }
//    get_steps();
//  }
//
//}
//
//void get_steps() {
//  int a1 = digitalRead(encoder0pinA);
//  int b1 = digitalRead(encoder0pinB);
//  int a2 = digitalRead(encoder1pinA);
//  int b2 = digitalRead(encoder1pinB);
//
//  // if a1 or b1 changes from 0 to 1, that means the encoder is count it as a step
//  // if it changes states 
//  if (a1 != lastStateA1 || b1 != lastStateB1) {
//    if (a1 == 1 && b1 == 0) {
//      stepsWheel1++;
//    }
//    if (a1 == 0 && b1 == 1) {
//      stepsWheel1--;
//    }
//  }
//  if (a2 != lastStateA2 || b2 != lastStateB2) {
//    if (a2 == 1 && b2 == 0) {
//      stepsWheel2++;
//    }
//    if (a2 == 0 && b2 == 1) {
//      stepsWheel2--;
//    }
//  }
//  lastStateA1 = a1;
//  lastStateB1 = b1;
//  lastStateA2 = a2;
//  lastStateB2 = b2;
//  Serial.print(stepsWheel1);
//  Serial.print(" - ");
//  Serial.print(stepsWheel2);
//  Serial.println();
//}
//
//int calculatePID(int error) {
//  unsigned long now = millis();
//  float timeChange = (float)(now - lastTime);
//
//  // Proportional term
//  float proportional = Kp * error;
//
//  // Integral term
//  integral += Ki * error * timeChange;
//
//  // Derivative term
//  float derivative = Kd * (error - lastError) / timeChange;
//
//  // Calculate total output
//  int output = proportional + integral + derivative;
//
//  // Remember some variables for next time
//  lastError = error;
//  lastTime = now;
//
//  return output;
//}
//
//void stopMotors(){
//    analogWrite(E1, 0); // Gradually increase speed
//    analogWrite(E2, 0);
//    digitalWrite(M1, LOW); // Move forward
//    digitalWrite(M2, HIGH);
//}
//void moveMotors() {
//  // Calculate the error (difference between desired and actual steps)
//  int error = steps - abs(stepsWheel1); // Assuming stepsWheel1 is for motor E1
//
//  // Calculate PID output
//  int pidOutput = calculatePID(error);
//
//  // Ensure the output is within a valid range for analogWrite (e.g., 0 to 255)
//  int motorSpeed = constrain(pidOutput, 0, 255);
//
//  digitalWrite(M1, HIGH); // Move forward
//  digitalWrite(M2, LOW);    
//  // Control motor speed using PID output
//  analogWrite(E1, motorSpeed);
//  // Repeat for other motors if necessary
//  analogWrite(E2,motorSpeed);
//}



//
//int E1 = 2;
//int M1 = 17;
//int E2 = 19;
//int M2 = 4;
//
//const byte encoder0pinA = 33; // 25 or 33
//const byte encoder0pinB = 32; // 26 or 32
//const byte encoder1pinA = 25;
//const byte encoder1pinB = 26; // 26 or 32
//
//int stepsWheel1 =0;
//int stepsWheel2 =0;
//
//int steps;
//
//bool stepsInit = false;
//
//int lastStateA1 = 0;
//int lastStateB1 = 0;
//int lastStateA2 = 0;
//int lastStateB2 = 0;
//
//void setup() {
//  Serial.begin(9600);
//  pinMode(encoder0pinA,INPUT);
//  pinMode(encoder0pinB,INPUT);
//  pinMode(encoder1pinA,INPUT);
//  pinMode(encoder1pinB,INPUT);
//
//  pinMode(M1, OUTPUT);
//  pinMode(M2, OUTPUT);
//  analogWrite(E2, 0); // Set speed
//  digitalWrite(M2, HIGH); // Move forward
//}
//
//void loop() {
//  if (Serial.available() > 0 && !stepsInit) {
//    steps = Serial.parseInt();
//    stepsInit = true;
//  }
//
//  if (stepsInit) {
//    Serial.print("steps wanted: ");
//    Serial.println(steps);
//
//    if (abs(stepsWheel1) >= steps || abs(stepsWheel2) >= steps) {
//      stopMotors();
//      stepsInit = false; // Reset the flag after stopping motors
//      stepsWheel1 = 0;
//      stepsWheel2 = 0;
//    } else {
//      moveMotors();
//    }
//    get_steps();
//  }
//}
//
//void get_steps() {
//  int a1 = digitalRead(encoder0pinA);
//  int b1 = digitalRead(encoder0pinB);
//  int a2 = digitalRead(encoder1pinA);
//  int b2 = digitalRead(encoder1pinB);
//
//  // if a1 or b1 changes from 0 to 1, that means the encoder is count it as a step
//  // if it changes states 
//  if (a1 != lastStateA1 || b1 != lastStateB1) {
//    if (a1 == 1 && b1 == 0) {
//      stepsWheel1--;
//    }
//    if (a1 == 0 && b1 == 1) {
//      stepsWheel1--;
//    }
//  }
//  if (a2 != lastStateA2 || b2 != lastStateB2) {
//    if (a2 == 1 && b2 == 0) {
//      stepsWheel2--;
//    }
//    if (a2 == 0 && b2 == 1) {
//      stepsWheel2--;
//    }
//  }
//  lastStateA1 = a1;
//  lastStateB1 = b1;
//  lastStateA2 = a2;
//  lastStateB2 = b2;
//  Serial.print(stepsWheel1);
//  Serial.print(" - ");
//  Serial.print(stepsWheel2);
//  Serial.println();
//}
//
//
//void stopMotors(){
//    analogWrite(E1, 0); // Gradually increase speed
//    analogWrite(E2, 0);
//    digitalWrite(M1, LOW); // Move forward
//    digitalWrite(M2, HIGH);
//}
//
//void move_left()
//{
//  analogWrite(E1, 50); // Set speed for motor 1
//  analogWrite(E2, 50); // Set speed for motor 2
//  digitalWrite(M1, HIGH); // Run motor 1 forward
//  digitalWrite(M2, HIGH); // Run motor 2 backward  
//}
//
//void move_right()
//{
//  analogWrite(E1, 50); // Set speed for motor 1
//  analogWrite(E2, 50); // Set speed for motor 2
//  digitalWrite(M1, LOW); // Run motor 1 backward
//  digitalWrite(M2, LOW); // Run motor 2 forward 
//}
//void moveMotors(){
//    analogWrite(E1, 100); // Gradually increase speed
//    analogWrite(E2, 100); 
//    digitalWrite(M1, HIGH); // Move forward
//    digitalWrite(M2, LOW);    
//  
//}
