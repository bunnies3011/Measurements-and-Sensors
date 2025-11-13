#include <Servo.h>
Servo myservo;        
const int servoPin = 9;  
const int in1 = 5; //bánh phải lùi
const int in2 = 6;  //bánh phải tiến
const int in3 = 7;  //bánh trái lùi
const int in4 = 8; //bánh trái tiến
const int trig = 2;
const int echo = 3;
unsigned long time;
float distance, leftdistance, rightdistance;
float khoangcachvatcan = 25;  // khoảng cách phát hiện vật cản

void setup() {
pinMode(in1, OUTPUT);
pinMode(in2, OUTPUT);
pinMode(in3, OUTPUT);
pinMode(in4, OUTPUT);
pinMode(trig, OUTPUT);
pinMode(echo, INPUT);
myservo.attach(servoPin);
myservo.write(90);
}

void loop() {
  dokhoangcach();
  if(distance > khoangcachvatcan) tien();
  else{
    dung();
    quaycbsangphai();
    rightdistance = distance;
    quaycbsangtrai();
    leftdistance = distance;
    while (rightdistance < 25 && leftdistance < 25) {
      lui();
      quaycbsangphai();
      rightdistance = distance;
      quaycbsangtrai();
      leftdistance = distance;
    }
    if(rightdistance > leftdistance) rephai();
    else retrai();
  }
}


void lui(){
  analogWrite(in1, 250);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(500);
}

void tien(){
  digitalWrite(in1, LOW);
  analogWrite(in2, 253);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

void dung(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}

void retrai(){
  digitalWrite(in1, LOW);
  analogWrite(in2, 250);
  digitalWrite(in3, HIGH); //bánh trái lùi để rẽ phải
  digitalWrite(in4, LOW);
  delay(200);
}

void rephai(){
  analogWrite(in1, 250); //bánh phải lùi để rẽ trái
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  delay(200);
}

void dokhoangcach() {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  unsigned long duration = pulseIn(echo, HIGH);
  if (duration == 0) 
    distance = khoangcachvatcan + 1; // ngoài vùng cảm biến = đi tiếp
  else
    distance = duration / 2.0 / 29.138;
}

void quaycbsangphai(){
    myservo.write(0);              // tell servo to go to position in variable 'pos'
    delay(500);
    dokhoangcach();
    myservo.write(90);  
}

void quaycbsangtrai(){
    myservo.write(180);              // tell servo to go to position in variable 'pos'
    delay(500);
    dokhoangcach();
    myservo.write(90);              // tell servo to go to position in variable 'pos'  
}