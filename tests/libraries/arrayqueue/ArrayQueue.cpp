#include "ArrayQueue.h"

// *******************************************************************
// Gestion d'un FIFO du type :
//  ---FxxxxR----
// *******************************************************************
ArrayQueue::ArrayQueue()
{
  Front = -1;
  Rear  = -1;
}      
 
// *******************************************************************
// On met un element dans la FIFO (par le Rear). Exemples:
// xR---------
// --FxxxxR---
// xR----Fxxxx  
// Quand la file est pleine, les nouveaux éléments sont ignorés! 
//  xxFwwwRxxxxx
// *******************************************************************
void ArrayQueue::enQueue(int element, char* texte)
{
  if (this->getSize() == MAX_QUEUE_SIZE-1) 
  {
    Serial.println(F("Warning: overwriting ArrayQueue"));
    Serial.println("Rear="+String(Rear));
    Serial.println("Front="+String(Front));
    return;
  }
  // On stocke les données founies
  Queue[Rear] = element;
  Data[element] = texte;
  
  // Modulo is used so that rear indicator can wrap around
  Rear = ++Rear % MAX_QUEUE_SIZE;
}      

// *******************************************************************
// On retire un élément de la liste. Renvoie -1 si la liste est vide.
// *******************************************************************
int ArrayQueue::deQueue()
{          
  if (this->isEmpty()) return -1;

  int Value = Queue[Front];

  // Modulo is used so that front indicator can wrap around
  Front = ++Front % MAX_QUEUE_SIZE;

  return Value;   
}      
 
// *******************************************************************
// Renvoie le dernier element de la FIFO, mais sans le dépiler.
// Renvoie -1 si la liste est vide.
// *******************************************************************
int ArrayQueue::getFirst()
{          
  if (this->isEmpty()) return -1;
  return Queue[Front];
}
 
// *******************************************************************
// Renvoie le nombre dans la liste. Exemples:
// xR---------  : Size = 0--1 = 1
// --FxxxxR---  : Size = 5 -1 = 4
// xR----Fxxxx  : Size = 0 -5 = 5
// *******************************************************************
int ArrayQueue::getSize()
{
  return abs(Rear - Front);
}
 
// *******************************************************************
// Renvoie TRUE si la FIFO est vide. (cad si R=F).
// *******************************************************************
bool ArrayQueue::isEmpty()
{
  return (Front == Rear) ? true : false;
}

// *******************************************************************
// *******************************************************************
String ArrayQueue::getData(int index)
{          
  return Data[index];
}
 

