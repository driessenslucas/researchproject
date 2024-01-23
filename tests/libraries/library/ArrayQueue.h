#ifndef ARRAYQUEUECLASS_H_INCLUDED
#define ARRAYQUEUECLASS_H_INCLUDED

#include <Arduino.h>

#define MAX_QUEUE_SIZE 15       
#define MAX_DATA_SIZE  50        

class ArrayQueue
{  
  public:
        ArrayQueue();
        void   enQueue(int element, char* texte);
        int    deQueue();
        int    getSize();
        int    getFirst();
        bool   isEmpty();
        String getData(int index);

  private:
        int     Queue[MAX_QUEUE_SIZE];
        int     Front;
        int     Rear;
        String  Data[MAX_DATA_SIZE];
};


#endif // ARRAYQUEUECLASS_H_INCLUDED
