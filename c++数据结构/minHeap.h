#ifndef MY_MINHEAP
#define MY_MINHEAP

#include <iostream>
#include <cassert>

using std::swap;

template<typename T>
class minHeap{

private:
    T *data;
    int count;
    int capacity;

    void shiftDown(int k){
        while(2*k<=count){
            int j = 2*k;
            if(j+1<=count && data[j+1]<data[j]) j++;
            if(data[k] <= data[j]) break;
            swap(data[k], data[j]);
            k = j;
        }
    }

    void shiftUp(int k){
        while(k>1 && data[k] < data[k/2]){
            swap(data[k], data[k/2]);
            k = k/2;
        }
    }

public:
    minHeap(int capacity){
        data = new T[capacity+1];
        count = 0;
        this->capacity = capacity;
    }

    ~minHeap(){
        delete[] data;
    }

    int size(){return count;}  // 堆中元素数目
    bool isEmpty(){return count==0;}  // 堆是否为空
    T getMin(){return data[1];}   // 堆顶元素

    // 插入元素t
    void insert(T t){
        assert(count < capacity);
        data[count++] = t;
        shiftUp(count);
    }

    // 取出堆顶元素
    T extractMin(){
        assert(count>0);
        T res = data[1];
        swap(data[1], data[count]);
        count--;
        shiftDown(1);

        return res;
    }

};


#endif