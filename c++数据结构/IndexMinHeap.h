#ifndef MY_INDEXMINHEAP
#define MY_INDEXMINHEAP

#include <iostream>
#include <cassert>

using std::swap;

template<typename T>
class IndexMinHeap{
private:
    T *data;
    int *index;  // index[x]=i表示 堆中x位置是索引i的元素
    int *reverse;  // reverse[i]=x表示 索引i的元素在堆中的x位置

    int count;  // 当前堆中元素个数
    int capacity;  // 堆的容量

    // 最小堆 如果存在父节点且比父节点小 节点上移
    void shiftUp(int k){
        while(k>1 && data[index[k]] < data[index[k/2]]){
            swap(index[k], index[k/2]);  // 堆中位置对应的元素索引互换一下
            reverse[index[k/2]] = k/2;  // 元素索引在堆中的位置更新一下 也可以互换
            reverse[index[k]] = k;
            k /= 2;
        }
    }

    // 如果存在子节点 选择子节点中最小的互换
    void shiftDown(int k){
        while(k*2 <= count){
            int j = 2*k;
            if(j+1 <= count && data[index[j+1]] < data[index[j]]) j=j+1;
            if(data[index[k]] < data[index[j]]) break;
            swap(index[j], index[k]);
            reverse[index[j]] = j;
            reverse[index[k]] = k;
            k=j;
        }
    }

public:
    IndexMinHeap(int capacity){
        data = new T[capacity+1];
        index = new int[capacity+1];
        reverse = new int[capacity+1];

        count = 0;
        this->capacity = capacity;

        // 初始堆中没有元素 也没有元素在堆中
        for(int i=0; i<=capacity;i++){
            reverse[i]=0;
            index[i]=0;
        }   
    }

    ~IndexMinHeap(){
        delete[] data;
        delete[] index;
        delete[] reverse;
    }

    int size(){return count;}  // 返回索引堆中的元素个数
    bool isEmpty(){return count == 0;}  // 返回一个布尔值, 表示索引堆中是否为空

    // 插入一个索引为i的元素t
    void insert(int i, T t){
        assert(count<capacity);  // 容量够
        assert(i>0 && i<capacity);  // 判断索引是否有效 用户从0开始索引
        
        i+=1;  // 堆从1开始索引 所以外部传进来的索引要加1
        data[i] = t;  // t元素插入索引i的位置
        index[count+1]=i;  // 堆中count+1的位置是索引为i的元素 即把t插入了堆尾
        reverse[i] = count+1;  
        count++;

        shiftUp(count);  // 新插入的堆尾元素做调整
    }

    // 取出堆顶元素  在这里虽然没有删除data中的元素 但是认为元素已经没有了
    T extractMin(){
        T e = data[1];
        swap(index[1], index[count]);  // 交换堆顶堆尾
        reverse[index[1]] =1;  // 元素index[1]在堆中1
        reverse[index[count]] =0;  // 元素index[count]不再在堆中了 
        count--;  // 删除堆尾元素
        shiftDown(1);  // 堆顶元素下移调整

        return e;
    }

    // 取出堆顶元素的索引 元素从堆的角度看不存在了 但是返回索引后可以根据索引来操作
    int etractMinIndex(){
        int idx = index[1]-1;
        swap(index[1], index[count]);
        reverse[index[1]] =1;
        reverse[index[count]] =0;
        count--;
        shiftDown(1);

        return idx;
    }

    T getMin(){return data[index[1]];}  // 获取序列中最小的元素
    T getMinIndex(){return index[1]-1;}  // 获取序列中最小元素的索引
    bool contain(int i){return reverse[i+1] != 0;}  // 看索引i所在的位置是否存在元素
    T getItem(int i){return data[i+1];}  //  获取序列中索引为i的元素

    // 序列的位置i处插入一个元素t
    void change(int i, T t){
        i+=1;
        data[i]=t;
        
        // 先尝试向上调整 或者调整成功 或者没有变化 相对应的两种结果再向下调整时 
        // 或者没有变化 或者调整成功
        shiftUp(reverse[i]); // reverse[i]表示索引为i的元素在堆中的位置 
        shiftDown(reverse[i]);  
    }
};

#endif