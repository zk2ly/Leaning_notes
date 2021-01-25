#ifndef MY_UNIONFIND
#define MY_UNIONFIND

#include <iostream>
#include <cassert>

class unionFind{

private:
    int *parent;
    int capacity;
    int *rank;

public:
    unionFind(int capacity){
        parent = new int[capacity];
        rank = new int[capacity];
        this->capacity = capacity;
        for(int i=0; i<capacity; i++){
            parent[i] = i;
            rank[i] = 1;
        }    
    }

    ~unionFind(){
        delete[] parent;
    }

    int find(int p){
        assert(p>=0 && p<capacity);
        while(p != parent[p]){
            parent[p] = parent[parent[p]];
            p = parent[p];
        }

        return p;
    }

    bool isConnected(int p, int q){
        return find(p)==find(q);
    }

    void unionE(int p, int q){
        int rootp = find(p);
        int rootq = find(q);
        if(rootp == rootq) return;

        if(rank[rootp] > rank[rootq]){
            parent[rootq] = rootp;
        }
        else if(rank[rootq] > rank[rootp]){
            parent[rootp] = rootp; 
        }
        else{
            parent[rootp] = rootp; 
            rank[rootp]++;
        }
    }   

};

#endif