#include <iostream>
#include <cassert>

class unionFind{

private:
    int* parent;
    int* rank;  // 记录每一棵树的高度
    int count;

public:
    unionFind(int n){
        parent = new int[10];
        rank = new int[10];
        count = n;

        // 初始化时每个元素自己一个组 互不连通 树高都为1
        for(int i=0; i<n; i++)
            parent[i] = i;
            rank[i] = 1;
    }

    ~unionFind(){
        delete[] parent;
        delete[] rank;
    }

    // 不做修改
    int find(int p){
        while(parent[p] != p)
            p = parent[p];

        return p;
    }

    // 不做修改
    bool isConnected(int p, int q){
        return find(p) == find(q);
    }

    // 合并时矮树指向高树，等高时随意指向，但被指向的那棵树rank+1
    void unionE(int p, int q){
        int pid = find(p);
        int qid = find(q);

        if(pid == qid)
            return;

        if(rank[pid] < rank[qid])
            parent[pid] = qid;
        else if(rank[qid] < rank[pid])
            parent[qid] = pid;
        else{
            parent[qid] = pid;
            rank[pid] ++;
        }  
    }
};

int main(){

    return 0;
}