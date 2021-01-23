#ifndef MY_PRIM
#define MY_PRIM

#include <iostream>
#include <cassert>
#include <vector>
#include "Edge.h"
#include "IndexMinHeap.h"

using namespace std;

template<typename Graph, typename Weight>
class prim{

private:
    Graph &G;                     // 图的引用
    IndexMinHeap<Weight> ipq;     // 最小索引堆  保存生成树到各个节点的最短边的权值
    vector<Edge<Weight>*> edgeTo; // 保存生成树到这个节点的最短边
    bool *marked;                 // 表示是否已经访问过了 即是否已经加入生成树
    vector<Edge<Weight>> mst;     // 最小生成树所包含的所有边
    Weight mstWeight;             // 最小生成树的总权值

    void visit(int v){
        marked[v] = true;  // 加入生成树中
        vector<Edge<Weight>> adj = G.adjE();  // 所有邻接边
        for(int i=0; i<adj.size(); i++){
            int w = adj[i].other(v);  // w是边另一头的节点
            if(!marked[w]){  // 如果节点没有在生成树中
                if(!edgeTo[w]){  // 如果还没有保存到这个节点的最短边
                    edgeTo[w] = &adj[i];  // 当前边就保存为这个节点的最短边
                    ipq.insert(w, adj[i].wt());  // 保存当前边的权值
                }
                else if(adj[i].wt() < edgeTo[w].wt()){  // 如果保存过生成树到这个节点的最短边 但是当前的边更短 则替换
                    edgeTo[w]=&adj[i];
                    ipq.change(w, adj[i].wt());
                }   
            }
        }
    }

public:
    prim(Graph &g):G(g),ipq(IndexMinHeap<double>(g.V())){
        assert(g.E()>0);

        marked = new bool[G.V()];
        for(int i=0; i<G.V();i++){
            marked[i]=false;
            edgeTo.push_back(nullptr);
        }

        mst.clear();

        //prim
        visit(0);
        while(!ipq.isEmpty()){
            int v = ipq.etractMinIndex();  // 取出此时生成树可以连接到的边中 权值最小的那个节点序号
            mst.push_back(*edgeTo[v]);  // 把到这个节点的这条边加入到生成树中
            visit(v);  // 访问这个新加入的节点
        }

        // 统计最小生成树的总权值
        mstWeight = mst[0].wt();
        for( int i = 1 ; i < mst.size() ; i ++ )
            mstWeight += mst[i].wt();
    }

    ~prim(){
        delete[] marked;
    }

    // 返回最小生成树
    vector<Edge<Weight>> mstEdges(){
        return mst;
    };
    
    // 返回最小生成树的总权值
    Weight result(){
        return mstWeight;
    };
};

#endif
