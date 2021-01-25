#ifndef MY_KRUSKAL
#define MY_KRUSKAL

#include <iostream>
#include <cassert>
#include <vector>
#include "Edge.h"
#include "minHeap.h"
#include "unionFind.h"

using namespace std;

template <typename Graph, typename Weight>
class kruskal{
private:
    vector<Edge<Weight>> mst;  // 存所有边
    Weight mstWeight;  // 总权值

public:
    kruskal(Graph &g){
        minHeap<Edge<Weight>> pq(g.E());  // 最小堆存所有边

        for(int v=0; v<g.V(); v++){  // 遍历所有结点
            vector<Edge<Weight>> adj = g.adjE(v);  
            for(int i=0;i<adj.size();i++){  // 结点的所有邻边
                if(adj[i].v() < adj[i].w())  // 无权图中v->w w->v 是一条边 为了防止存两次 只在v<w时存一次
                    pq.insert(adj[i]);
            }
        }

        unionFind uf = unionFind(g.V());  // 并查集记录连接情况

        while(!pq.isEmpty() && mst.size() < g.V()-1){
            Edge<Weight> e = pq.extractMin();
            if(uf.isConnected(e.v(), e.w()))  // 这条边的两个顶点如果已经相连接了 说明已经在树中 
                continue;
            mst.push_back(e);
            uf.unionE(e.v(), e.w());
        }

    }

    ~kruskal(){}

    // 返回最小生成树的所有边
    vector<Edge<Weight>> mstEdges(){
        return mst;
    }

    // 返回最小生成树的权值
    Weight result(){
        return mstWeight;
    }
};

#endif