#ifndef MY_DIJKSTRA
#define MY_DIJKSTRA

#include <iostream>
#include <cassert>
#include "Edge.h"
#include "IndexMinHeap.h"
#include <vector>
#include <stack>

using namespace std;

template<typename Graph, typename Weight>
class dijkstra{
private:
    Graph &G;  // 图的引用
    int s;  // 原点
    Weight *disTo;  // 原点到各店的最短距离的数组
    bool *marked;  // 标记是否访问过的数组
    vector<Edge<Weight> *> from;  // 最短路径中 当前点的上一个点

public:
    dijkstra(Graph &g, int s):G(g){
        this->s = s;
        disTo = new Weight[G.V()];
        marked = new bool[G.V()];
        IndexMinHeap<Weight> ipq(G.V());  // 使用索引堆记录当前找到的到达每个顶点的最短距离
        
        for(int i=0;i<G.V();i++){
            disTo[i] = Weight();
            marked[i] = false;
            from.push_back(nullptr);
        }

        disTo[s] = Weight();
        from[s] = new Edge<Weight>(s,s,Weight());
        marked[s] = true;
        ipq.insert(s, disTo[s]);  // 到点s的最短距离入队

        while(!ipq.isEmpty()){
            int v = ipq.etractMinIndex();
            marked[v] = true;

            vector<Edge<Weight>> adj = G.adjE(v);
            for(int i=0; i<adj.size();i++){
                int w = adj[i].other(v);
                if(!marked[w]){
                    if(from[w]==nullptr || disTo[v] + adj[i].wt() < disTo[w]){
                        disTo[w] = disTo[v] + adj[i].wt();
                        from[w] = v;
                        if(ipq.contain(w))
                            ipq.change(w, disTo[w]);
                        else
                            ipq.insert(w, disTo[w]);
                    }
                }
            }

        }
    }

    ~dijkstra(){
        delete[] disTo;
        delete[] marked;
        delete from[s];
    }

    // 返回从s点到w点的最短路径长度
    Weight shortestPathTo( int w ){
        return disTo[w];
    }

    // 判断从s点到w点是否联通
    bool hasPathTo( int w ){
        return marked[w];
    }

     // 寻找从s到w的最短路径, 将整个路径经过的边存放在vec中
    void shortestPath( int w, vector<Edge<Weight>> &vec ){

        assert( w >= 0 && w < G.V() );
        assert( hasPathTo(w) );

        // 通过from数组逆向查找到从s到w的路径, 存放到栈中
        stack<Edge<Weight>*> s;
        Edge<Weight> *e = from[w];
        while( e->v() != this->s ){
            s.push(e);
            e = from[e->v()];
        }
        s.push(e);

        // 从栈中依次取出元素, 获得顺序的从s到w的路径
        while( !s.empty() ){
            e = s.top();
            vec.push_back( *e );
            s.pop();
        }
    }
};


#endif