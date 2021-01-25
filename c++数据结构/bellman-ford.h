#ifndef MY_BELLMANFORD
#define MY_BELLMANFORD

#include <stack>
#include <vector>
#include "Edge.h"

using namespace std;

template<typename Graph, typename Weight>
class bellmanFord{

private:
    Graph &G; 
    int s;
    Weight *distTo;
    vector<Edge<Weight>*> from;

    bool detectCycle(){
        // 对于所用顶点再做一个松弛操作 如果还能更新更短的距离 说明有负权边
        for(int i=0;i<G.V();i++){  
                vector<Edge<Weight>> adj = G.adjE(i);  
                for(int j=0; j<adj.size();j++){  
                    int w = adj[j].other(i); 
                    if(from[i] && (!from[w] || distTo[i] + adj[j].wt() < distTo[w]))
                        return true;      
                }
            }
            return false;
    }

public:
    bellmanFord(Graph &g, int s):G(g){
        this->s = s;
        distTo = new Weight[G.V()];
        for(int i=0;i<G.V();i++)
            from.push_back(nullptr);

        // bellman-ford
        from[s] = new Edge<Weight>(s, s, Weight());
        distTo[s] = Weight();

        // 对所有结点进行v-1次松弛操作 
        for(int pass=0;pass<G.V();pass++){
            for(int i=0;i<G.V();i++){  // 所有顶点
                vector<Edge<Weight>> adj = G.adjE(i);  
                for(int j=0; j<adj.size();j++){  // 的所有邻边
                    int w = adj[j].other(i);  // adj[j]这条边除了i结点以外的另一个结点w
                    if(from[i] && (!from[w] || distTo[i] + adj[j].wt() < distTo[w])){  // 在可以达到i的情况下 判断是否更新w
                        distTo[w] = distTo[i] + adj[j].wt();
                        from[w] = i;
                    }
                }
            }
        }
    }

    ~bellmanFord(){
        delete[] distTo;
        delete from[s];
    }

    Weight shortestWeightTo(int w){return distTo[w];}  // 返回从s到w最短路径的权值

    bool hasPathTo(int w){return from[w]!=nullptr;}  // 返回是否有s到w的路径

    bool hasCycle(){return detectCycle();}  // 返回是否有负权环

    // s到w最短路径经过的边 保存在vec中
    void shortestPath(int w, vector<Edge<Weight>> &vec){
        stack<Edge<Weight>*> s;
        Edge<Weight> *e = from[w];
        while(e->v() != this->s){  // 没返到原点前都继续循环
            s.push(e);
            e = from[e->v()];  // 变成连接边前面的那个结点
        } 
        s.push(e);  // 最后退出时，边的另一边是原点，即连接原点的边 也要入栈

        while(!s.empty()){
            e = s.top();
            vec.push_back(e);
            s.pop();
        }
    }

    // 打印出从s点到w点的路径
    void showPath(int w){
        vector<Edge<Weight>> vec;
        shortestPath(w, vec);
        for( int i = 0 ; i < vec.size() ; i ++ ){
            cout<<vec[i].v()<<" -> ";
            if( i == vec.size()-1 )
                cout<<vec[i].w()<<endl;
        }
    }
};

#endif