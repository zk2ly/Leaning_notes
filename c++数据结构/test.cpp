#include <iostream>
#include <vector>
#include <cassert>
#include <stack>
#include <queue>

using namespace std;

// class sparseGraph{

// private:
//     int v, e;  // 记录结点和边的数量
//     bool directed;  // 是否是有向图
//     vector<vector<int>> g;  // 邻接矩阵

// public:
//     sparseGraph(int n, bool directed){
//         assert(n>0);

//         this->v=n;
//         this->e=0;
//         this->directed = directed;
//         this->g = vector<vector<int>>(n, vector<int>());  // g初始化为n个空的vector, 表示每一个g[i]都为空, 即没有任和边
//     }

//     ~sparseGraph(){}

//     // 图中的节点数
//     int numV(){
//         return v;
//     }

//     // 图中的边数
//     int numE(){
//         return e;
//     }

//     // 判断是否有p到q的连接
//     bool hasE(int p, int q){
//         assert(p>0 && p<=v && q>0 && q<=v);  // 确定没有越界

//         for(int i=0; i<g[p].size(); i++)
//             if(g[p][i] == q)
//                 return true;
//         return false;
//     }

//     // 添加p到q的边
//     void addE(int p, int q){
//         assert(p>0 && p<=v && q>0 && q<=v);

//         if(hasE(p,q))
//             return;

//         g[p].push_back(q);  // 添加边

//         if(!directed)
//             g[q].push_back(p);

//         e++;
//     }

//     // 遍历p的邻边
//     vector<int> adjE(int p){
//         vector<int> adje = g[p];
//         return adje;
//     }
// };

template<typename Graph>
class shortestPath{

private:
    Graph &G;  
    bool *visited;  
    int *from; // 记录从哪个结点遍历过来的
    int *dis;  // 记录到原点的距离
    int s;  // 原点

    // 从点p开始广度优先遍历
    void bfs(int p){
        queue<int> q;

        q.push(p);  
        visited[p] = true;  // 入队即访问过了
        

        while(!q.empty()){
            int v = q.front();  // 取队首元素
            q.pop();

            vector<int> adj = G.adjE(v);  // 遍历队首元素的所有邻接结点 没访问过的入队
            for(int i=0; i<adj.size(); i++){
                if(!visited[i]){
                    q.push(i);  
                    visited[i] = true;  // 入队即访问过了
                    from[i] = v;
                    dis[i] = dis[v] + 1;  // 到原地的距离是此时队首元素加1
                }
            }
        }

    }

public:
    shortestPath(Graph &g, int s):G(g){  
        assert(s>=0 && s<G.numV());

        visited = new bool[G.numV()];  
        from = new int[G.numV()];  
        dis = new int[G.numV()]; 

        for(int i=0; i<G.numV(); i++){
            visited[i] = false;
            from[i] = -1;  
            dis[i] = -1;  // 初始到原点距离都无限大 设为-1
        }

        dis[s] = 0;  // 原点到自己的距离是0

        this->s = s;

        bfs(s);  
    }

    ~shortestPath(){
        delete[] visited;
        delete[] from;
        delete[] dis;
    }

    // 判断w是否与s相连
    bool hasPath(int w){
        assert(w>=0 && w<G.numV());

        return visited[w];
    }

    // 得到s到w的路径
    void getPath(int w, vector<int> &vec){
        
        stack<int> stk;

        while(w){
            stk.push(w);
            w = from[w];
        }

        vec.clear();

        while(!stk.empty()){
            vec.push_back(stk.top());
            stk.pop();
        }
    }

    // 打印路径
    void showPath(int w){
        vector<int> vec;
        getPath(w, vec);

        for(int i=0; i<vec.size(); i++)
            cout<<vec[i]<<' ';
    }
};


int main()
{   

    return 0;
}

