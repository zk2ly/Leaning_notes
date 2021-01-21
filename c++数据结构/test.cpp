#include <iostream>
#include <cassert>
#include <vector>

using namespace std;

class sparseGraph{

private:
    int v, e;  // 记录结点和边的数量
    bool directed;  // 是否是有向图
    vector<vector<bool>> g;  // 邻接矩阵

public:
    sparseGraph(int n, bool directed){
        assert(n>0);

        this->v=n;
        this->e=0;
        this->directed = directed;
        this->g = vector<vector<bool>>(n, vector<bool>());  // g初始化为n个空的vector, 表示每一个g[i]都为空, 即没有任和边
    }

    ~sparseGraph(){}

    // 图中的节点数
    int numV(){
        return v;
    }

    // 图中的边数
    int numE(){
        return e;
    }

    // 判断是否有p到q的连接
    bool hasE(int p, int q){
        assert(p>0 && p<=v && q>0 && q<=v);  // 确定没有越界

        for(int i=0; i<g[p].size(); i++)
            if(g[p][i] == q)
                return true;
        return false;
    }

    // 添加p到q的边
    void addE(int p, int q){
        assert(p>0 && p<=v && q>0 && q<=v);

        if(hasE(p,q))
            return;

        g[p].push_back(q);  // 添加边

        if(!directed)
            g[q].push_back(p);

        e++;
    }
};



int main(){

    return 0;
}