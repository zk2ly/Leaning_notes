#ifndef MY_EDGE
#define MY_EDGE

#include <iostream>
#include <cassert>

template<typename Weight>  // 权重可能是不同的数据类型
class Edge{
private:
    int a,b;
    Weight weight;

public:
    Edge(int a, int b, Weight weight){
        this->a = a;
        this->b = b;
        this-> weight = weight;
    }

    // 空的构造函数 所有成员变量都取默认值
    Edge(){}

    ~Edge(){}

    int v(){ return a;} // 返回第一个顶点
    int w(){ return b;} // 返回第二个顶点
    Weight wt(){ return weight;}    // 返回权值

    // 给定一个顶点返回另一个顶点
    int other(int x){
        assert(x==a || x==b);
        return x==a ? b : a;
    }

    // 边的大小比较, 是对边的权值的大小比较
    bool operator < (Edge<Weight> &e){return weight < e.wt();}
    bool operator > (Edge<Weight> &e){return weight > e.wt();}
    bool operator <= (Edge<Weight> &e){return weight <= e.wt();}
    bool operator >= (Edge<Weight> &e){return weight >= e.wt();}
    bool operator == (Edge<Weight> &e){return weight == e.wt();}


};

#endif
