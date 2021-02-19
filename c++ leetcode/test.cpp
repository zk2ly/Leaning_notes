#include <iostream>
#include <set>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
private:
    int dis(int x1, int y1, int x2, int y2)
        return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
public:
    int numberOfBoomerangs(vector<vector<int>>& points) {
        if(points.size()<3) return 0;
        int d, ans=0;
        unordered_map<int,int> recode;
        for(int i=0;i<points.size();i++){
            for(int j=i+1;j<points.size();j++){
                d = dis(points[i][0],points[i][1], points[j][0],points[j][1]);
                if(recode.count(d)) 
                    ans += recode[d];
                ++recode[d];            
        }
        recode.clear();    
    }
};
