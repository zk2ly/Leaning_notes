#include <iostream>
#include <set>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        if(strs.size()==0) return {};

        unordered_map<string, vector<string>> recode;
        for(auto str:strs){
            string key = str;
            sort(key.begin(),key.end());
            recode[key].emplace_back(str);
        }

        vector<vector<string> > ans;
        for(auto it=recode.begin(); it!=recode.end();it++)
            ans.emplace_back(it->second);

        
        return ans;     
    }
};
