#include <iostream>
#include <vector>
#include <string>

using namespace std;


class Solution {
public:
    // 滑动窗口[l,r]内没有重复字符 u保存窗口中有的字符 如果当前字符不包含其中 r++ 如果包含其中 l++直到不包含
    int lengthOfLongestSubstring(string s) {
        int l=0, r=-1;  // 初始没有字符[0,-1]
        int res = 0;
        int u[128]{0}; 

        // l==0 r==-1开始 
        // l==n  r==n-1结束
        while(l<s.size()){
            if(!u[s[r+1]] && r+1 < s.size()){
                r++;
                u[s[r]] = true;
                res = max(res, r-l+1); 
            }
            else{  // r+1这个元素包含在窗口中
                u[s[l]] = false;
                l++;
            }
        }
        return res;
    }
};

int main(){

    return 0;
}