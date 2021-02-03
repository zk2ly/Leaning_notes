#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;


class Solution {
public:
    // 窗口[l,r)
    // 当前元素s[r] 加入窗口
    // 判断还缺不缺t中的元素 如果不缺  缩减窗口大小  要保证始终包含所有t的字符
    // 记录此时的开始长度和大小
    // 踢出左侧元素 继续循环
    string minWindow(string s, string t) {
        int l=0, r =0;
        int need[128]{0};
        int needCnt = t.size();
        int start=0, length=s.size()+1;

        for(int i =0;i<t.size();i++)
            need[t[i]]++;

        while(r<s.size()){
            if(need[s[r]]>0) needCnt--;
            need[s[r]]--;
            r++;
            if(needCnt==0){
                while(l<r && need[s[l]] < 0){
                    need[s[l]]++;
                    l++;
                }

                if(r-l < length){
                    length = r-l;
                    start = l;
                }
            
                need[s[l]]++;
                l++;
                needCnt++;
            }
        }
        if(length == s.size()+1)
            return "";
        else
            return s.substr(start,length);
    }
};

int main(){
    Solution().minWindow("ADOBECODEBANC", "ABC");

    return 0;
}