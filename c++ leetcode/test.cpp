#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
private:
    int need[128];
public:
    string minWindow(string s, string t) {
        // 此时窗口需要的字符和需要的个数
        for(string::size_type i =0; i<t.szie();i++)
            need[t[i]]++;
        int needCnt = t.size();

        // 窗口[i,j]  
        int i=0,j=-1;
        int len=s.size()-1, start = 0;
        while(j<s.size()){
            while(!needCnt && i+1<s.size()){  // 没有需要的元素了 即窗口内包含所有元素 尝试去掉第一个元素
                need[i]++;  // 去掉一个 还需要一个 如果本身是多余的 那么加进来的时侯会减成负数 此时加1 不会大于0
                if(need[i]) needCnt++;
                i++;
            }

            if(j+1 < t.size() && need[j+1]){
                j++;
                need[j]--;
                needCnt--;
            }

            if(!needCnt){
                if((j-i+1) < num){
                    num = j-i+1;
                    start = i;
                }
            }

        }

        
    }
};

int main(){
    // string s = "A man, a plan, a canal: Panama";
    // Solution().isPalindrome(s);

    return 0;
}