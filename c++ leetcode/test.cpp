#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
// #include <unordered_map>
#include <algorithm>

using namespace std;

class Solution {
public:
    bool isHappy(int n) {
        int res=0;

        while(n<INT_MAX){
            while(n){
                res += (n%10)*(n%10);
                n /= 10;
            }
            if(res==1)
                return true;
            n = res;
            res = 0;
            cout<<n<<endl;
        }

        cout<<"false"<<endl;
        return false;
    }
};

int main(){
    Solution().isHappy(19);

    return 0;
}