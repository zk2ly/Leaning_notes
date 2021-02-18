#include <iostream>
#include <set>

using namespace std;

class Solution {
public:
    bool wordPattern(string pattern, string s) {
        map<char, string> recode;
        set<string> contain;
        stringstream ss(s);
        string str;

        for(int i =0; i<pattern.size();i++){

        
            ss>>str;

            if(str.empty()) return false;

            char c = pattern[i];
            if(recode.find(c)==recode.end()){
                // if(contain.find(str) != contain.end()) return false;
                recode[c] = str;
                contain.insert(str);
            }else{
                if(recode[c] != str)
                    return false;
            }
        }
        return true;
         
    }
};

int main(){

    return 0;
}

