#include <iostream>
#include <set>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int res=INT_MAX;
        int ans;
        int i,j;
        sort(nums.begin(),nums.end());
        for(int k=0;k<nums.size();k++){
            if(k>0 && nums[k]==nums[k-1]) continue;
            i=k+1;
            j=nums.size()-1;
            while(i<j){
                int sum = nums[k] + nums[i] + nums[j];
                if(sum - target == 0) return sum;
                if(abs(sum-target) < res){
                    res = abs(sum-target);
                    ans = sum;
                }
                if(sum > target)
                    j--;
                else
                    i++;
            }
        }
        return ans;
    }
};

