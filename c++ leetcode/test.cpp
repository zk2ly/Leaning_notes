#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    // 二分搜索  没查找到返回插入位置
    int searchInsert(vector<int>& nums, int target) {
        int l = 0, r = nums.size()-1;
        bool flag=false;

        // [l,r]中查找
        while(l<=r){
            int mid = l + (r-l)/2;
            if(nums[mid] == target)
                return mid;
            else if(nums[mid] < target)
                l = mid + 1;
            else{
                r = mid - 1;
                flag = true;
            }
                
        }
        
        // 如果没找到 r左移 说明这个数小于原来的nums[mid] 即应该插入在mid的位置 即r当前的位置+1
        // l右移 说明这个数大于原来的nums[mid] 即应该插在mid+1的位置 即当前l的位置
        return flag ? r+1:l ;
    }
};

int main(){
    // vector<int> nums = {3,2,4};
    // vector<int> res = Solution().twoSum(nums, 6);
    // for(int i=0; i<res.size();i++)
    //     cout<<res[i]<<' ';
    // cout<<endl;

    return 0;
}