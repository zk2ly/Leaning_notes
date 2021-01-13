#include <iostream>

int main()
{
    int val, sum=0;
    while(std::cin>>val){
        sum += val;
    }
    std::cout<<sum<<std::endl;

    return 0;
}