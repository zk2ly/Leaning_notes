#include <iostream>

// int main()
// {
//     int sum=0, val=50;
//     while(val<101){
//         sum += val;
//         ++val;
//     }
//     std::cout<<sum<<std::endl;

//     return 0;
// }

// int main()
// {
//     int val=10;
//     while(val>-1){
//         std::cout<< val << " ";
//         val--;
//     }
//     return 0;
// }

int main()
{
    int start = 0, end = 0;
    std::cin >> start >> end;
    while(start <= end){
        std::cout<<start<< " ";
        start++;
    }
}