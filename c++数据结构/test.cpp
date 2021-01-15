#include <iostream>

using namespace std;

template<typename T>
void __QuickSort(T arr[], int l, int r){ 

    if(l>=r)
        return;

    swap(arr[l], arr[rand()%(r-l+1)+l]);  

    T v = arr[l];
    
    // arr[l+1,lt] < v  arr[lt+1, i)=v  arr[gt,r]>0  i是当前判断的位置
    int lt=l, i=l+1, gt=r+1;
    while(i<gt){
        if(arr[i]<v){
            swap(arr[i], arr[lt+1]);
            lt++;
            i++;
        }
        else if(arr[i]>v){
            swap(arr[i], arr[gt-1]);
            gt--;
        }
        else{
            i++;
        }
    }
    swap(arr[l], arr[lt]);
    __QuickSort(arr, l, lt);
    __QuickSort(arr, gt, r);
}

template<typename T>
void QuickSort(T arr[], int n){
    srand(time(NULL)); 
    __QuickSort(arr, 0, n-1);
}

int main() {

    int a[10] = {10,9,8,7,6,5,4,3,2,1};
    QuickSort(a,10);
    for( int i = 0 ; i < 10 ; i ++ )
        cout<<a[i]<<" ";
    cout<<endl;

    return 0;
}


