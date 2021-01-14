#include <iostream>

using namespace std;

template<typename T>
void __Merge(T arr[], int l, int mid, int r){
    T aux[r-l+1];
    for(int i=l; i<=r;i++)
        aux[i-l] = arr[i];
    int i=l, j=mid+1;
    for(int k=l; k<=r; k++){
        if(i>mid){
            arr[k]=aux[j-l];
            j++;
        }
        else if(j>r){
            arr[k]=aux[i-l];
            i++;
        }
        else if(aux[i-l]>aux[j-l]){
            arr[k]=aux[j-l];
            j++;
        }
        else{
            arr[k]=aux[i-l];
            i++;
        }

    }
}


// 归并arr的[l,r]区域的数组
template<typename T>
void __MergeSort(T arr[], int l, int r){
    if(l >= r)
        return;
    
    int mid = (l+r)/2;
    __MergeSort(arr, l, mid);
    __MergeSort(arr, mid+1, r);
    __Merge(arr, l, mid, r);

}

// 封装成统一的参数形式
template<typename T>
void MergeSort(T arr[], int n){
    __MergeSort(arr, 0, n-1);
}

int main() {

    int a[10] = {10,9,8,7,6,5,4,3,2,1};
    MergeSort(a,10);
    for( int i = 0 ; i < 10 ; i ++ )
        cout<<a[i]<<" ";
    cout<<endl;

    return 0;
}