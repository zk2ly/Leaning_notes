

# 1.线性结构
## 1.1 O(n^2)的排序算法
### 1.1.1 选择排序
>算法步骤：

遍历序列的所有位置，每次都从未排序的序列中选择出最小的值，放到当前位置。

>代码：

```c++
#include <iostream>

template<typename T>  // 模板函数，可以接受任意的数据类型
void SelectionSort(T arr[], int n){   
    for(int i=0; i<n; i++){
        int minidx = i;
        for(int j=i; j<n; j++)
            if(arr[j] < arr[minidx])
                minidx = j;
        swap(arr[i], arr[minidx]);
    }
}
```

### 1.1.2 插入排序
>算法步骤：

遍历序列的所有位置，每次都把当前位置的值与前面的值做比较，如果小于前面的值，则交换，然后再和更前面的一个比较，直至遇到比他小的则停止。

>代码：

```c++
#include <iostream>

template<typename T>  
void InsertionSort(T arr[], int n){   
    for(int i=1; i<n; i++)
        for(int j=i; j>0; j--)
            if(arr[j] < arr[j-1])
                swap(arr[j], arr[j-1]);
            else
                break;
}
```
>改进

选择排序和插入排序都是o(n^2)的算法，而且插入排序在比较时可能会中途终止循环，但此时的代码插入排序更耗时，因为每次比较成功后的swap交换耗费了时间。可以先保存此时要排序的值，并设置一个变量保存当前的位置，与前一个值比较，小于前一个值时，用前一个值直接覆盖当前位置，大于前一个值时退出，放到当前位置。
```c++
#include <iostream>

template<typename T>
void InsertionSort(T arr[], int n){   
    for(int i=1; i<n; i++){
        T val=arr[i];  // 当前要排序的数
        int j;   // 记录要插入的位置
        for(j=i; j>0; j--)
            if(val < arr[j-1])
                arr[j]=arr[j-1];
            else
                break;   
        arr[j]=val;        
    }
}
```
>分析

插入排序对于基本有序的数组，速度非常快，甚至快过许多O(nlogn)的算法，当数组完全有序时，插入排序的时间复杂度是O(n)，相当于只遍历一遍数组，每次都不交换。因此插入排序应用十分广泛。

### 1.1.3 冒泡排序
>算法步骤：

比较当前元素和后一个元素，大的往后放，遍历n-1次后变有序

>代码：
```c++
template<typename T>
void BubbleSort(T arr[], int n){   
    for(int i=0; i<n-1; i++)
        for(int j=0; j<n-1-i; j++)
            if(arr[j]>arr[j+1])
                swap(arr[j], arr[j+1]);
}
```

### 1.1.4 希尔排序
>算法步骤：

插入排序的改进版，每次不是把当前位置的值和前一个比较，而是和第前k个位置的比较，k的取值是一个递减序列，每排序一次，序列变得更有序一些，k值就下降一些，最后一次k=1，就是普通的插入排序，但是由于此时已经基本有序，所以非常快。

>代码：
```c++
template<typename T>
void ShellSort(T arr[], int n){
    int k=1;
    while(k < n/3)
        k = 3 * k + 1;

    while(k>=1){
        for(int i=0; i<n; i++){
            T val = arr[i];
            int j;
            for(j=i; j>=k; j=j-k)  // 要保证j-k要>=0
                if(arr[j-k] > val)
                    arr[j] = arr[j-k];
                else
                    break;
            arr[j] = val;
        }

        k = k / 3;
    }
}
```

>分析

希尔排序是这几个O(n^2)的算法中性能最优的

## 1.2 O(nlogn)的排序算法
### 1.2.1 归并排序
>算法步骤：

把数组平分，保证平分后的两个数组是有序的，然后把这两个数组归并起来。

保证平分后数组有序的方法是递归的调用排序算法。

归并两个数组的方法是额外开辟一个新数组，把这两个数组的值复制进去，然后用i,j分别在新数组上表示这两个数组的当前要比较元素的位置，小的值放到原数组的位置k上。

>代码：
```c++
// 归并arr[l,mid]和arr[mid+1,r]两个数组
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
```
>改进1

随机排序序列情况下，归并排序比插入排序快得多，但是对于基本有序序列，插入排序快得多，可以改进一下归并排序算法，使得在排序基本有序序列时，性能可以得到提升。
```c++

```
改进之后对于随机的序列，时间会有所增加，因为判断语句本身也要耗时，因此对于不会出现基本有序序列的情况下，可以不用判断语句，对于有可能会出现基本有序序列的情况下，可以加上判断语句。

>改进2

对于所有的递归的排序算法都有的一种改进方式，就是递归边界改进。这里的递归边界是`l>=r`,即只有一个元素时，不用排序直接返回。可以改成剩下n个元素时，用归并排序，因为此时数组较小，有序的可能性比较大，而且数组比较小时，即n比较小，此时O(nlogn)和O(n^2)的差距没有那么大。

### 1.2.2 快速排序