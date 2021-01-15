

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
// 归并arr的[l,r]区域的数组
template<typename T>
void __MergeSort(T arr[], int l, int r){
    if(l >= r)
        return;
    
    int mid = (l+r)/2;
    __MergeSort(arr, l, mid);
    __MergeSort(arr, mid+1, r);
    if(arr[mid] > arr[mid+1])  // 添加判断
        __Merge(arr, l, mid, r);

}
```
改进之后对于随机的序列，时间会有所增加，因为判断语句本身也要耗时，因此对于不会出现基本有序序列的情况下，可以不用判断语句，对于有可能会出现基本有序序列的情况下，可以加上判断语句。

>改进2

对于所有的递归的排序算法都有的一种改进方式，就是递归边界改进。这里的递归边界是`l>=r`,即只有一个元素时，不用排序直接返回。可以改成剩下n个元素时，用归并排序，因为此时数组较小，有序的可能性比较大，而且数组比较小时，即n比较小，此时O(nlogn)和O(n^2)的差距没有那么大。
```c++
// 归并arr的[l,r]区域的数组
template<typename T>
void __MergeSort(T arr[], int l, int r){
    if(r-l<=15){  // 元素少时 做插入排序
        InsertSort(arr, r-l+1);
        return;
    }
    
    int mid = (l+r)/2;
    __MergeSort(arr, l, mid);
    __MergeSort(arr, mid+1, r);
    __Merge(arr, l, mid, r);

}
```
>迭代的方法自底向上

先划分成一个一个的，相邻两个做归并，然后划分成两个两个的，相邻两组做归并，最后剩下两个组，归并到一个组，此时序列有序。

```c++
void MergeSort(T arr[], int n){
    for(int sz=1; sz <=n; sz = sz * 2)  // size从1开始，每次扩大一倍，最后size是整个序列
        for(int i=0; i+sz<n; i+= 2* sz)  // 每次对arr[i,i+sz-1]和arr[i+sz,i+sz+sz-1]归并
            _Merge(arr, i, i+sz-1, min(i+sz+sz-1, n-1));
}
```
### 1.2.2 快速排序
选择一个元素，遍历序列把这个元素放到正确位置，此时这个元素把序列分成两段，即前面的数比这个数小，后面的数比这个数大。

>代码:
```c++
// 对arr[l,r]做partition操作，使得arr[l,p-1]<arr[p]<arr[p+1,r]，返回p点索引
template<typename T>
int __partition(T arr[], int l, int r){  // 注意这里返回Int型且数组arr后面要有[]
    T v = arr[l];
    int j=l;  // j是arr[l,p-1]这个数组的右边界指针，此时还没开始遍历，数组中没有值
    for(int i =l+1; i<=r; i++)  // i是遍历序列的指针，从l+1开始
        if(arr[i] < v){
            swap(arr[i], arr[j+1]);  // 如果小于v，则放到arr[l,p-1]这个数组中
            j++;
        }
    swap(arr[l], arr[j]); // j位置是小于v的最后一个元素 j+1是大于v的第一个元素，即p点

    return j;
}

template<typename T>
void __QuickSort(T arr[], int l, int r){
    if(l>=r)
        return;
    int p = __partition(arr, l, r);
    __QuickSort(arr, l, p-1);
    __QuickSort(arr, p+1, r);
}

template<typename T>
void QuickSort(T arr[], int n){
    __QuickSort(arr, 0, n-1);
}
```
>改进1

问题：v选在了一个不平衡的位置，使得分两段分得不好。

比如当整个元素完全有序时，每次选择第一个元素，此时没有任何一个元素小于他，因此只能分成一段，算法退化成O(n^2)。此时可以不选第一个元素作为v，而是随机选择一个数。
```c++
template<typename T>
int __partition(T arr[], int l, int r){ 
    swap(arr[l], arr[rand()%(r-l+1)+l]);  // 第一个数和随机位置的一个数交换

    T v = arr[l];
    int j=l;  
    for(int i =l+1; i<=r; i++)  
        if(arr[i] < v){
            swap(arr[i], arr[j+1]);  
            j++;
        }
    swap(arr[l], arr[j]); 

    return j;
}

template<typename T>
void __QuickSort(T arr[], int l, int r){
    if(l>=r)
        return;
    int p = __partition(arr, l, r);
    __QuickSort(arr, l, p-1);
    __QuickSort(arr, p+1, r);
}

template<typename T>
void QuickSort(T arr[], int n){
    srand(time(NULL)); // 随机种子
    __QuickSort(arr, 0, n-1);
}
```

>改进2

上述算法中，我们把小于v的值都放到前面，大于等于v的值都在后面，当序列中包含大量重复值时，有可能被分成两段不平衡的数组，再次退化到O(n^2)。可以用双路快排或者三路快排解决。

双路快排把数组分成小于等于v和大于等于v两部分，等于v的值被平均分到两段中。
```c++
template<typename T>
int __partition(T arr[], int l, int r){ 
    swap(arr[l], arr[rand()%(r-l+1)+l]);  

    T v = arr[l];
    
    // arr[l+1,i) <= v   arr(j,r] >= v i和j是正在考察的元素 不放在区间内
    int i=l+1, j=r;
    while(true){
        while(i<=r && arr[i]< v) i++;  // 如果i指向的元素小于v 则直接移动i 遍历完或者遇到大于等于v的值停止
        while(j>=l+1 && arr[j]>v) j--;  // 如果j指向的元素大于v 则直接移动j 遍历完或者遇到小于等于v的值停止
        if(i>j) break;  // 如果此时i已经越过j  则退出循环
        swap(arr[i], arr[j]);  // 否则交换 交换完后移动i j
        i ++;
        j --;
    }
    swap(arr[l], arr[j]);  // 最后j停在最后一个小于等于v的值上，和第一个元素交换，j位置前面都是小于等于v的
    
    return j;
}
```

三路快排把数组分成小于v，等于v和大于v三部分。
```c++
template<typename T>
void __QuickSort(T arr[], int l, int r){ 

    if(l>=r)
        return;

    swap(arr[l], arr[rand()%(r-l+1)+l]);  

    T v = arr[l];
    
    // arr[l+1,lt] < v  arr[lt+1, i)=v  arr[gt,r]>0  i是当前判断的位置
    int lt=l, i=l+1, gt=r+1;  //初始数组中都为0 i从l+1开始
    while(i<gt){
        if(arr[i]<v){  // 小于v时 交换到中间这个数组的第一个去 然后移动数组边界和遍历指针
            swap(arr[i], arr[lt+1]);
            lt++;
            i++;
        }
        else if(arr[i]>v){ // 大于v时交换到右边数组的前一个元素 然后移动数组边界 此时交换到i的元素仍是一个未排序元素 不用移动i
            swap(arr[i], arr[gt-1]);
            gt--;
        }
        else{  // 等于v时直接移动i
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
```

 # 2.树结构

