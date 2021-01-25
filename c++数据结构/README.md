

# 1.线型数据结构
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

# 2.树型数据结构
## 2.1 堆
### 2.1.1 实现一个堆
实现一个堆的类，输入是堆的容量，可以查询此时堆的大小和是否为空
```c++
#include <iostream>

using namespace std;

/*
完全二叉树：若设二叉树的深度为k，除第 k 层外，其它各层的结点数都达到最大个数，第k 层所有的结点都连续集中在最左边
堆：是一个完全二叉树，从1开始索引的话，i结点的左孩子是2i,右孩子是2i+1，父节点是i/2,因此可以不用树形结构，而用数组表示

*/
template<typename T>
class MaxHeap{

private:
    T *data;
    int count;

public:

    // 构造函数，传入堆的容量，构建一个数组
    MaxHeap(int capacity){
        data = new T[capacity+1];  // 从 1 开始索引 0没有元素
        count = 0;
    }

    ~MaxHeap(){
        delete[] data;
    }

    int size(){
        return count;
    }

    bool isEmpty(){
        return count==0;
    }
};

// 测试
int main(){

    // 前面是创建类 后面是调用构建函数
    MaxHeap<int> mh = MaxHeap<int>(100);
    cout<< mh.size() <<endl;

    return 0;

}
```
新增功能，可以向堆中插入一个新元素和删除一个元素(堆顶出列)
```c++
#include <iostream>
#include <cassert>

using namespace std;

template<typename T>
class MaxHeap{

private:
    T *data;
    int count;
    int capacity;


    // k不是根节点并且大于父节点的时候移动 
    void shiftUp(int k){
        while(k>1 && data[k]>data[k/2]){
            swap(data[k], data[k/2]);
            k /= 2;
        }
    }

    // 有子节点且子节点大于自身时移动，并且是和更大的那个子节点互换
    void shiftDown(int k){
        while(2*k<=count){
            int j = 2*k;  // j是要交换的结点，此时是左节点
            if(j+1 <= count && data[j+1] > data[j])  // 右结点存在且大于左节点，则j变为右结点
                j++;
            if(data[k] > data[j]) break;
            swap(data[k], data[j]);
            k = j;
        }

    }

public:

    
    MaxHeap(int capacity){
        data = new T[capacity+1];  
        count = 0;
        this->capacity = capacity;  // 给类中的capacity赋值
    }

    ~MaxHeap(){
        delete[] data;
    }

    int size(){
        return count;
    }

    bool isEmpty(){
        return count==0;
    }

    // 插入一个元素t，首先放到堆的最后位置，然后和父节点比较，如果大于父节点则交换位置，直到小于父节点
    void insert(T t){
        assert(count + 1 <= capacity);  
        data[count+1] = t;
        count++;
        shiftUp(count);  // 把加入的元素移动到合适位置 
    }

    // 取出堆顶元素，首先堆顶元素和堆尾元素互换，然后提出堆尾元素，此时的堆顶元素再和他的两个子节点比较，和最大的子节点交换，直至大于所有子节点
    T extractMax(){
        T res = data[1];  
        swap(data[1], data[count]);
        count--;
        shiftDown2(1);

        return res;
    }

    void show(){
        for(int i =1; i<=count; i++)
            cout<<data[i]<<' ';
    }
};

// 测试
int main(){

    MaxHeap<int> mh = MaxHeap<int>(100);

    srand(time(NULL));
    for(int i =0; i<7; i++)
        mh.insert( rand() % 100);

    mh.show();
    int res = mh.extractMax();
    cout<<'\n'<<res<<endl;
    mh.show();

    return 0;
}
```
### 2.1.2 堆排序 
堆排序就是把序列insert到堆中，然后一个一个extractMax出来

创建堆和取出元素都是O(nlogn)  但始终比快排和归并慢  用的多的是动态数据的维护
```c++
// 堆排序，输入数组和数组的大小，对数组进行排序
template<typename T>
void heapSort(T arr[], int n){
    MaxHeap<T> maxheap = MaxHeap<T>(n);

    for(int i=0; i<n; i++)
        maxheap.insert(arr[i]);
    for(int i=n-1; i>=0; i--)
        arr[i] = maxheap.extractMax();
}
```
可以优化创建堆的过程到O(n)

heapify创建堆:直接把数组复制到堆数组中，不采用insert的形式，此时堆数组的所有叶子结点，都已经是一个堆了，从第一个非叶子结点count/2开始做shifrDown，最后得到一个正确的堆

```c++
// 新增一个构造函数
MaxHeap(T arr[], int n){
    data = new T[n+1];  

    for(int i=0; i<n; i++)
        data[i+1] = arr[i];
    
    count = n;
    for(int i=count/2; i>0; i--)
        shiftDown(i);
}

// 基于heapify构造函数创建堆 然后依次输出堆顶元素做排序
template<typename T>
void heapSort2(T arr[], int n){
    MaxHeap<T> maxheap = MaxHeap<T>(arr, n);

    for(int i=n-1; i>=0; i--)
        arr[i] = maxheap.extractMax();
}
```
还可以优化取出元素extractMax中shiftDown的过程
```c++
// 用赋值替换交换
void shiftDown2(int k){
    T e = data[k];

    while(2*k<=count){
        int j = 2*k;  
        if(j+1 <= count && data[j+1] > data[j])  
            j++;

        if(e > data[j]) break;  // 子节点小于当前要重排序的结点值 终止
        data[k] = data[j];  //子节点覆盖了父节点，此时两个结点的值都是子节点的值，但是我们用k保存了此时要替换的子节点的序号，用e保存了它的值
        k = j;  // k保存新的要排序结点的序号
    }
    data[k] = e; // 最后不用替换了，所以把替换结点的值e赋值给替换结点当前的序号k
}
```
最后，不仅可以在时间上优化堆排序，还可以在空间上优化堆排序

堆的数据结构本身是一个数组，因此可以在数组上进行原地排序

```c++
// 对容量为n的数组arr的第k个元素做shiftdown
// 此时从0开始索引，左子节点为2*k+1
// k是要排序的结点  j是要互换的子节点
template<typename T>
void shiftDown(T arr[], int n, int k){
    T e = arr[k];
    while(2*k+1<n){
        int j = 2*k+1;
        if(j+1<n && arr[j+1]>arr[j]) j++;
        if(e > arr[j]) break;

        arr[k] = arr[j];
        k = j;  // k保存要排序结点的序号  e保存它的值
    }
    arr[k] = e;
}

// 注意，此时我们的堆是从0开始索引的
// 从第一个非叶子结点(最后一个元素的索引-1)/2开始heapify建堆  最后一个元素的索引 = n-1
// 第一个元素是堆中最大的，交换到数组末尾，然后对交换到前面的元素做shiftDown
template<typename T>
void heapSort3(T arr[], int n){

    // 建堆
    for(int i =(n-1-1)/2; i>=0; i--)
        shiftDown(arr, n, i);

    for(int i=n-1; i>=0; i--){
        swap(arr[0], arr[i]);
        shiftDown(arr, i, 0);  // !!!特别注意 这里arr的容量是i，因为i之后的元素是已经排好序的，不参与shiftdown,不然这些大的元素又会被替换到前面
    }
}
```

### 2.1.3 索引堆
堆在insert和extractMax时，不管是shiftUp还是shiftDown都要移动元素,元素可能是一个比较大的类，移动很耗时

一个数组建立成堆之后，就不能索引原来的元素了，因此希望有一种数据结构，既可以有数组的性质，又有堆的性质

索引堆用索引建堆  数据位置保持不变  从外看还是原来的数组  但是也可以按照索引来当作堆进行操作
```c++
template<typename T>
class IndexMaxHeap{

private:
    T *data;  // data对应外部数组
    T *indexes;  // indexes对应内部的堆 操作堆时只操作索引 因为根据索引可以找到原来应该操作的数据是哪一个 indexes[i]=j堆的i号结点是数组的j号结点

    int count;
    int capacity;


    // 操作堆的k号结点时要比较他和父节点的值，通过indexes[k]可以知道要操作的真实数据应该是数组中的第indexes[k]个元素
    void shiftUp(int k){
        while(k>1 && data[indexes[k]]>data[indexes[k/2]]){
            swap(data[indexes[k]], data[indexes[k/2]]);
            k /= 2;
        }
    }

    
    void shiftDown(int k){
        while(2*k<=count){
            int j = 2*k; 
            if(j+1 <= count && data[indexes[j+1]] > data[indexes[j]])  
                j++;
            if(data[indexes[k]] > data[indexes[j]]) break;
            swap(data[indexes[k]], data[indexes[j]]);
            k = j;
        }

    }

public:

    MaxHeap(int capacity){
        data = new T[capacity+1];  
        indexes = new T[capacity+1]
        count = 0;
        this->capacity = capacity;  
    }

    ~MaxHeap(){
        delete[] data;
        delete[] indexes;
    }

    int size(){
        return count;
    }

    bool isEmpty(){
        return count==0;
    }

    //  对外是一个数组的插入操作 即在位置i插入元素t
    // i位置原来不能有值
    void insert(int i, T t){
        i = i+1;  // 外部数组从0开始索引 内部从1开始索引
        assert(count + 1 <= capacity);  
        assert(i>=1 && i<=capacity);

        data[i] = t;
        indexes[count+1] = i;
        count++;

        shiftUp(count); 
    }

    T extractMax(){
        assert(count>0);

        T res = data[indexes[1]];  // indexes[1]代表堆中的1号结点在数组中的索引号 然后在data[]中取出它的值
        swap(data[indexes[1]], data[indexes[count]]);
        count--;
        shiftDown(1);  // 把堆中的1号索引的结点往下移

        return res;
    }
};
```
此时的索引堆可以有一些新的方法
```c++
// 像普通数组一样O(1)索引元素
T getItem( int i ){
    assert( i + 1 >= 1 && i + 1 <= capacity );
    return data[i+1];
}

// O(1)获得数组中最大的元素
T getMax(){
    assert( count > 0 );
    return data[indexes[1]];
}

// O(1)获得原数组中最大元素的索引号
int getMaxIndex(){
    assert( count > 0 );
    return indexes[1]-1;
}

// 将i号元素改为t
void change( int i , T t){

    i += 1;
    data[i] = t;  // i是在数组中的索引号 直接改变

    // 要修改堆，首先要找到它在堆中的索引号，即indexes[j]=i
    // 之后shiftUp(j), 再shiftDown(j)
    for( int j = 1 ; j <= count ; j ++ )
        if( indexes[j] == i ){
            shiftUp(j);
            shiftDown(j);
            return;
        }
}
```
这里change是O(nlogn)的，可以优化一下，使用反向查找表

indexes[i]  表示堆中的i号元素在数组中的位置

reverse[i]  表示数组中的i号元素在堆中的位置

change时已知数组中的元素i,reverse[i]即可知道在堆中的位置，此时再对此位置结点进行操作即可
```c++
// 最小索引堆
#ifndef MY_INDEXMINHEAP
#define MY_INDEXMINHEAP

#include <iostream>
#include <cassert>

using std::swap;

template<typename T>
class IndexMinHeap{
private:
    T *data;
    int *index;  // index[x]=i表示 堆中x位置是索引i的元素
    int *reverse;  // reverse[i]=x表示 索引i的元素在堆中的x位置

    int count;  // 当前堆中元素个数
    int capacity;  // 堆的容量

    // 最小堆 如果存在父节点且比父节点小 节点上移
    void shiftUp(int k){
        while(k>1 && data[index[k]] < data[index[k/2]]){
            swap(index[k], index[k/2]);  // 堆中位置对应的元素索引互换一下
            reverse[index[k/2]] = k/2;  // 元素索引在堆中的位置更新一下 也可以互换
            reverse[index[k]] = k;
            k /= 2;
        }
    }

    // 如果存在子节点 选择子节点中最小的互换
    void shiftDown(int k){
        while(k*2 <= count){
            int j = 2*k;
            if(j+1 <= count && data[index[j+1]] < data[index[j]]) j=j+1;
            if(data[index[k]] < data[index[j]]) break;
            swap(index[j], index[k]);
            reverse[index[j]] = j;
            reverse[index[k]] = k;
            k=j;
        }
    }

public:
    IndexMinHeap(int capacity){
        data = new T[capacity+1];
        index = new int[capacity+1];
        reverse = new int[capacity+1];

        count = 0;
        this->capacity = capacity;

        // 初始堆中没有元素 也没有元素在堆中
        for(int i=0; i<=capacity;i++){
            reverse[i]=0;
            index[i]=0;
        }   
    }

    ~IndexMinHeap(){
        delete[] data;
        delete[] index;
        delete[] reverse;
    }

    int size(){return count;}  // 返回索引堆中的元素个数
    bool isEmpty(){return count == 0;}  // 返回一个布尔值, 表示索引堆中是否为空

    // 插入一个索引为i的元素t
    void insert(int i, T t){
        assert(count<capacity);  // 容量够
        assert(i>0 && i<capacity);  // 判断索引是否有效 用户从0开始索引
        
        i+=1;  // 堆从1开始索引 所以外部传进来的索引要加1
        data[i] = t;  // t元素插入索引i的位置
        index[count+1]=i;  // 堆中count+1的位置是索引为i的元素 即把t插入了堆尾
        reverse[i] = count+1;  
        count++;

        shiftUp(count);  // 新插入的堆尾元素做调整
    }

    // 取出堆顶元素  在这里虽然没有删除data中的元素 但是认为元素已经没有了
    T extractMin(){
        T e = data[1];
        swap(index[1], index[count]);  // 交换堆顶堆尾
        reverse[index[1]] =1;  // 元素index[1]在堆中1
        reverse[index[count]] =0;  // 元素index[count]不再在堆中了 
        count--;  // 删除堆尾元素
        shiftDown(1);  // 堆顶元素下移调整

        return e;
    }

    // 取出堆顶元素的索引 元素从堆的角度看不存在了 但是返回索引后可以根据索引来操作
    int etractMinIndex(){
        int idx = index[1]-1;
        swap(index[1], index[count]);
        reverse[index[1]] =1;
        reverse[index[count]] =0;
        count--;
        shiftDown(1);

        return idx;
    }

    T getMin(){return data[index[1]];}  // 获取序列中最小的元素
    T getMinIndex(){return index[1]-1;}  // 获取序列中最小元素的索引
    bool contain(int i){return reverse[i+1] != 0;}  // 看索引i所在的位置是否存在元素
    T getItem(int i){return data[i+1];}  //  获取序列中索引为i的元素

    // 序列的位置i处插入一个元素t
    void change(int i, T t){
        i+=1;
        data[i]=t;
        
        // 先尝试向上调整 或者调整成功 或者没有变化 相对应的两种结果再向下调整时 
        // 或者没有变化 或者调整成功
        shiftUp(reverse[i]); // reverse[i]表示索引为i的元素在堆中的位置 
        shiftDown(reverse[i]);  
    }
};

#endif
```

## 2.2 二分搜索树
二分搜索树左孩子都比父节点小，右孩子都比父节点大
### 2.2.1 二分查找
```c++
// 在长为n的数组arr中查找target 返回索引 没有返回-1
template<typename T>
int binarySearch(T arr, int n, T target){

    int l=0, r=n-1, mid;

    while(l<=r){
        mid = (r-l)/2 + l; // 防止两个Int型相加越界

        if(arr[mid] == target) 
            return mid;

        if(arr[mid] > target)
            r = mid-1;
        else
            l = mid+1;
    }

    return -1;
}

// 递归实现
template<typename T>
int binarySearch2(T arr[], int l, int r, T target ){

    // 递归边界
    if(l>r)
        return -1;

    int mid = (r-l)/2 + l;
    if(arr[mid] == target) 
        return mid;
    if(arr[mid]>target)
        binarySearch2(arr, 0, mid-1, target);
    else
        binarySearch2(arr, mid+1, r, target);

}
```
### 2.2.2 实现
```c++
template<typename Key, typename Value>
class BST{

private:
    struct Node{
        Key key;
        Value value;
        Node *left;
        Node *right;

        Node(Key key, Value value){
            this->key = key;
            this->value = value;
            this->left = this->right = NULL;
        }
    };

    // 私有变量 一个根节点和一个节点数
    Node *root;
    int count;

public:
    BST(){
        root = NULL;
        count = 0;
    }

    ~BST(){
        // 需要后续遍历来析构
    }

    int size(){
        return count;
    }

    bool isEmpty(){
        return count==0;
    }
};
```
依次添加功能： 插入元素 查找元素 深度优先遍历  广度优先遍历  删除元素
```c++
#include <queue>

template<typename Key, typename Value>
class BST{

private:
    struct Node{
        Key key;
        Value value;
        Node *left;
        Node *right;

        Node(Key key, Value value){
            this->key = key;
            this->value = value;
            this->left = this->right = NULL;
        }
    };

    // 私有变量 一个根节点和一个节点数
    Node *root;
    int count;

public:
    BST(){
        root = NULL;
        count = 0;
    }


    ~BST(){
        destroy(root);
    }

    int size(){
        return count;
    }

    bool isEmpty(){
        return count==0;
    }

    // 在搜索二叉树中插入结点(key, value)
    void insert(Key key, Value value){
        root = insert(root, key, value);
    }

    // 查找二分搜索树中是否有键为key的结点
    bool contain(Key key){
        return contain(root, key);
    }

    // 查找key对应的值value，在之前要用contain函数确认在树中,不存在返回NULL
    Value* search(Key key){
        return search(root, key);
    }

    // 前序遍历
    void preOrder(){
        preOrder(root);
    }

    // 中序遍历
    void inOrder(){
        inOrder(root);
    }

    // 后序遍历
    void postOrder(){
        postOrder(root);
    }

    // 层序遍历
    void levelOrder(){
        levelOrder(root);
    }

    // 找到最小的结点
    Key minnode(){
        assert(count != 0);
        Node *node = minnode(root);
        return node->key;
    }

    // 找到最大的结点
    Key maxnode(){
        assert(count != 0);
        Node *node = maxnode(root);
        return node->key;
    }

    // 删除最小的结点
    void removeMin(){
        removeMin(root);
    }

    // 删除最大的结点
    void removeMax(){
        removeMax(root);
    }

    // 删除键值为key的结点
    void remove(Key key){
        remove(root, key);
    }

private:
    //在以*node为根的搜索二叉树中插入结点(key,value)
    Node* insert(Node *node, Key key, Value value){
        if(node = NULL){
            count++;
            return new Node(key, value);
        }

        if(node->key == key)
            node->value = value;
        else if(node->key > key)
            insert(node->left, key, value);
        else
            insert(node->right, key, value);

        return node;    
    }

    bool contain(Node *node, Key key){
        if(node = NULL)
            return false;
        
        if(node->key == key)
            return true;
        else if(k < node->key)
            contain(node->left, key);
        else
            contain(node->right, key);
    }

    Value* search(Node* node, Key key){
        if(node == NULL)
            return NULL;

        if(node->key == key)
            return &(node->value);  // 返回的是指针类型
        else if(k < node->key)
            search(node->left, key);
        else
            search(node->right, key);
    }

    void preOrder(Node *node){
        if(node != NULL){
            cout<<node->key>>endl;
            preOrder(node->left);
            preOrder(node->right);
        }
    }

    void inOrder(Node *node){
        if(node != NULL){
            inOrder(node->left);
            cout<<node->key>>endl;
            inOrder(node->right);
        }
    }

    void postOrder(Node *node){
        if(node != NULL){
            postOrder(node->left);
            postOrder(node->right);
            cout<<node->key>>endl;
        }
    }

    void destroy(Node *node){
        if(node != NUll){
            destroy(node->left);
            destroy(node->right);

            delete node;
            count--;
        }
    }

    // 层序遍历 先把根结点入队，然后每弹出一个结点，就把这个结点的左右结点依次入队
    void levelOrder(Node *node){
        if(node == NULL)
            return;

        queue<Node *> q;
        q.push(node);

        while(!q.empty()){
            Node * n = q.front();
            q.pop();
            cout<<n->key<<endl;

            if(n->left != NULL)
                q.push(n->left);
            if(n->right != NULL)
                q.push(n->right);
        }
    }

    // 左子树为空 则找到最小的结点 否则一直向左寻找
    Node* minnode(Node *node){
        if(node->left == NULL)  
            return node;

        return minnode(node->left);
    }

    Node* maxnode(Node* node){
        if(node->right == NULL)
            return node;

        return maxnode(node->right);
    }

    Node* removeMin(Node* node){
        // 如果找到最小结点，返回它的右子树
        if(node->left == NULL){
            Node *r = node->right;
            delete node;
            count --;
            return r;
        }

        // 删除了最小节点的左子树重新挂载到左指针上
        node->left = removeMin(node->left);

        return node;
    }

    Node* removeMax(Node* node){
        if(node->right == NULL){
            Node* l = node->left;
            delete node;
            count--;
            return l;
        }

        node->right = removeMax(node->right);
        
        return node;
    }

    Node* remove(Node* node, Key key){
        if(node == NULL)
            return node;
        
        if(key < node->key){
            node->left = remove(node->left, key);
            return node;
        }
        else if(key > node->key){
            node->right = remove(node->right, key);
            return node;
        }
        // 找到要删除的结点了
        else{
            // 1. 没有左子树
            if(node->left == NULL){
                Node* r = node->right;
                delete node;
                count--;
                return r;
            }
            // 2. 没有右子树
            else if(node->right == NULL){
                Node* l = node->left;
                delete node;
                count--;
                return l;
            }
            // 3. 左右子树都有,找到前驱或者后继代替
            // 这里找后继，即大于该结点的最小的结点，即右子树中最小的结点
            else{
                Node* post = new Node(minnode(node->right));  // 创建一个和后继一样的新节点
                count ++;

                post->right = removeMin(node->right);  // 结点右指针指向删除了后继的右子树 删除时会count-- 所以前面++
                post->left = node->left;

                delete node;
                count--;

                return post;
            }
        }
    }
};
```
搜索二叉树的缺陷：对于有序如递增序列，向树插入新节点会一直在结点的右孩子插入，最后导致树退化成一个链表

可以使用平衡二叉树，平衡二叉树是左右高度不超过1的搜索二叉树，其中一种实现形式就是红黑树
## 2.3 并查集
并查集可以非常高效的回答结点是否连接在一起的问题,而不求出具体的路径

并 union(p,q)  把p q并到一个组中

查 find(p)  查找p在哪个组  isConnected(p,q)  判断p q是否连接

### 2.3.1 线性数据结构
定义一个并查集数组，数组中保存的就是当前元素属于哪个组，这里不单独使用包含数据的数组，就把数组的索引当作要分组的数据

这种方式叫quick find  即查找很快  并很慢
```c++
class unionFind{

private:
    int* id;
    int count;

public:
    unionFind(int n){
        id = new int[10];
        count = n;

        // 初始化时每个元素自己一个组 互不连通
        for(int i=0; i<n; i++)
            id[i] = i;
    }

    ~unionFind(){
        delete[] id;
    }

    // 查找元素p所在的索引  O(1)
    int find(int p){
        return id[p];
    }

    // 判断p q是否在一个组
    bool isConnected(int p, int q){
        return find(p) == find(q);
    }

    // 连接p q O(n)
    void unionE(int p, int q){
        int pid = find(p);
        int qid = find(q);

        if(pid == qid)
            return;

        // 连通pq 要把所在的两个组合并 因此要遍历所有元素 找到属于其中一个组的 组号都改成另一个组的
        for(int i=0; i<count; i++)
            if(id[i] == pid)
                id[i] = qid;
        
    }
};
```

### 2.3.2 树型数据结构
为了减少合并时的时间复杂的，用数组构建一棵指向父节点的树，两个元素的根节点相同，即属于同一组。
```c++
class unionFind{

private:
    int* parent;
    int count;

public:
    unionFind(int n){
        parent = new int[10];
        count = n;

        // 初始化时每个元素自己一个组 互不连通
        for(int i=0; i<n; i++)
            parent[i] = i;
    }

    ~unionFind(){
        delete[] parent;
    }

    // 查找元素p所在的索引  O(h) h是这个组的树高
    // parent[p] == p 时代表是根结点 即当且的组号
    int find(int p){
        while(parent[p] != p)
            p = parent[p];

        return p;
    }

    // 判断p q是否在一个组
    bool isConnected(int p, int q){
        return find(p) == find(q);
    }

    // 连接p q O(h) h是两个组中高的那个树的树高
    void unionE(int p, int q){
        int pid = find(p);
        int qid = find(q);

        if(pid == qid)
            return;

        // 一个根指向另一个根 合并成一组
        parent[pid] = qid;
        
    }
};
```
这种方法有一个不好的地方，就是时间复杂度和每一组的树高h相关，但是合并两个树的时候是随机的合并的，如果是矮树指向高树则高度不变，相反则高度加一，这使得树可能越来越高，最坏情况时间复杂度退化成O(n)，因此在合并两个树时，要判断树高，将矮树的父亲指针指向高树
```c++
class unionFind{

private:
    int* parent;
    int* rank;  // 记录每一棵树的高度
    int count;

public:
    unionFind(int n){
        parent = new int[10];
        rank = new int[10];
        count = n;

        // 初始化时每个元素自己一个组 互不连通 树高都为1
        for(int i=0; i<n; i++)
            parent[i] = i;
            rank[i] = 1;
    }

    ~unionFind(){
        delete[] parent;
        delete[] rank;
    }

    // 不做修改
    int find(int p){
        while(parent[p] != p)
            p = parent[p];

        return p;
    }

    // 不做修改
    bool isConnected(int p, int q){
        return find(p) == find(q);
    }

    // 合并时矮树指向高树，等高时随意指向，但被指向的那棵树rank+1
    void unionE(int p, int q){
        int pid = find(p);
        int qid = find(q);

        if(pid == qid)
            return;

        if(rank[pid] < rank[qid])
            parent[pid] = qid;
        else if(rank[qid] < rank[pid])
            parent[qid] = pid;
        else{
            parent[qid] = pid;
            rank[pid] ++;
        }  
    }
};
```
### 2.3.3 路径压缩
版本一：查找时，如果当前元素自身不是根节点，那么让它指向它父亲的父亲

情况一： 它的父亲不是根节点，那么指向的是另一个更靠近根的结点

情况二：它的父亲是根节点，根节点的父亲还是自己，因此直接指向了根节点

这种方式在查找时，不再遍历整棵树，而是跳跃遍历，因此查找更快，而且查找过程中可能减少树高，降低整体的查找和合并的时间复杂度
```c++
int find(int p){
    while(parent[p] != p){
        parent[p] = parent[parent[p]];
        p = paren[p];
    }

    return p;
}
```
版本二： 查找时，如果当前元素自身不是根节点，那么让他直接指向根节点，最后所有树高都为2，因为所有元素都是直接指向根节点的
```c++
int find(int p){
    while(parent[p] != p)
        parent[p] = find(parent[p]);  // 不能find(p) 死循环了

    return parent[p];  // 最后要返回的是p所在的根节点
}
```
理论上版本二更快 ，实际应用中版本一更快

路径压缩后的树型并查集所有操作的时间复杂度都是近乎于1的

# 3.图型数据结构
图可以分为有向图和无向图，还可以分为有权图和无权图。

无向完全图任一结点都连接了所有的结点，即E=V*(V-1)/2

## 3.1 图基础
### 3.1.1 邻接矩阵和邻接表
图有两种数据结构可以表示，一般稠密图用邻接矩阵，稀疏图用邻接表
```c++
// 邻接矩阵
class denseGraph{

private:
    int v, e;  // 记录结点和边的数量
    bool directed;  // 是否是有向图
    vector<vector<bool>> g;  // 邻接矩阵

public:
    denseGraph(int n, bool directed){
        assert(n>0);

        this->v=n;
        this->e=0;
        this->directed = directed;
        this->g = vector<vector<bool>>(n, vector<bool>(n, false));
    }

    ~denseGraph(){}

    // 图中的节点数
    int numV(){
        return v;
    }

    // 图中的边数
    int numE(){
        return e;
    }

    // 判断是否有p到q的连接
    bool hasE(int p, int q){
        assert(p>0 && p<=v && q>0 && q<=v);  // 确定没有越界

        return g[p][q];
    }

    // 添加p到q的边
    void addE(int p, int q){
        assert(p>0 && p<=v && q>0 && q<=v);

        if(hasE(p,q))
            return;

        g[p][q] = true;

        if(!directed)
            g[q][p] = true;

        e++;
    }
};
```
```c++
// 邻接表
class sparseGraph{

private:
    int v, e;  // 记录结点和边的数量
    bool directed;  // 是否是有向图
    vector<vector<int>> g;  // 邻接表 存储节点号 int型

public:
    sparseGraph(int n, bool directed){
        assert(n>0);

        this->v=n;
        this->e=0;
        this->directed = directed;
        this->g = vector<vector<int>>(n, vector<int>());  // g初始化为n个空的vector, 表示每一个g[i]都为空, 即没有任和边
    }

    ~sparseGraph(){}

    // 图中的节点数
    int numV(){
        return v;
    }

    // 图中的边数
    int numE(){
        return e;
    }

    // 判断是否有p到q的连接
    bool hasE(int p, int q){
        assert(p>0 && p<=v && q>0 && q<=v);  // 确定没有越界

        for(int i=0; i<g[p].size(); i++)
            if(g[p][i] == q)
                return true;
        return false;
    }

    // 添加p到q的边
    void addE(int p, int q){
        assert(p>0 && p<=v && q>0 && q<=v);

        if(hasE(p,q))
            return;

        g[p].push_back(q);  // 添加边

        if(!directed)
            g[q].push_back(p);

        e++;
    }
};
```
### 3.1.2 遍历邻边
遍历图G中点p的邻边，直接遍历邻接矩阵或者邻接表中点p这一行的元素即可
```c++
// 邻接矩阵
vector<bool> adjE(int p){
    vector<bool> adje = g[p];
    return adje;
}

// 邻接表
vector<int> adjE(int p){
    vector<int> adje = g[p];
    return adje;
}
```
### 3.1.3 图的深度优先遍历
深度优先遍历就是从一个点开始，访问它的所有邻接结点，对于没有访问过的进行递归的深度优先遍历。

深度优先遍历的时间复杂度 

邻接矩阵:O(v^2)  因为要遍历每一个点  每一个点又要遍历其他点确认是否相邻

邻接表:O(v+e)  遍历每一个点 每个点只要遍历他的邻接结点 即边数
```c++
// 深度优先遍历求连通块
template<typename Graph>
class component{

private:
    Graph &G;  // 要遍历的图的引用
    bool *visited;  // 标记结点是否访问过
    int count;  // 记录连通块

    // 从点p开始深度优先遍历所在连通块
    void dfs(int p){
        visited[p] = true;  // 当前结点标记已经遍历到了
        vector<bool> adj = G.adjE(p);  // vector必须指定类型 邻接矩阵返回的是bool型 邻接表是int型
        for(int i=0;i<adj.size();i++){  // 遍历它的邻接矩阵或者邻接表
            if(adj[i] && !visited[i])  // 邻接表存储的都是邻接结点 邻接矩阵标记为真值的才是邻接结点 保证是邻接结点并且没有被访问
                dfs(i);  // 遍历这个结点
        }
    }

public:
    component(Graph &g):G(g){  // 因为G是一个引用 要进行初始化 不能赋值，因此这里先用&g来引用一个图 然后用g初始化G
        visited = new bool[G.numV()];  // 开辟一段bool空间 大小是图的节点数 用于保存visited
        for(int i=0; i<G.numV(); i++)  // 全部未访问过
            visited[i] = false;
        count = 0;
    }

    ~component(){
        delete[] visited;
    }

    int getCount(){
        for(int i=0; i<G.numV(); i++)  // 遍历所有结点
            if(!visited[i]){  // 如果这个结点没有遍历过
                dfs(i);  // 深度优先遍历它  函数完成时已经遍历了i所在连通块的所有结点
                count++;  // 连通块数目加一 表示已经遍历完了一个连通块
            }
        return count;
    }
};
```
判断两个结点是否相连接，多维护一个记录结点所在连通块序号的数组就可以了 id[p]==id[q]代表相连
```c++
// 深度优先遍历求路径 不是最短路径
template<typename Graph>
class path{

private:
    Graph &G;  
    bool *visited;  
    int *from; // 记录从哪个结点遍历过来的
    int s;  // 原点  


    void dfs(int p){
        visited[p] = true;  

        vector<int> adj = G.adjE(p);  // 这里返回一个邻接表的元素 用Int型
        for(int i=0;i<adj.size();i++){  
            if(!visited[i]){
                from[i] = p;  // 遍历i结点之间 标记i结点是由p结点遍历过去的
                dfs(i); 
            }      
        }
    }



public:
    // 构造函数输入图 和 要求路径的原点
    path(Graph &g, int s):G(g){  
        assert(s>=0 && s<G.numV());

        visited = new bool[G.numV()];  
        from = new int[G.numV()];  

        for(int i=0; i<G.numV(); i++){
            visited[i] = false;
            from[i] = -1;  // 初始都没有遍历过 因此没有上一个结点
        }
        this->s = s;

        dfs(s);  // 初始化时直接得到与s相关的所有路径
    }

    ~path(){
        delete[] visited;
        delete[] from;
    }

    // 判断w是否与s相连
    // 遍历过就一定相连
    bool hasPath(int w){
        assert(w>=0 && w<G.numV());

        return visited[w];
    }

    // 得到s到w的路径
    void getPath(int w, vector<int> &vec){
        
        stack<int> stk;

        // 从w开始 把from数组放进栈中 
        while(w){
            stk.push(w);
            w = from[w];
        }

        vec.clear();

        // 依次出栈 因为进去的是from路径 因此出来的是to路径 即要求的路径
        while(!stk.empty()){
            vec.push_back(stk.top());
            stk.pop();
        }
    }

    // 打印路径
    void showPath(int w){
        vector<int> vec;
        getPath(w, vec);

        for(int i=0; i<vec.size(); i++)
            cout<<vec[i]<<' ';
    }
};
```

### 3.1.4 图的广度优先遍历
从一个点开始进行广度优先遍历，先把这个点入队，然后开始对队进行操作。即当队不为空时，就弹出队首元素，并把它没有入队的邻接结点全部入队。

广度优先遍历时间复杂度和深度优先遍历一致

广度优先遍历可以求最短路径，因为他是从原点开始一层层的遍历，求原点到某一点的路径时，也是一层层的寻找，每层只经过一个结点，即最短
```c++
template<typename Graph>
class shortestPath{

private:
    Graph &G;  
    bool *visited;  
    int *from; // 记录从哪个结点遍历过来的
    int *dis;  // 记录到原点的距离
    int s;  // 原点

    // 从点p开始广度优先遍历
    void bfs(int p){
        queue<int> q;

        q.push(p);  
        visited[p] = true;  // 入队即访问过了
        

        while(!q.empty()){
            int v = q.front();  // 取队首元素
            q.pop();

            vector<int> adj = G.adjE(v);  // 遍历队首元素的所有邻接结点 没访问过的入队
            for(int i=0; i<adj.size(); i++){
                if(!visited[i]){
                    q.push(i);  
                    visited[i] = true;  // 入队即访问过了
                    from[i] = v;
                    dis[i] = dis[v] + 1;  // 到原地的距离是此时队首元素加1
                }
            }
        }

    }

public:
    shortestPath(Graph &g, int s):G(g){  
        assert(s>=0 && s<G.numV());

        visited = new bool[G.numV()];  
        from = new int[G.numV()];  
        dis = new int[G.numV()]; 

        for(int i=0; i<G.numV(); i++){
            visited[i] = false;
            from[i] = -1;  
            dis[i] = -1;  // 初始到原点距离都无限大 设为-1
        }

        dis[s] = 0;  // 原点到自己的距离是0

        this->s = s;

        bfs(s);  
    }

    ~shortestPath(){
        delete[] visited;
        delete[] from;
        delete[] dis;
    }

    // 判断w是否与s相连
    bool hasPath(int w){
        assert(w>=0 && w<G.numV());

        return visited[w];
    }

    // 得到s到w的路径
    void getPath(int w, vector<int> &vec){
        
        stack<int> stk;

        while(w){
            stk.push(w);
            w = from[w];
        }

        vec.clear();

        while(!stk.empty()){
            vec.push_back(stk.top());
            stk.pop();
        }
    }

    // 打印路径
    void showPath(int w){
        vector<int> vec;
        getPath(w, vec);

        for(int i=0; i<vec.size(); i++)
            cout<<vec[i]<<' ';
    }
};
```

## 3.2 最小生成树 
带权无向图  找V-1条边 连接v个顶点 使得生成树权值最小
### 3.2.1 带权图的边
```c++
#ifndef MY_EDGE
#define MY_EDGE

#include <iostream>
#include <cassert>

template<typename Weight>  // 权重可能是不同的数据类型
class Edge{
private:
    int a,b;
    Weight weight;

public:
    Edge(int a, int b, Weight weight){
        this->a = a;
        this->b = b;
        this-> weight = weight;
    }

    // 空的构造函数 所有成员变量都取默认值
    Edge(){}

    ~Edge(){}

    int v(){ return a;} // 返回第一个顶点
    int w(){ return b;} // 返回第二个顶点
    Weight wt(){ return weight;}    // 返回权值

    // 给定一个顶点返回另一个顶点
    int other(int x){
        assert(x==a || x==b);
        return x==a ? b : a;
    }

    // 边的大小比较, 是对边的权值的大小比较
    bool operator < (Edge<Weight> &e){return weight < e.wt();}
    bool operator > (Edge<Weight> &e){return weight > e.wt();}
    bool operator <= (Edge<Weight> &e){return weight <= e.wt();}
    bool operator >= (Edge<Weight> &e){return weight >= e.wt();}
    bool operator == (Edge<Weight> &e){return weight == e.wt();}


};

#endif
```
### 3.2.2 prim算法 
prim算法 O(ElogV)  最小索引堆

prim算法首先把原点加入生成树中，然后把生成树能达到各个节点的最短边加入最小堆中(没有变达到就不加入),然后从最小堆中取出权值最小的边，连接的节点加入生成树中，然后更新生成树能达到各个节点的最短边并维护堆，循环直至所有边都加入了生成树中。
```c++
#ifndef MY_PRIM
#define MY_PRIM

#include <iostream>
#include <cassert>
#include <vector>
#include "Edge.h"
#include "IndexMinHeap.h"

using namespace std;

template<typename Graph, typename Weight>
class prim{

private:
    Graph &G;                     // 图的引用
    IndexMinHeap<Weight> ipq;     // 最小索引堆  保存生成树到各个节点的最短边的权值
    vector<Edge<Weight>*> edgeTo; // 保存生成树到这个节点的最短边
    bool *marked;                 // 表示是否已经访问过了 即是否已经加入生成树
    vector<Edge<Weight>> mst;     // 最小生成树所包含的所有边
    Weight mstWeight;             // 最小生成树的总权值

    void visit(int v){
        marked[v] = true;  // 加入生成树中
        vector<Edge<Weight>> adj = G.adjE();  // 所有邻接边
        for(int i=0; i<adj.size(); i++){
            int w = adj[i].other(v);  // w是边另一头的节点
            if(!marked[w]){  // 如果节点没有在生成树中
                if(!edgeTo[w]){  // 如果还没有保存到这个节点的最短边
                    edgeTo[w] = &adj[i];  // 当前边就保存为这个节点的最短边
                    ipq.insert(w, adj[i].wt());  // 保存当前边的权值
                }
                else if(adj[i].wt() < edgeTo[w].wt()){  // 如果保存过生成树到这个节点的最短边 但是当前的边更短 则替换
                    edgeTo[w]=&adj[i];
                    ipq.change(w, adj[i].wt());
                }   
            }
        }
    }

public:
    prim(Graph &g):G(g),ipq(IndexMinHeap<double>(g.V())){
        assert(g.E()>0);

        marked = new bool[G.V()];
        for(int i=0; i<G.V();i++){
            marked[i]=false;
            edgeTo.push_back(nullptr);
        }

        mst.clear();

        //prim
        visit(0);
        while(!ipq.isEmpty()){
            int v = ipq.etractMinIndex();  // 取出此时生成树可以连接到的边中 权值最小的那个节点序号
            mst.push_back(*edgeTo[v]);  // 把到这个节点的这条边加入到生成树中
            visit(v);  // 访问这个新加入的节点
        }

        // 统计最小生成树的总权值
        mstWeight = mst[0].wt();
        for( int i = 1 ; i < mst.size() ; i ++ )
            mstWeight += mst[i].wt();
    }

    ~prim(){
        delete[] marked;
    }

    // 返回最小生成树
    vector<Edge<Weight>> mstEdges(){
        return mst;
    };
    
    // 返回最小生成树的总权值
    Weight result(){
        return mstWeight;
    };
};

#endif
```
### 3.2.3 kruskal算法
kruskal算法 O(ElogE)  并查集

kruskal先把所有边按权值从小到大排序，然后每次加入一条边不构成环即可，直至有v-1条边加入树中
```c++
template <typename Graph, typename Weight>
class kruskal{
private:
    vector<Edge<Weight>> mst;  // 存所有边
    Weight mstWeight;  // 总权值

public:
    kruskal(Graph &g){
        minHeap<Edge<Weight>> pq(g.E());  // 最小堆存所有边

        for(int v=0; v<g.V(); v++){  // 遍历所有结点
            vector<Edge<Weight>> adj = g.adjE(v);  
            for(int i=0;i<adj.size();i++){  // 结点的所有邻边
                if(adj[i].v() < adj[i].w())  // 无权图中v->w w->v 是一条边 为了防止存两次 只在v<w时存一次
                    pq.insert(adj[i]);
            }
        }

        unionFind uf = unionFind(g.V());  // 并查集记录连接情况

        while(!pq.isEmpty() && mst.size() < g.V()-1){
            Edge<Weight> e = pq.extractMin();
            if(uf.isConnected(e.v(), e.w()))  // 这条边的两个顶点如果已经相连接了 说明已经在树中 
                continue;
            mst.push_back(e);
            uf.unionE(e.v(), e.w());
        }

    }

    ~kruskal(){}

    // 返回最小生成树的所有边
    vector<Edge<Weight>> mstEdges(){
        return mst;
    }

    // 返回最小生成树的权值
    Weight result(){
        return mstWeight;
    }
};
```
## 3.3 最短路径
带权图 有向无向均成立 单源最短路径

### 3.3.1 dijkstra算法 
不能有负权变  O(ElogV)  最小索引堆

### 3.3.2 bellman-ford算法 

可以有负权变 不能有负权环

## 3.4 补充：

拓扑排序 解决有向无环图的单源最短路径问题

floyed算法解决无负权环的所有对最短路径算法