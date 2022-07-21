# 排序算法

## 快速排序

1．先从数列中取出一个数作为基准数。

2．分区过程，将比这个数大的数全放到它的右边，小于或等于它的数全放到它的左边。

3．再对左右区间重复第二步，直到各区间只有一个数。

快速排序的一些**改进方案**：

![[公式]](https://www.zhihu.com/equation?tex=%281%29) 将快速排序的递归执行改为非递归执行

![[公式]](https://www.zhihu.com/equation?tex=%282%29) 当问题规模 ![[公式]](https://www.zhihu.com/equation?tex=n) 较小时 ![[公式]](https://www.zhihu.com/equation?tex=%28n+%5Cle+16%29)  ,采用直接插入排序求解

![[公式]](https://www.zhihu.com/equation?tex=%283%29) 每次选取 ![[公式]](https://www.zhihu.com/equation?tex=prior) 前将数组打乱

![[公式]](https://www.zhihu.com/equation?tex=%284%29) 每次选取 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BE%5Bfirst%5D%2BE%5BLast%5D%7D%7B2%7D) 或 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BE%5Bfirst%5D%2BE%5Blast%5D%2BE%5B%28first%2Blast%29%2F2%5D%7D%7B3%7D) 作为 ![[公式]](https://www.zhihu.com/equation?tex=prior)   

```python
def quick(lst,left, right):
    if right <= left:
        return
    
    i = left
    j = right
    # 取第一个数作为pivot
    mid = lst[i]
    
    while i < j:
        #先从j向左找 小于 mid 的元素，此时 i 为空位
        while i < j and lst[j] >= mid:
            j -= 1
        #交换， 此时可能 i == j 但不要紧
        lst[i] = lst[j]
        # lst[i], lst[j] = lst[j], lst[i]
        
        #从i向右找 大于 mid 此时j为空位
        while i < j and lst[i] <= mid:
            i += 1
        # 交换
        lst[j] = lst[i]
        # lst[i], lst[j] = lst[j], lst[i]
    # i == j, 在i处加入mid，并递归mid左侧和mid右侧
    lst[i] = mid
    quick(lst, left, i-1)
    quick(lst, i+1, right)
```



## 归并排序

+ 不断将数组从中点位置划分开（即二分法），将整个数组的排序问题转化为子数组的排序问题
+ 划分到子数组长度为 1 时，开始向上合并，不断将 **较短排序数组** 合并为 **较长排序数组**，直至合并至原数组时完成排序；

```python
def merge_sort(nums, temp, l, r):
    # 结束
    if l >= r:
        return
    
    mid = (l + r) // 2
    merge_sort(nums, temp, l, mid)
    merge_sort(nums, temp, mid + 1, r)
    # 两段已经有序,合并阶段
    # 使用temp暂存 l -- r
    i = l 
    j = mid + 1
    temp[l: r + 1] = nums[l: r + 1]
    for k in range(l, r + 1):
        # 左边已经遍历完, 直接把右边加入
        if i == mid + 1:
            nums[k] = temp[j]
            j += 1 
        # 右边遍历完, 直接把左边加入
        elif j == r + 1:
            nums[k] = temp[i]
            i += 1
        
        elif temp[i] <= temp[j]:
            nums[k] = temp[i]
            i += 1
        else:
            nums[k] = temp[j]
            j += 1
    # 合并完成，l -- r 有序
    return
```



# 二分搜索

+ python中的bisect.bisect()
+ 查找应该插入的位置，即第一个比x大的数字的下标

```python
def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    
    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1
    return lo
```

+ python中的bisect.bisect_left()
+ 查找应该插入的位置，即第一个大于等于的数字的下标

```python
def bisect_left(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo
```



# 堆排序

+ python

```python
#基于堆的优先队列
class PriorQueueError(ValueError):
    pass

class PriQueHeap():
    def __init__(self, elist=[]):
        self._elem = []
        self.buildheap(list(elist))
        
    def is_empty(self):
        return not self._elem
    
    def peek(self):
        if self.is_empty():
            raise PriorQueueError('in peek')
        return self._elem[0]
    
    #入队
    def enqueue(self, e):
        self._elem.append(e)
        self.siftup()
    
    
    
    #向上筛选，从最后一个元素
    def siftup(self):
        child = len(self._elem) - 1
        while child > 0:
            parent = (child - 1) // 2 #父节点的关系
            if self._elem[parent] > self._elem[child]:
                self._elem[parent], self._elem[child] = self._elem[child], self._elem[parent]
                child = parent
            else:
                break
    
    #弹出元素
    def dequeue(self):
        if self.is_empty():
            raise PriorQueueError('in peek')
        
        head_val = self._elem[0]
        self._elem[0] = self._elem[-1]#把最后一个点放在头上
        self._elem.pop()#把最后一个弹出去
        self.siftdown()#向下筛选一次
        return head_val
        
    def siftdown(self):
        parent = 0
        lth = len(self._elem) - 1
        
        while 2 * parent + 1 <= lth:
            left_child = 2 * parent + 1
            right_child = 2 * parent + 2
            #没有右节点，只需要和左节点再比较一次
            if right_child > lth and self._elem[parent] > self._elem[left_child]:
                self._elem[parent], self._elem[left_child] = self._elem[left_child], self._elem[parent]
                break
            #parent 最小，直接结束
            if self._elem[parent] <= self._elem[left_child] and self._elem[parent] <= self._elem[right_child]:
                break
            #找left 和 right 中小的那一个
            elif self._elem[left_child] <= self._elem[right_child]:#left 小
                self._elem[parent], self._elem[left_child] = self._elem[left_child], self._elem[parent]
                parent = left_child
            else:
                self._elem[parent], self._elem[right_child] = self._elem[right_child], self._elem[parent]
                parent = right_child
    #构建堆
    def buildheap(self, lst):
        for val in lst:
            self.enqueue(val)
```



+ Java

```java
public class HeapTest {
    public static void main(String[] args) {
        int[] arr = new int[]{3,5,7,8,1,1,2,4,4,10,20};
        PriQue pq = new PriQue(100);
        for (int num : arr) {
            pq.add(num);
        }
        while (!pq.isEmpty()) {
            System.out.println(pq.poll());
        }
    }
}


// int 型
class PriQue {
    int size;
    int[] arr;
    int max_size;

    public PriQue(int _max_size) {
        size = 0;
        max_size = _max_size;
        arr = new int[max_size];
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public int peek() {
        if (isEmpty()) {
            throw new RuntimeException("empty");
        }
        return arr[0];
    }

    public void add(int x) {
        arr[size] = x;
        size += 1;
        siftup();
    }

    public int poll() {
        if (isEmpty()) {
            throw new RuntimeException("empty");
        }
        int min = arr[0];
        arr[0] = arr[size - 1];
        size -= 1;
        siftdown();
        return min;
    }

    void siftup() {
        int child = size - 1;
        while (child > 0) {
            int parent = (child - 1) / 2;
            if (arr[parent] > arr[child]) {
                int temp = arr[parent];
                arr[parent] = arr[child];
                arr[child] = temp;
            }else {
                break;
            }
            child = parent;
        }
    }

    void siftdown() {
        int parent = 0;
        while (2 * parent + 1 < size) {
            int left_child = 2 * parent + 1;
            int right_child = 2 * parent + 2;
            int small_child = left_child;
            if (right_child < size && arr[right_child] < arr[left_child]) {
                small_child = right_child;
            }
            if (arr[parent] <= arr[small_child]) {
                break;
            }
            int temp = arr[parent];
            arr[parent] = arr[small_child];
            arr[small_child] = temp;
            parent = small_child;
        }
    }
}
```

