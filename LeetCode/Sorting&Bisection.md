# 排序算法

## 快速排序

1．先从数列中取出一个数作为基准数。

2．分区过程，将比这个数大的数全放到它的右边，小于或等于它的数全放到它的左边。

3．再对左右区间重复第二步，直到各区间只有一个数。  

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

