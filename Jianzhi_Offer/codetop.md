



## [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

+ 迭代

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None 
        cur = head 
        while cur is not None:
            next_node = cur.next
            cur.next = pre 
            pre = cur 
            cur = next_node 
        return pre 
```



+ 递归

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head 
        new_head = self.reverseList(head.next)
        head.next.next = head 
        head.next = None 
        return new_head
```





## [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        hset = set()
        max_len = 0
        n = len(s)
        i = 0
        j = 0
        while j < n:
            while j < n and s[j] not in hset:
                hset.add(s[j])
                j += 1
            max_len = max(max_len, j - i)
            hset.remove(s[i])
            i += 1
        return max_len
```



## [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

快速排序



```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        self.qsearch(nums, 0, len(nums) - 1, k)
        return nums[-k]


    def qsearch(self, nums, left, right, k):
        if right <= left:
            return 
        i = left
        j = right 
        # 随机index
        rand_idx = random.randint(left, right)
        nums[i], nums[rand_idx] = nums[rand_idx], nums[i]
        pivot = nums[i]
        while i < j:
            # 找小于pivot
            while i < j and nums[j] >= pivot:
                j -= 1
            nums[i] = nums[j]
            # 大于pivot
            while i < j and nums[i] <= pivot:
                i += 1
            nums[j] = nums[i]
        nums[i] = pivot 
        if right - i + 1 == k:
            return 
        elif right - i + 1 < k:
            self.qsearch(nums, left, i - 1, k - (right - i + 1))
        else:
            self.qsearch(nums, i + 1, right, k)
```







## [912. 排序数组](https://leetcode.cn/problems/sort-an-array/)

快速排序

```python
import random 
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.qsort(nums, 0, len(nums) - 1)
        return nums

    def qsort(self, nums, left, right):
        if right <= left:
            return 
        i = left
        j = right 
        # 随机index
        rand_idx = random.randint(left, right)
        nums[i], nums[rand_idx] = nums[rand_idx], nums[i]
        pivot = nums[i]
        while i < j:
            # 找小于pivot
            while i < j and nums[j] >= pivot:
                j -= 1
            nums[i] = nums[j]
            # 大于pivot
            while i < j and nums[i] <= pivot:
                i += 1
            nums[j] = nums[i]
        nums[i] = pivot 
        self.qsort(nums, left, i - 1)
        self.qsort(nums, i + 1, right)
```



