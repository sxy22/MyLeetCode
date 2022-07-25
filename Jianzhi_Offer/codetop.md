



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





## [15. 三数之和](https://leetcode.cn/problems/3sum/)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []
        ans = []
        nums = sorted(nums)
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue 
            a = nums[i]
            if a > 0:
                break 
            j = i + 1
            k = len(nums) - 1
            while j < k:
                b = nums[j]
                c = nums[k]
                if a + b + c < 0:
                    j += 1
                elif a + b + c > 0:
                    k -= 1
                else:
                    ans.append([a, b, c])
                    while j < k and nums[j] == b:
                        j += 1
                    while j < k and nums[k] == c:
                        k -= 1
        return ans 

```



## [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        max_sum = nums[0]
        for i in range(1, n):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
            max_sum = max(max_sum, nums[i])
        return max_sum
```



## [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy_node = ListNode(-1)
        dummy_node.next = head
        pre = dummy_node
        tail = dummy_node
        while tail is not None:
            step = k
            while step > 0 and tail is not None:
                step -= 1
                tail = tail.next
            if tail is None:
                break 
            r_head, r_tail = self.reverse_k(pre.next, tail)
            pre.next = r_head 
            pre = r_tail 
            tail = r_tail
        return dummy_node.next


    
    def reverse_k(self, head, tail):
        tail_next_node = tail.next
        pre = None
        cur = head
        while pre != tail:
            nxt = cur.next
            cur.next = pre 
            pre = cur 
            cur = nxt 
        head.next = tail_next_node
        return tail, head 

```



## [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        cur = dummy
        while list1 is not None and list2 is not None:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2  
                list2 = list2.next
            cur = cur.next 
        if list1 is None:
            cur.next = list2
        if list2 is None:
            cur.next = list1 
        return dummy.next
   
```



## [1. 两数之和](https://leetcode.cn/problems/two-sum/)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        val2idx = dict()
        for i in range(len(nums)):
            val = nums[i]
            if target - val in val2idx:
                return [val2idx[target - val], i]
            else:
                val2idx[val] = i
        return []
            
```



## [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        ans = []
        deque = collections.deque()
        deque.append(root)
        while deque:
            layer = []
            for _ in range(len(deque)):
                node = deque.popleft()
                layer.append(node.val)
                if node.left is not None:
                    deque.append(node.left)
                if node.right is not None:
                    deque.append(node.right)
            ans.append(layer)
        return ans 
```



## [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None or root == p or root == q:
            return root 
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left is not None and right is not None:
            return root 
        if left is None:
            return right 
        else:
            return left 
```

