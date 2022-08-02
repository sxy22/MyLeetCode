



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




## [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node_a, node_b = headA, headB
        while node_a != node_b:
            if node_a is None:
                node_a = headB
            else:
                node_a = node_a.next
            if node_b is None:
                node_b = headA 
            else:
                node_b = node_b.next
        return node_b
```

## [46. 全排列](https://leetcode.cn/problems/permutations/)

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        self.visited = [False] * n
        self.path = []
        self.ans = [] 
        self.dfs(nums)
        return self.ans 
    
    def dfs(self, nums):
        if len(self.path) == len(nums):
            self.ans.append(self.path.copy())
            return 
        
        for i in range(len(nums)):
            if not self.visited[i]:
                self.visited[i] = True 
                self.path.append(nums[i])
                self.dfs(nums)
                self.path.pop()
                self.visited[i] = False 

```



## [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head is None:
            return None
        slow = head 
        fast = head 
        while fast.next is not None and fast.next.next is not None:
            slow = slow.next
            fast = fast.next.next 
            if slow == fast:
                break 
        if fast.next is None or fast.next.next is None:
            return None 
        
        while head != slow:
            head = head.next
            slow = slow.next
        return head



## [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1 
        left = 0
        right = len(nums) - 1
        #寻找相对有序的分段
        #判断mid是在前半段升序，还是后半段降序
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:#情况一：nums[mid]就是target
                return mid
            if nums[left] <= nums[mid]:#情况二：[left, mid]这个区间是升序的
               if nums[left] <= target < nums[mid]:#如果target在这一段中，所以这段（升序）可以用二分查找
                   right = mid - 1
               else:#如果target不在这一段升序中，说明left要右移到mid+1（等于mid的话有单独的if判断，所以不需要包括）
                   left = mid + 1
            else:#情况三：[left, mid]先升后降，[mid, right]是降序
                if nums[mid] < target <= nums[right]:#如果target在[mid, right]这个降序区间内，则可以用二分查找
                    left = mid + 1
                else:#如果不在降序区间内，right移动到mid左边
                    right = mid - 1
        return -1 
	
```



## [92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head 
        pre = dummy
        while left - 1 > 0:
            pre = pre.next
            left -= 1
            right -= 1
        cur = pre.next 
        pre_copy = pre 
        while right > 0:
            right -= 1
            nxt = cur.next 
            cur.next = pre 
            pre = cur 
            cur = nxt 
        pre_copy.next.next = cur 
        pre_copy.next = pre 
        return dummy.next
```



## [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

```
解法二：动态规划 + 二分查找
解题思路：

    降低复杂度切入点： 解法一中，遍历计算 dpdpdp 列表需 O(N)O(N)O(N)，计算每个 dp[k]dp[k]dp[k] 需 O(N)O(N)O(N)。
        动态规划中，通过线性遍历来计算 dpdpdp 的复杂度无法降低；
        每轮计算中，需要通过线性遍历 [0,k)[0,k)[0,k) 区间元素来得到 dp[k]dp[k]dp[k] 。我们考虑：是否可以通过重新设计状态定义，使整个 dpdpdp 为一个排序列表；这样在计算每个 dp[k]dp[k]dp[k] 时，就可以通过二分法遍历 [0,k)[0,k)[0,k) 区间元素，将此部分复杂度由 O(N)O(N)O(N) 降至 O(logN)O(logN)O(logN)。

    设计思路：
        新的状态定义：
            我们考虑维护一个列表 tails，其中每个元素 tails[k] 的值代表 长度为 k+1 的子序列尾部元素的值。
            如 [1,4,6] 序列，长度为 1,2,3 的子序列尾部元素值分别为 tails=[1,4,6]。


```



```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        min_tail = [0] * n 
        min_tail[0] = nums[0]
        size = 1
        for num in nums:
            if num > min_tail[size - 1]:
                min_tail[size] = num 
                size += 1
            else:
                idx = self.bi_right(min_tail, num, size)
                min_tail[idx] = num 
        # print(min_tail)
        return size 


    def bi_right(self, nums, value, size):
        i = 0
        j = size - 1
        while i < j:
            mid = (i + j) // 2
            if nums[mid] < value:
                i = mid + 1
            else:
                j = mid
        return i 
        
```





```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] min_tail = new int[n];
        min_tail[0] = nums[0];
        int len = 1;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            if (num > min_tail[len - 1]) {
                min_tail[len] = num;
                len += 1;
            }else if (num < min_tail[len - 1]) {
                int idx = bisect(min_tail, num, len);
                min_tail[idx] = num;
            }
        }
        return len;

    }

    private int bisect(int[] nums, int x, int j) {
        int i = 0;
        while (i < j) {
            int mid = (i + j) / 2;
            if (nums[mid] < x) {
                i = mid + 1;
            }else {
                j = mid;
            }
        }
        return i;
    }
}
```



## [143. 重排链表](https://leetcode.cn/problems/reorder-list/)

+ 找链表中间点
+ 后半部分reverse
+ 交叉连接

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        slow = head 
        fast = head 
        while fast.next is not None and fast.next.next is not None:
            slow = slow.next
            fast = fast.next.next

        l1 = head
        l2 = self.reverse(slow.next)
        slow.next = None  # 要断开，否则链表有环
        dummy = ListNode(-1)
        pre = dummy
        while l2 is not None:
            l1_next = l1.next
            # l2_next = l2.next
            l1.next = l2 
            pre.next = l1 
            pre = l2 
            l1 = l1_next
            l2 = l2.next

        pre.next = l1

    def reverse(self, head):
        pre = None 
        cur = head 
        while cur != None:
            nxt = cur.next
            cur.next = pre 
            pre = cur 
            cur = nxt 
        return pre
```



## [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = root.val
        self.dfs(root)
        return self.max_sum
        
    def dfs(self, node):
        if node is None:
            return 0 
        left_max = max(self.dfs(node.left), 0)
        right_max = max(self.dfs(node.right), 0)
        self.max_sum = max(self.max_sum, node.val + left_max + right_max)
        return node.val + max(left_max, right_max)
```



## [148. 排序链表](https://leetcode.cn/problems/sort-list/)

+ 中间拆分
+ merge sort

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head 
        return self.merge_sort_list(head)

    def merge_sort_list(self, head):
        if head.next is None:
            return head 
        l1, l2 = self.split(head)
        sort_l1 = self.merge_sort_list(l1)
        sort_l2 = self.merge_sort_list(l2)
        return self.merge(sort_l1, sort_l2)

    def merge(self, l1, l2):
        dummy = ListNode(-1)
        pre = dummy
        while l1 is not None and l2 is not None:
            if l1.val <= l2.val:
                pre.next = l1 
                l1 = l1.next
            else:
                pre.next = l2 
                l2 = l2.next
            pre = pre.next
        if l1 is not None:
            pre.next = l1 
        elif l2 is not None:
            pre.next = l2
        return dummy.next

    def split(self, head):
        slow = head 
        fast = head 
        while fast.next is not None and fast.next.next is not None:
            slow = slow.next
            fast = fast.next.next
        tmp = slow.next
        slow.next = None
        return head, tmp 
```

