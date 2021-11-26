# Leetcode每日一题

# 11/2021

## 11/26/2021

[700. 二叉搜索树中的搜索](https://leetcode-cn.com/problems/search-in-a-binary-search-tree/)

+ 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if root is None:
            return None 
        if root.val == val:
            return root 
        elif root.val > val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)
```

+ 迭代

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if root.val == val:
                return root 
            elif val < root.val:
                root = root.left
            else:
                root = root.right 
        return None 
```





## 11/23/2021

[859. 亲密字符串](https://leetcode-cn.com/problems/buddy-strings/)

**思路**

+ 长度要相同
+ 若两字符串完全相等，则要求有重复字符
+ 有两个位置不同，且错位相等

```python
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False
        idx = [] 
        cnt = collections.defaultdict(int)
        twosame = False
        for i in range(len(s)):
            if s[i] != goal[i]:
                if len(idx) == 2:
                    return False 
                idx.append(i)
            cnt[s[i]] += 1
            if twosame is False and cnt[s[i]] >= 2:
                twosame = True 
        if len(idx) == 1:
            return False
        if len(idx) == 0:
            return twosame
        i, j = idx 
        if s[i] != goal[j] or s[j] != goal[i]:
            return False 
        return True  
```







## 11/22/2021

[384. 打乱数组](https://leetcode-cn.com/problems/shuffle-an-array/)

**思路**

+ 从前往后填充 [0,n−1] 

+ 对于下标 x 而言，我们从[x,n−1] 中随机出一个位置与 xxx 进行值交换

```python
class Solution:
    def __init__(self, nums: List[int]):
        self.original = nums.copy()
        self.nums = nums
        self.n = len(nums)

    def reset(self) -> List[int]:
        self.nums = self.original.copy()
        return self.nums

    def shuffle(self) -> List[int]:
        for i in range(self.n - 1):
            j = random.randint(i, self.n - 1)
            self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
        return self.nums 
```





## 11/21/2021

[559. N 叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-n-ary-tree/)

**思路**1 

+ DFS

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if root is None:
            return 0 
        # if root.children is None:
        #     return 1 
        child_dep = 0
        for child in root.children:
            child_dep = max(child_dep, self.maxDepth(child))
        return 1 + child_dep
```

**思路2**1

+ BFS

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if root is None:
            return 0
        ans = 0
        queue = collections.deque()
        queue.append(root)
        while queue:
            ans += 1
            for _ in range(len(queue)):
                node = queue.popleft() 
                for child in node.children:
                    queue.append(child)
        
        return ans 
```





## 11/20/2021

[594. 最长和谐子序列](https://leetcode-cn.com/problems/longest-harmonious-subsequence/)

**思路**

+ 每个子序列中只有两个相邻的数
+ 用哈希表记录每个数字出现的次数

```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        ans = 0
        hashmap = dict()
        for num in nums:
            if num not in hashmap:
                hashmap[num] = 0
            hashmap[num] += 1
        
        for fir in hashmap.keys():
            sec_cnt = hashmap.get(fir + 1, 0)
            if sec_cnt != 0:
                ans = max(ans, hashmap[fir] + sec_cnt)
        return ans 
```



## 11/19/2021

[397. 整数替换](https://leetcode-cn.com/problems/integer-replacement/)

**思路1**

+ 直接递归
+ 偶数时， 计算 n // 2
+ 奇数时，计算min(n-1, n+1)

```python
class Solution:
    def integerReplacement(self, n: int) -> int:
        if n == 1:
            return 0 
        if n % 2 == 0:
            return 1 + self.integerReplacement(n // 2)
        else:
            return 2 + min(self.integerReplacement(n // 2), self.integerReplacement(n // 2 + 1)
        return -1 
```

**思路2**

+ 记忆化减低复杂度

```python
class Solution:
    def __init__(self):
        self.memo = dict()
    def integerReplacement(self, n: int) -> int:
        if n == 1:
            return 0 
        if n in self.memo:
            return self.memo[n]
        steps = -1
        if n % 2 == 0:
            steps = 1 + self.integerReplacement(n // 2)
        else:
            steps = 2 + min(self.integerReplacement(n // 2), self.integerReplacement(n // 2 + 1))
        self.memo[n] = steps
        return steps

```





## 11/18/2021

[563. 二叉树的坡度](https://leetcode-cn.com/problems/binary-tree-tilt/)

**思路**

+ DFS递归，返回以该节点为root的所有结点的和，left + right + node.val
+ 并同时得到该结点的坡度



```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findTilt(self, root: TreeNode) -> int:
        self.ans = 0
        _ = self.DFS_sum(root)
        return self.ans 
	
    # 计算root的结点和
    def DFS_sum(self, root):
        if root is None:
            return 0 
        left_sum = self.DFS_sum(root.left)
        right_sum = self.DFS_sum(root.right)
        self.ans += abs(left_sum - right_sum)
        return left_sum + right_sum + root.val 
```



## 11/17/2021

[318. 最大单词长度乘积](https://leetcode-cn.com/problems/maximum-product-of-word-lengths/)

**思路**

+ 先用26位二进制表示每个word
+ 与为0则没有相同字符串

```python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        n = len(words)
        bits = [0] * n
        for i in range(n):
            word = words[i]
            for w in word:
                # 或运算
                bits[i] = bits[i] | (1 << (ord(w) - ord('a')))
        ans = 0

        for i in range(n):
            for j in range(i+1, n):
                if (bits[i] & bits[j]) == 0:
                    ans = max(ans, len(words[i]) * len(words[j]))
        return ans 
```



## 11/15/2021

[319. 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/)

**思路**

+ 只需要计算1-n有几个平方数！
+ 非平方数一定是暗的
+ 1-n 中平方数的个数为sqrt(n)向下取整

```python
class Solution:
    def bulbSwitch(self, n: int) -> int:
        i = 1
        ans = 0 
        while i * i <= n:
            ans += 1
            i += 1
        return ans
```

```python
class Solution:
    def bulbSwitch(self, n: int) -> int:
        return (int(math.sqrt(n)))
```















## 11/14/2021

[677. 键值映射](https://leetcode-cn.com/problems/map-sum-pairs/)

**思路**

+ Trie树的思路，每个结点存当前的sum
+ 如果重复insert，需要挑战该路径上的所有val，addi = val - pre 

```python
class MapSum:
    def __init__(self):
        self.map = dict()
        self.trie = Trie()

    def insert(self, key: str, val: int) -> None:
        pre = self.map.get(key, 0)
        addi = val - pre 
        self.map[key] = val 
        node = self.trie
        for ch in key:
            idx = ord(ch) - ord('a')
            if node.children[idx] is None:
                node.children[idx] = Trie() 
            node.children[idx].val += addi 
            node = node.children[idx]

    def sum(self, prefix: str) -> int:
        node = self.trie
        for ch in prefix:
            idx = ord(ch) - ord('a')
            if node.children[idx] is None:
                return 0
            node = node.children[idx]
        return node.val 

class Trie:
    def __init__(self):
        # 保存子节点，小写字母情况下最多26个，可以用数组模拟
        self.children = [None] * 26
        # 记录该节点 是否作为一个字符串的结束
        self.val = 0

# Your MapSum object will be instantiated and called as such:
# obj = MapSum()
# obj.insert(key,val)
# param_2 = obj.sum(prefix)
```



## 11/13/2021

[520. 检测大写字母](https://leetcode-cn.com/problems/detect-capital/)

```python
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if len(word) == 1:
            return True 
        fir = self.is_up(word[0])
        sec = self.is_up(word[1])
        if fir is False and sec is True:
            return False 
        rest = False
        if fir is True and sec is True:
            rest = True
        for i in range(2, len(word)):
            if self.is_up(word[i]) is not rest:
                return False 
        return True 


    def is_up(self, s):
        if s >= 'A' and s <= 'Z':
            return True
        return False
```



## 11/12/2021

[375. 猜数字大小 II](https://leetcode-cn.com/problems/guess-number-higher-or-lower-ii/)

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for l in range(2, n + 1):
            for i in range(1, n + 2 - l):
                j = i + l - 1
                mini = n * n
                for k in range(i, j + 1):
                    mini = min(mini, k + max(self.get(dp, i, k-1), self.get(dp, k+1, j)))
                dp[i][j] = mini
        return dp[1][n]
    def get(self, arr, i, j):
        if i > j:
            return 0
        return arr[i][j]
```



## 11/10/2021

[495. 提莫攻击](https://leetcode-cn.com/problems/teemo-attacking/)

```python
class Solution:
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        ans = duration
        for i in range(len(timeSeries) - 1):
            t1 = timeSeries[i]
            t2 = timeSeries[i+1]
            ans += min(t2 - t1, duration)
        return ans
```







## 11/8/2021

[299. 猜数字游戏](https://leetcode-cn.com/problems/bulls-and-cows/)

**思路**

+ 相同的直接记录bu;ll
+ 不相同的用map记录，取min记录cow



```python
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        map1 = collections.defaultdict(int)
        map2 = collections.defaultdict(int)
        bull, cow = 0, 0
        for s1, s2 in zip(secret, guess):
            if s1 == s2:
                bull += 1
            else:
                map1[s1] += 1
                map2[s2] += 1
        for s in map2:
            cow += min(map1[s], map2[s])
        return f'{bull}A{cow}B'
```



## 11/7/2021

[598. 范围求和 II](https://leetcode-cn.com/problems/range-addition-ii/)

```python
class Solution:
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        min_a = m
        min_b = n
        for a, b in ops:
            min_a = min(min_a, a)
            min_b = min(min_b, b)
        return min_a * min_b
```





## 11/6/2021

[268. 丢失的数字](https://leetcode-cn.com/problems/missing-number/)

+ 异或
+ 先求[1,n] 的异或和 ans，然后用 anss 对各个 nums[i]nums[ii]进行异或


```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        xorsum = 0
        n = len(nums)
        for x in nums:
            xorsum ^= x 
        for x in range(n + 1):
            xorsum ^= x 
        return xorsum
```



## 11/5/2021

[1218. 最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/)

```python
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        num_len = dict() 
        ans = 1
        for num in arr:
            l = num_len.get(num - difference, 0) + 1
            ans = max(ans, l)
            num_len[num] = l 
        
        return ans
```



## 11/4/2021

[367. 有效的完全平方数](https://leetcode-cn.com/problems/valid-perfect-square/)

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num == 1:
            return True
        l = 2
        r = num // 2
        while l <= r:
            mid = (l + r) // 2
            sq = mid * mid
            if sq == num:
                return True
            elif sq > num:
                r = mid - 1
            else:
                l = mid + 1
        return False
```



## 11/2/2021

[237. 删除链表中的节点](https://leetcode-cn.com/problems/delete-node-in-a-linked-list/)

**思路**

+ 将需要删除的结点值设置为下一个结点的值
+ 删除下一个结点
+ 效果等同于删除node

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val 
        node.next = node.next.next
```



## 11/1/2021

[575. 分糖果](https://leetcode-cn.com/problems/distribute-candies/)

```python
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        hset = set()
        for c in candyType:
            hset.add(c)
        
        max_candy = len(candyType) // 2
        return min(max_candy, len(hset))
```



# 10/2021

## 10/31/2021

[500. 键盘行](https://leetcode-cn.com/problems/keyboard-row/)

```python
class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        hashmap = dict()
        groups = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        lines = [0, 1, 2]
        for w, l in zip(groups, lines):
            for s in w:
                hashmap[s] = l 
        
        ans = []
        for word in words:
            line = -1
            valid = True
            for s in word:
                s = s.lower()
                if line == -1:
                    line = hashmap[s]
                    continue 
                if hashmap[s] != line:
                    valid = False
                    break 
            if valid:
                ans.append(word)
        return ans
```

