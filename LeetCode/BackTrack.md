# 回溯(BackTrack)

## [46. 全排列](https://leetcode-cn.com/problems/permutations/)

**思路**

+ 经典回溯
+ 深度优先，可以共用track，不会冲突
+ visited记录是否访问过

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.res = []
        # 记录是否访问过该数字
        self.visited = set()
        # 共用的track
        self.track = []

        self.backtrack_dfs(nums)
        return self.res 

    def backtrack_dfs(self, nums):
        if len(self.track) == len(nums):
            self.res.append(self.track.copy())
            return
        for num in nums:
            if num not in self.visited:
                self.track.append(num)
                self.visited.add(num)
                self.backtrack_dfs(nums)
                self.track.pop()
                self.visited.remove(num)
        return 
```

## [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

**思路**

+ 有重复数字，为了避免重复的答案，每一次只能选择相同数字中的一个
+ 使用dict记录每个数字的个数
+ 寻找下一个数字时要遍历dict.keys()  即不重复数组

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        self.res = []
        # 记录是数字剩余可用个数，每次只能取一个
        self.cnt = collections.defaultdict(int)
        for num in nums:
            self.cnt[num] += 1
        # 共用的track
        self.track = []
        self.n = len(nums)

        self.backtrack_dfs()
        return self.res 

    def backtrack_dfs(self):
        if len(self.track) == self.n:
            self.res.append(self.track.copy())
            return
        # 遍历self.cnt.keys()
        for num in self.cnt.keys():
            if self.cnt[num] != 0:
                self.track.append(num)
                self.cnt[num] -= 1
                self.backtrack_dfs()
                self.track.pop()
                self.cnt[num] += 1
        return 
```



## [90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

**思路**

+ 回溯，找子集
+ 规定递增，去重

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        self.res = []
        self.cnt = collections.defaultdict(int)
        for num in nums:
            self.cnt[num] += 1
        
        self.track = []
        self.keys = self.cnt.keys()

        self.backtrack_dfs(-10e3)
        return self.res

    def backtrack_dfs(self, last):
        self.res.append(self.track.copy())
        for num in self.keys:
            if num >= last and self.cnt[num] != 0:
                self.track.append(num)
                self.cnt[num] -= 1
                self.backtrack_dfs(num)
                self.track.pop()
                self.cnt[num] += 1
        return
```





## [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        self.res = []
        self.track = []
        self.cnt = [n, n]
        self.backtrack_dfs(n)
        return self.res 

    def backtrack_dfs(self, n):
        if len(self.track) == 2 * n:
            self.res.append(''.join(self.track))
            return 
        # 左括号
        if self.cnt[0] > 0:
            s = '('
            self.track.append(s)
            self.cnt[0] -= 1
            self.backtrack_dfs(n)
            self.track.pop()
            self.cnt[0] += 1
        # 右括号剩的比左括号多, 才能加入一个右括号
        if self.cnt[1] > self.cnt[0]:
            s = ')'
            self.track.append(s)
            self.cnt[1] -= 1
            self.backtrack_dfs(n)
            self.track.pop()
            self.cnt[1] += 1
        return 
```

## [77. 组合](https://leetcode-cn.com/problems/combinations/)

**思路**

+ 回溯，track长度为k时，加入res
+ 可以剪枝，剩余的数字不足以填满k, 不需要再继续
+ 要求数字依次递增来解决重复

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        self.track = []
        self.ans = []
        self.backtrack(0, n, k)
        return self.ans 

    def backtrack(self, last, n, k):
        if len(self.track) == k:
            self.ans.append(self.track.copy())
            return 
        # 剪枝
        if k - len(self.track) > n - last:
            return 
        for num in range(last + 1, n + 1):
            self.track.append(num)
            self.backtrack(num, n, k)
            self.track.pop()
        return 
```
