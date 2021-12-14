# 动态规划&DFS

## [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        l1 = len(s)
        l2 = len(t)
        if l1 < l2:
            return 0
            
        dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
        # 先初始化dp
        for i in range(l1 + 1):
            dp[i][l2] = 1
		# 根据dp性质，需要从后往前遍历i， j
        for j in range(l2 - 1, -1, -1):
            for i in range(l1 - 1, -1, -1):
                # 这个if是多余的
                if l1 - i < l2 - j:
                    dp[i][j] = 0
                    continue
                if s[i] == t[j]:
                    dp[i][j] = dp[i+1][j+1] + dp[i+1][j]
                else:
                    dp[i][j] = dp[i+1][j]
        
        return dp[0][0]
```



## [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

**思路**

+ 动态规划

![image-20210403122103741](https://gitee.com/sxy22/note_images/raw/master/image-20210403122103741.png)

![image-20210403122130602](https://gitee.com/sxy22/note_images/raw/master/image-20210403122130602.png)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        l1 = len(text1)
        l2 = len(text2)
        # dp[0][0]留给空字符串
        dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]

        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                s1 = text1[i - 1]
                s2 = text2[j - 1]
                if s1 == s2:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        
        return dp[-1][-1]
```



## [673. 最长递增子序列的个数](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)



```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n 
        cnt = [1] * n 
        for i in range(1, n):
            last = nums[i]
            for j in range(i):
                if nums[j] < last:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        cnt[i] = cnt[j]
                    elif dp[j] + 1 == dp[i]:
                        cnt[i] += cnt[j]
        ans = 0
        L = max(dp)
        for i in range(n):
            if dp[i] == L:
                ans += cnt[i]
        return ans 
```

