## Leetcode每日一题



## 11/10/2021

#### [495. 提莫攻击](https://leetcode-cn.com/problems/teemo-attacking/)

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



