## Leetcode每日一题

## 11/14/2021

#### [677. 键值映射](https://leetcode-cn.com/problems/map-sum-pairs/)



实现一个 MapSum 类，支持两个方法，insert 和 sum：

+ MapSum() 初始化 MapSum 对象
+ void insert(String key, int val) 插入 key-val 键值对，字符串表示键 key ，整数表示值 val 。如果键 key 已经存在，那么原来的键值对将被替代成新的键值对。
+ int sum(string prefix) 返回所有以该前缀 prefix 开头的键 key 的值的总和。

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

#### [319. 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/)

初始时有 n 个灯泡处于关闭状态。第一轮，你将会打开所有灯泡。接下来的第二轮，你将会每两个灯泡关闭一个。

第三轮，你每三个灯泡就切换一个灯泡的开关（即，打开变关闭，关闭变打开）。第 i 轮，你每 i 个灯泡就切换一个灯泡的开关。直到第 n 轮，你只需要切换最后一个灯泡的开关。

找出并返回 n 轮后有多少个亮着的灯泡。

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

