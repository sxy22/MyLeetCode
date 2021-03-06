# 并查集

```python
class UnionFind():
    def __init__(self, n):
        self.parent = list(range(n))
        # 记录分支数
        self.count = n
	# 寻找父节点
    def find(self, node):
        if node != self.parent[node]:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        self.parent[self.find(node1)] = self.find(node2)
        self.count -= 1
        return

    def get_branch_count(self):
        return self.count
```





# 记录

## [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

+ hashmap 保存 数组的值 - 下标

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int a = nums[i];
            int b = target - a;
            if (map.containsKey(b)) {
                return new int[]{i, map.get(b)};
            }
            map.put(a, i);
        }
        return null;
    }
}
```



## [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

**思路**

- 先排序
- 固定一个i
- j, k 相对方向移动
- 避免重复， i j k 更新是应该更新到下一个不同的数字上



```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        i = 0
        # 对数组排序
        nums = sorted(nums)

        for i in range(n - 2):
            # 更新 i 时，要找下一个不同的数
            if i != 0 and nums[i] == nums[i-1]:
                continue
            if nums[i] > 0:
                break 
            
            j = i + 1
            k = n - 1
            while j < k:
                if nums[i] + nums[j] + nums[k] == 0:
                    # 加入res
                    res.append([nums[i], nums[j], nums[k]])
                    # 只需要将 j 更新到下一个不同的数，k会被外层循环自动更新
                    temp = nums[j]
                    while j < k and nums[j] == temp:
                        j += 1
                # 直接 +- 1 更新即可，同一个j不可能满足条件，不会导致重复答案
                elif nums[i] + nums[j] + nums[k] < 0:
                    j += 1
                else:
                    k -= 1
        return res 
```



## [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        ops = []
        res, num, sign = 0, 0, 1

        for ch in s:
            if ch.isdigit():
                num = 10 * num + int(ch)
            elif ch == '+' or ch =='-':
                res += sign * num 
                num = 0
                if ch == '+':
                    sign = 1
                else:
                    sign = -1
            elif ch == '(':
                stack.append(res)
                ops.append(sign)
                res = 0
                sign = 1
            elif ch == ')':
                res += sign * num 
                r = stack.pop()
                s = ops.pop()
                res = r + s * res
                num = 0
        res += sign * num 
        return res
```



## [456. 132模式](https://leetcode-cn.com/problems/132-pattern/)

**思路1 暴力 O(N^2)**

+ 维护 132模式 中间的那个数字 3
+ 维护3左边的最小数字min_left作为1
+ 向右遍历，找2

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        L = len(nums)
        if L < 3:
            return  False
        
        min_left = nums[0]
        for i in range(1, L - 1):
            mid = nums[i]
            if min_left < mid:
                # 满足 1 < 3
                for j in range(i + 1, L):
                    if nums[j] < mid and nums[j] > min_left:
                        return True 
            min_left = min(min_left, mid)
        return False
```



**思路2 单调栈 O(N)**

+ 从右往左遍历mid作为3，实现维护min_left，找到mid左边的最小数作为1
+ 找到mid右边小于mid的最大数，维护单调栈，小于mid的pop，最后一个pop的即为right
+ 检查是否满足条件
+ 注意
  + 从单调栈pop出的元素不会 影响答案，因为随着mid左移，left是递增的，
  + pop出去的right，要么满足条件return了，要么是大于等于left的
  + 所以之后left递增，pop出去的right不可能作为满足条件的right再出现

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        L = len(nums)
        if L < 3:
            return  False
        # 维护左边的最小数字
        min_left = [nums[0]] * L
        for i in range(1, L):
            min_left[i] = min(min_left[i - 1], nums[i - 1])
        # 单调栈，底大
        big_stack = []
        for i in reversed(range(L)):
            left = min_left[i]
            mid = nums[i]
            # right 设置为+1， 若没有pop，一定不会满足条件
            right = mid + 1
            while big_stack and big_stack[-1] < mid:
                right = big_stack.pop()
            big_stack.append(mid)
            if right < mid and left < right:
                return True
        return False
```



## [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

```python
class Trie:

    def __init__(self):
        # 保存子节点，小写字母情况下最多26个，可以用数组模拟
        self.children = dict()
        # 记录该节点 是否作为一个字符串的结束
        self.isEnd = False
    
    def insert(self, word: str) -> None:
        """Inserts a word into the trie.
        """
        node = self
        for ch in word:
            # 存在则不操作，不存在则设置一个新节点
            node.children.setdefault(ch, Trie())
            # 继续向下，插入新的字符
            node = node.children[ch]
        node.isEnd = True

    def search(self, word: str) -> bool:
        """Returns if the word is in the trie.
        """
        node = self
        for ch in word:
            next_node = node.children.get(ch, None)
            if next_node is None:
                return False
            node = next_node
        #  检查isEnd，为True说明有word以它结尾
        return node.isEnd

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self
        for ch in prefix:
            next_node = node.children.get(ch, None)
            if next_node is None:
                return False
            node = next_node
        return True
```



## [87. 扰乱字符串](https://leetcode-cn.com/problems/scramble-string/) 记忆化递归 困难

```python
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        self.hashmap = dict()
        l = len(s1)
        res = self.recur(s1, s2, 0, l - 1, 0, l - 1)
        print(self.hashmap)
        return res  

    def recur(self, s1, s2, l1, r1, l2, r2):
        L = r1 - l1 + 1
        key =  (l1, r1, l2, r2)
        # 检查是否已计算过
        value = self.hashmap.get(key, None)
        if value is not None:
            return value
        
        res = False
        # 长度为1, 直接比较相应位置的字符
        if L == 1:
            res = (s1[l1] == s2[l2])
            self.hashmap[key] = res  
            return res
        # 检查s1 s2 包含的字符是否相同
        if collections.Counter(s1[l1: r1+1]) != collections.Counter(s2[l2: r2+1]):
            res = False
            self.hashmap[key] = res  
            return res
        # l 表示s1左段长度,左右不能是空字符串, l的范围 1 -- len(s1) - 1  
        for l in range(1, L):
            if res:
                break 
            # 不交换
            res = self.recur(s1, s2, l1, l1 + l - 1, l2, l2 + l -1) and self.recur(s1, s2, l1 + l, r1, l2 + l, r2)
            if res:
                break
            # 交换
            res = self.recur(s1, s2, l1, l1 + l - 1, r2 - l + 1, r2) and self.recur(s1, s2, l1 + l, r1, l2, r2 - l)
        
        self.hashmap[key] = res  
        return res

```



## [212. 单词搜索 II](https://leetcode-cn.com/problems/word-search-ii/)

```python
class Trie():
    def __init__(self):
        self.word = ''
        self.children = dict()
    
    def insert(self, word):
        cur = self
        for w in word:
            cur.children.setdefault(w, Trie())
            cur = cur.children[w]
        cur.word = word

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        self.ans = set()
        trie = Trie()
        for w in words:
            trie.insert(w)
        maxi = len(board) - 1
        maxj = len(board[0]) - 1
        for i in range(maxi + 1):
            for j in range(maxj + 1):
                self.dfs(trie, i, j, board, maxi, maxj)
        res = [x for x in self.ans]
        return res
    
    def dfs(self, node, i, j, board, maxi, maxj):
        if i < 0 or j < 0 or i > maxi or j > maxj:
            return
        ch = board[i][j]
        if ch not in node.children:
            return
        newnode = node.children[ch]
        if newnode.word != '':
            # 匹配到了一个words中的单词
            self.ans.add(newnode.word)
        board[i][j] = '#'
        for p, q in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            self.dfs(newnode, p, q, board, maxi, maxj)
        board[i][j] = ch
```



## [373. 查找和最小的K对数字 ](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)多路归并

![image-20220113211001257](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220113211001257.png)

```java
class Solution {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> ans = new ArrayList<>();
        int n = nums1.length, m = nums2.length;
        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return nums1[o1[0]] + nums2[o1[1]] - nums1[o2[0]] - nums2[o2[1]];
            }
        });
        for (int i = 0; i < n; i++) {
            pq.add(new int[]{i, 0});
        }
        while (k-- > 0 && !pq.isEmpty()) {
            int[] first = pq.poll();
            int i = first[0], j = first[1];
            List<Integer> pair = Arrays.asList(nums1[i], nums2[j]);
            ans.add(pair);
            if (j + 1 < m) {
                pq.add(new int[]{i, j + 1});
            }
        }
        return ans;
    }
}
```



## [786. 第 K 个最小的素数分数 ](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)多路归并

```python
class Frac:
    def __init__(self, idx: int, idy: int, x: int, y: int) -> None:
        self.idx = idx
        self.idy = idy
        self.x = x
        self.y = y

    def __lt__(self, other: "Frac") -> bool:
        return self.x * other.y < self.y * other.x


class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        n = len(arr)
        q = [Frac(0, i, arr[0], arr[i]) for i in range(1, n)]
        heapq.heapify(q)

        for _ in range(k - 1):
            frac = heapq.heappop(q)
            i, j = frac.idx, frac.idy
            if i + 1 < j:
                heapq.heappush(q, Frac(i + 1, j, arr[i + 1], arr[j]))
        frac = heapq.heappop(q)
        return [frac.x, frac.y]
```



## [面试题 08.06. 汉诺塔问题](https://leetcode-cn.com/problems/hanota-lcci/)

```java
class Solution {
    public void hanota(List<Integer> A, List<Integer> B, List<Integer> C) {
        int n = A.size();
        move(A, B, C, n);
    }  
    
    private void move(List<Integer> A, List<Integer> B, List<Integer> C, int n) {
        if (n == 1) {
            C.add(A.remove(A.size() - 1));
        }else {
            move(A, C, B, n - 1);
            C.add(A.remove(A.size() - 1));
            move(B, A, C, n - 1);
        }
    }
}
```



## [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

+ 贪心 + 二分

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



## [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

+ 约瑟夫环

![image-20210422155709170](https://gitee.com/sxy22/note_images/raw/master/image-20210422155709170.png)

```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        prev = 0
        cur = 0
        for k in range(2, n + 1):
            t = m % k
            cur = (prev + t) % k
            prev = cur 
        
        return cur
```



```java
class Solution {
    public int lastRemaining(int n, int m) {
        int pre = 0, cur = 0;
        for (int k = 2; k <= n; k++) {
            int t = (m - 1) % k;
            cur = pre + t + 1;
            if (cur > k - 1) {
                cur -= k;
            }
            pre = cur;
        }
        return cur;
    }
}
```



## [440. 字典序的第K小数字](https://leetcode-cn.com/problems/k-th-smallest-in-lexicographical-order/)

+ long

```java
class Solution {
    public int findKthNumber(int n, int k) {
        int cur = 1;
        k -= 1;
        while (k > 0) {
            int cnt = cnt(cur, n);
            if (cnt <= k) {
                k -= cnt;
                cur += 1;
            }else {
                k -= 1;
                cur = cur * 10;
            }
        }
        return cur;
    }

    private int cnt(int cur, long n) {
        int cnt = 0;
        long first = cur;
        long last = cur;
        while (first <= n) {
            cnt += Math.min(last, n) - first + 1;
            first = first * 10;
            last = last * 10 + 9;
        }
        return cnt;
    }
}
```



