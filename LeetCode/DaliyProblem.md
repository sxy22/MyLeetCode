# Leetcode每日一题

# 1/2022

## 1/21/2022

[1345. 跳跃游戏 IV](https://leetcode-cn.com/problems/jump-game-iv/)

```java
class Solution {
    public int minJumps(int[] arr) {
        //if (arr.length == 1) return 0;
        Set<Integer> visited = new HashSet<>();
        Map<Integer, List<Integer>> val2idx = new HashMap<>();
        Deque<Integer> deque = new LinkedList<>();
        deque.add(0);
        visited.add(0);
        int n = arr.length;
        int step = 0;
        for (int i = 0; i < n; i++) {
            int val = arr[i];
            if (!val2idx.containsKey(val)) {
                val2idx.put(val, new ArrayList<>());
            }
            val2idx.get(val).add(i);
        }
        //System.out.println(val2idx.toString());
        while (!deque.isEmpty()) {
            int l = deque.size();
            for (int i = 0; i < l; i++) {
                int idx = deque.removeFirst();
                if (idx == n - 1) return step;
                if (idx - 1 >= 0 && !visited.contains(idx - 1)) {
                    deque.add(idx - 1);
                    visited.add(idx - 1);
                }
                if (idx + 1 < n && !visited.contains(idx + 1)) {
                    deque.add(idx + 1);
                    visited.add(idx + 1);
                }
                if (val2idx.containsKey(arr[idx])) {
                    for (int next : val2idx.get(arr[idx])) {
                        if (!visited.contains(next)) {
                            deque.add(next);
                            visited.add(next);
                        }
                    }         
                    val2idx.remove(arr[idx]);           
                }
            }
            step += 1;
        }
        return -1;
    }
}
```



## 1/19/2022

[219. 存在重复元素 II](https://leetcode-cn.com/problems/contains-duplicate-ii/)

```java
class Solution {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (set.contains(num)) return true;
            set.add(num);
            if (i - k >= 0) {
                set.remove(nums[i - k]);
            }
        }
        return false;
    }
}
```



## 1/18/2022

[539. 最小时间差](https://leetcode-cn.com/problems/minimum-time-difference/)

```java
class Solution {
    public int findMinDifference(List<String> timePoints) {
        if (timePoints.size() >= 1440) return 0;
        int min_diff = 1440;
        List<Integer> minutes = new ArrayList<>();
        for (String t : timePoints) {
            minutes.add(minute(t));
        }

        Collections.sort(minutes);
        minutes.add(1440 + minutes.get(0));

        for (int i = 1; i < minutes.size(); i++) {
            min_diff = Math.min(min_diff, minutes.get(i) - minutes.get(i - 1));
        }
        return min_diff;


    }

    int minute(String t) {
        int h = Integer.parseInt(t.substring(0, 2));
        int m = Integer.parseInt(t.substring(3));
        return h * 60 + m;
    }
}
```



## 1/16/2022

[1220. 统计元音字母序列的数目](https://leetcode-cn.com/problems/count-vowels-permutation/)

```python
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        M = 1000000007
        a, e, i, o, u = 1, 1, 1, 1, 1
        for _ in range(n - 1):
            ta = e + i + u
            te = a + i
            ti = e + o
            to = i
            tu = i + o
            a = ta % M
            e = te % M
            i = ti % M
            o = to % M
            u = tu % M
        return (a + e + i + o + u) % M
```



## 1/15/2022

[1716. 计算力扣银行的钱](https://leetcode-cn.com/problems/calculate-money-in-leetcode-bank/)

```python
class Solution:
    def totalMoney(self, n: int) -> int:
        week = n // 7
        day = n % 7
        week_sum = (28 + (28 + 7 * (week - 1))) * week // 2
        day_sum = day * (day + 1) // 2 + week * day
        return week_sum + day_sum
```



## 1/14/2022

[373. 查找和最小的K对数字](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)

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



## 1/13/2022

[47. 至少是其他数字两倍的最大数](https://leetcode-cn.com/problems/largest-number-at-least-twice-of-others/)

```java
class Solution {
    public int dominantIndex(int[] nums) {
        if (nums.length == 1) return 0;
        int fir_idx = 0;
        int sec_idx = 1;
        if (nums[0] < nums[1]) {
            fir_idx = 1;
            sec_idx = 0;
        }
        for (int i = 2; i < nums.length; i++) {
            if (nums[i] > nums[fir_idx]) {
                sec_idx = fir_idx;
                fir_idx = i;
            }else if (nums[i] > nums[sec_idx]) {
                sec_idx = i;
            }
        }
        if (nums[fir_idx] >= 2 * nums[sec_idx]) {
            return fir_idx;
        }
        return -1;
    }
}
```



## 1/12/2022

[334. 递增的三元子序列](https://leetcode-cn.com/problems/increasing-triplet-subsequence/)

+ 判断当前的数之前以后没有比它更小的数，称为第二大数，需要维护一个min_val
+ 维护当前最小的第二大数

```java
class Solution {
    public boolean increasingTriplet(int[] nums) {
        if (nums.length < 3) return false;
        int n = nums.length;
        int min_val = Integer.MAX_VALUE;
        int min_sec = Integer.MAX_VALUE;
        for (int val : nums) {
            // 判断val是否> min_sec
            if (val > min_sec) return true;
            // 更新
            if (val > min_val && val < min_sec) {
                min_sec = val;
            }
            min_val = Math.min(min_val, val);
        }
        return false;
    }
}
```



## 1/10/2022

[306. 累加数](https://leetcode-cn.com/problems/additive-number/)

```python
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        self.num = num
        if len(num) < 3:
            return False
        for i in range(len(self.num) // 2):
            fir = int(self.num[:i + 1])
            for j in range(i + 1, len(self.num) - 1):
                sec = int(self.num[i + 1: j + 1])
                if self.isValid(fir, sec, j + 1):
                    return True
                if self.num[i + 1] == "0":
                    break
            if self.num[0] == "0":
                break
        return False
        

    def isValid(self, fir: int, sec: int, start):
        if start == len(self.num):
            return True
        third = str(fir + sec)
        if len(third) > len(self.num) - start:
            return False
        for i in range(len(third)):
            if third[i] != self.num[start]:
                return False
            start += 1 
        return self.isValid(sec, fir + sec, start)
```



## 1/9/2022

[1629. 按键持续时间最长的键](https://leetcode-cn.com/problems/slowest-key/)

```java
class Solution {
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        char max_key = keysPressed.charAt(0);
        int max_time = releaseTimes[0];
        for (int i = 1; i < keysPressed.length(); i++) {
            char key = keysPressed.charAt(i);
            int time = releaseTimes[i] - releaseTimes[i - 1];
            if (time > max_time || (time == max_time && key > max_key)) {
                max_key = key;
                max_time = time;
            }
        }
        return max_key;
    }
}
```

```python
class Solution:
    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        max_key = keysPressed[0]
        max_time = releaseTimes[0]
        for i in range(1, len(releaseTimes)):
            key = keysPressed[i]
            time = releaseTimes[i] - releaseTimes[i - 1]
            if time > max_time or (time == max_time and key > max_key):
                max_key = key
                max_time = time
        return max_key
```



## 1/8/2022

[89. 格雷编码](https://leetcode-cn.com/problems/gray-code/)

+ 设 n 阶格雷码集合为 G(n)，其中数字为n-1位二进制表示
+ 将G(n)反转，每个数最前面加上二进制1，放在G(n)后面得到G(n+1)

```java
class Solution {
    public List<Integer> grayCode(int n) {
        List<Integer> code = new ArrayList<>();
        code.add(0);
        code.add(1);
        int add = 2;
        for (int l = 0; l < n - 1; l++) {
            int i = code.size() - 1;
            while (i >= 0) {
                int val = code.get(i);
                code.add(val + add);
                i -= 1;
            }
            add = add << 1;
        }
        return code;
    }
}
```

**Python**

```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        code = [0, 1]
        add = 2
        for _ in range(n - 1):
            for val in reversed(code):
                code.append(val + add)
            add = add << 1
        return code
```







## 1/7/2022

[1614. 括号的最大嵌套深度](https://leetcode-cn.com/problems/maximum-nesting-depth-of-the-parentheses/)

```java
class Solution {
    public int maxDepth(String s) {
        Deque<Integer> stack = new LinkedList<>();
        int max_dep = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                stack.add(0);
            }
            if (ch == ')') {
                int dep = stack.removeLast() + 1;
                if (stack.isEmpty()) {
                    max_dep = Math.max(max_dep, dep);
                }else {
                    stack.add(Math.max(stack.removeLast(), dep));
                }
            }
        }
        return max_dep;
    }
}
```

```java
class Solution {
    public int maxDepth(String s) {
        int max_dep = 0;
        int dep = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                dep += 1;
            }
            if (ch == ')') {
                dep -= 1;
            }
            max_dep = Math.max(max_dep, dep);
        }
        return max_dep;
    }
}
```



## 1/6/2022

[71. 简化路径](https://leetcode-cn.com/problems/simplify-path/)

```java
class Solution {
    public String simplifyPath(String path) {
        LinkedList<String> stack = new LinkedList<>();
        StringBuilder s = new StringBuilder();
        path = path + "/";
        for (int i = 0; i < path.length(); i++) {
            char ch = path.charAt(i);
            if (ch == '/') {
                if (s.length() == 0) continue;
                String dirc = s.toString();
                s = new StringBuilder();
                if (dirc.equals(".")) {
                    //pass
                }else if (dirc.equals("..")) {
                    if (!stack.isEmpty()) {
                        stack.removeLast();
                    }
                }else {
                    stack.add(dirc);
                }
            }else {
                s.append(ch);
            }
        }
        //System.out.println(stack.toString());
        return "/" + String.join("/", stack);
    }
}
```



```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        step = path.split('/')
        stack = []
        for s in step:
            if len(s) == 0:
                continue
            if s == '.':
                pass
            elif s == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(s)
        
        return '/' + '/'.join(stack)
```



## 1/5/2022

[1576. 替换所有的问号](https://leetcode-cn.com/problems/replace-all-s-to-avoid-consecutive-repeating-characters/)

```java
class Solution {
    public String modifyString(String s) {
        StringBuilder ans = new StringBuilder(s);
        char left, right;
        for (int i = 0; i < ans.length(); i++) {
            if (ans.charAt(i) != '?') {
                continue;
            }
            if (i == 0) {
                left = '?';
            }else {
                left = ans.charAt(i - 1);
            }
            if (i == ans.length() - 1) {
                right = '?';
            }else {
                right = ans.charAt(i + 1);
            }
            ans.setCharAt(i, getchar(left, right));
        }
        return ans.toString();
        
    }

    char getchar(char left, char right) {
        char ch = 'a';
        for (int i = 0; i < 26; i++) {
            if (ch != left && ch != right) {
                return ch;
            }
            ch = (char) (ch + 1);
        }
        return ' ';
    }
}
```



## 1/2/2022

[390. 消除游戏](https://leetcode-cn.com/problems/elimination-game/)

```java
class Solution {
    public int lastRemaining(int n) {
        if (n == 1) return 1;
        int pre = n / 2;
        return 2 * (pre - lastRemaining(pre) + 1);

    }
}
```

```python
class Solution:
    def lastRemaining(self, n: int) -> int:
        a1 = 1
        an = n
        step = 1
        while a1 != an:
            # print(a1, an)
            temp = a1
            k = (an - a1) // step + 1
            if k & 1 == 1:
                a1 = an - step 
            else:
                a1 = an 
            an = temp + step 
            step *= -2
        return a1
```





## 1/1/2022

[2022. 将一维数组转变成二维数组](https://leetcode-cn.com/problems/convert-1d-array-into-2d-array/)

+ Arrays.copyOfRange

```java
class Solution {
    public int[][] construct2DArray(int[] original, int m, int n) {
        int L = original.length;
        if (m * n != L) return new int[0][];
        int[][] ans = new int[m][n];
        int i = 0;
        int start = 0;
        while (i < m) {
            ans[i] = Arrays.copyOfRange(original, start, start + n);
            i += 1;
            start = start + n;
        }
        return ans;
    }
}
```



+ 直接计算小标i对应的二维下标

```java
class Solution {
    public int[][] construct2DArray(int[] original, int m, int n) {
        int L = original.length;
        if (m * n != L) return new int[0][];
        int[][] ans = new int[m][n];
        for (int idx = 0; idx < L; idx ++) {
            int i = idx / n;
            int j = idx % n;
            ans[i][j] = original[idx];
        }
        return ans;
    }
}
```



---

# 12/2021

## 12/31/2021

[507. 完美数](https://leetcode-cn.com/problems/perfect-number/)

```java
class Solution {
    public boolean checkPerfectNumber(int num) {
        if (num == 1) return false;
        int sum = 1;
        int up = (int)Math.sqrt(num + 1);
        for (int i = 2; i <= up; i++) {
            int j = num / i;
            if (i * j == num) {
                if (i < j) {
                    sum += i;
                    sum += j;
                }
                if (i == j) {
                    sum += i;
                }
            }
        }
        return sum == num;
    }
}
```



## 12/30/2021

[846. 一手顺子](https://leetcode-cn.com/problems/hand-of-straights/)

```python
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if groupSize == 1:
            return True
        if len(hand) % groupSize != 0:
            return False

        cnt = collections.defaultdict(int)
        for x in hand:
            cnt[x] += 1
        hand = [x for x in cnt]
        hand = sorted(hand)
        i = 0
        while i < len(hand):
            # 找到起始数字
            start = hand[i]
            if cnt[start] == 0:
                i += 1
                continue
            # 判断是否能组成顺子
            for j in range(start, start + groupSize):
                if cnt[j] == 0:
                    return False 
                cnt[j] -= 1
        return True 
```





## 12/29/2021

[1995. 统计特殊四元组](https://leetcode-cn.com/problems/count-special-quadruplets/)

```python
class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        hmap = collections.defaultdict(int)
        n = len(nums)
        ans = 0
        for c in range(n - 2, 1, -1):
            for d in range(c + 1, n):
                hmap[nums[d] - nums[c]] += 1
            b = c - 1
            for a in range(b):
                ans += hmap[nums[a] + nums[b]]
        return ans 
```



## 12/28/2021

[472. 连接词](https://leetcode-cn.com/problems/concatenated-words/)

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
    
    def DFS(self, word, start):
        if start == len(word):
            return True 
        node = self
        for i in range(start, len(word)):
            next_node = node.children.get(word[i], None)
            if next_node is None:
                return False 
            if next_node.isEnd:
                if self.DFS(word, i + 1):
                    return True 
            node = next_node 
        return False 

class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words.sort(key=len)
        myTrie = Trie()
        ans = []
        for w in words:
            if w == '':
                continue
            if myTrie.DFS(w, 0):
                ans.append(w)
            else:
                myTrie.insert(w)
        return ans 
```



## 12/27/2021

[825. 适龄的朋友](https://leetcode-cn.com/problems/friends-of-appropriate-ages/)

```python
class Solution:
    def numFriendRequests(self, ages: List[int]) -> int:
        ages = sorted(ages)
        ans = 0
        for x in ages:
            left = 0.5 * x + 7
            right = x
            cnt = self.bisec(ages, right) - self.bisec(ages, left)
            if x > 14:
                cnt -= 1
            ans += max(0, cnt)
        return ans 

    def bisec(self, arr, x):
        lo = 0
        hi = len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo 
```



```python
class Solution:
    def numFriendRequests(self, ages: List[int]) -> int:
        cnt = [0] * 121
        for age in ages:
            cnt[age] += 1
        presum = [0] * 121
        for i in range(1, 121):
            presum[i] = presum[i - 1] + cnt[i]
        ans = 0
        for age in ages:
            if age <= 14:
                continue 
            lo = int(0.5 * age + 7) + 1
            ans += presum[age] - presum[lo - 1] - 1
        return ans 
```



## 12/26/2021

[1078. Bigram 分词](https://leetcode-cn.com/problems/occurrences-after-bigram/)

```python
class Solution:
    def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        lst = text.split(' ')
        n = len(lst)
        fir_idx = []
        sec_idx = []
        ans = []
        for i, word in enumerate(lst):
            if word == first:
                fir_idx.append(i)
            if word == second:
                sec_idx.append(i)
        i = 0
        j = 0
        while i < len(fir_idx) and j < len(sec_idx):
            idx1 = fir_idx[i]
            idx2 = sec_idx[j]
            if idx1 + 1 == idx2:
                if idx2 + 1 < n:
                    ans.append(lst[idx2 + 1])
                i += 1
                j += 1
            elif idx1 < idx2:
                i += 1
            elif idx1 >= idx2:
                j += 1
        return ans
```





## 12/25/2021

[1609. 奇偶树](https://leetcode-cn.com/problems/even-odd-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        queue = collections.deque()
        queue.append(root)
        level = 0
        while queue:
            layer = []
            for _ in range(len(queue)):
                node = queue.popleft()
                layer.append(node.val)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            if not self.is_valid(layer, level):
                return False 
            level += 1
        return True 


    def is_valid(self, layer, level):
        if level % 2 == 0:
            sign = 1
            resi = 1
        else:
            sign = -1
            resi = 0
        pre = layer[0]
        if pre % 2 != resi:
            return False
        for i in range(1, len(layer)):
            cur = layer[i]
            if cur % 2 != resi:
                return False
            if (cur - pre) * sign <= 0:
                return False 
            pre = cur 
        return True 
```



## 12/24/2021

[1705. 吃苹果的最大数目](https://leetcode-cn.com/problems/maximum-number-of-eaten-apples/)

```python
class Solution:
    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        apple_cnt = collections.defaultdict(int)
        pq = []
        day = 1
        ans = 0
        for i in range(len(apples)):
            # print(apple_cnt)
            app = apples[i]
            ds = days[i]
            while pq and pq[0] <= day:
                heapq.heappop(pq)
            if app != 0:
                apple_cnt[day + ds] += app 
                heapq.heappush(pq, day + ds)
            if len(pq) > 0:
                apple_cnt[pq[0]] -= 1
                ans += 1
                if apple_cnt[pq[0]] == 0:
                    heapq.heappop(pq)
            day += 1
        # print(ans)
        while pq:
            while pq and pq[0] <= day:
                heapq.heappop(pq)
            if len(pq) > 0:
                apple_cnt[pq[0]] -= 1
                ans += 1
                if apple_cnt[pq[0]] == 0:
                    heapq.heappop(pq)
            day += 1
        return ans 
```





## 12/22/2021

[686. 重复叠加字符串匹配](https://leetcode-cn.com/problems/repeated-string-match/)

```python
class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        k = len(b) // len(a)
        if len(b) % len(a) != 0:
            k += 1
        if self.is_sub(a * k, b):
            return k 
        if self.is_sub(a * (k + 1), b):
            return k + 1
        return -1 

    def is_sub(self, ss, s):
        i = 0
        j = len(s) - 1
        while j < len(ss):
            if hash(ss[i: j + 1]) == hash(s):
                return True
            i += 1
            j += 1
        return False
    
```



## 12/20/2021

[475. 供暖器](https://leetcode-cn.com/problems/heaters/)

+ 二分查找距离最近的值

**Java**

```java
class Solution {
    public int findRadius(int[] houses, int[] heaters) {
        int ans = 0;
        Arrays.sort(heaters);
        int n = heaters.length;
        for (int val : houses) {
            int min_dist;
            int idx = bisect(heaters, val);
            if (idx == 0) {
                min_dist = heaters[0] - val;
            }else if (idx == n) {
                min_dist = val - heaters[n - 1];
            }else {
                int l = val - heaters[idx - 1];
                int r = heaters[idx] - val;
                min_dist = Math.min(l, r);
            }
            ans = Math.max(ans, min_dist);
        }
        return ans;

    }

    int bisect(int[] arr, int value) {
        int le = 0, ri = arr.length;
        while (le < ri) {
            int mid = (le + ri) / 2;
            if (arr[mid] <= value) {
                le = mid + 1;
            }else {
                ri = mid;
            }
        }
        return le;
    }
}
```



**Python**

```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        heaters = sorted(heaters)
        ans = 0
        for val in houses:
            idx = bisect.bisect(heaters, val)
            if idx == 0:
                min_dist = heaters[0] - val
            elif idx == len(heaters):
                min_dist = val - heaters[-1]
            else:
                min_dist = min(val - heaters[idx - 1], heaters[idx] - val)
            ans = max(ans, min_dist)
        return ans 
```





## 12/19/2021

[997. 找到小镇的法官](https://leetcode-cn.com/problems/find-the-town-judge/)

+ JAVA

```java
class Solution {
    public int findJudge(int n, int[][] trust) {
        int[] trust_others = new int[n + 1];
        int[] trusted = new int[n + 1];
        for (int i = 0; i < trust.length; i++) {
            int a = trust[i][0];
            int b = trust[i][1];
            trust_others[a] = 1;
            trusted[b] += 1;
        }
        for (int i = 1; i < n + 1; i++) {
            if (trusted[i] == n - 1 && trust_others[i] == 0) {
                return i;
            }
        }
        return -1;
    }
}
```

+ Python

```python
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        trust_other = [0] * (n + 1)
        trusted = [0] * (n + 1)
        for a, b in trust:
            trust_other[a] = 1
            trusted[b] += 1
        for i in range(1, n + 1):
            if trusted[i] == n - 1 and trust_other[i] == 0:
                return i 
        return -1 
```





## 12/18/2021

[419. 甲板上的战舰](https://leetcode-cn.com/problems/battleships-in-a-board/)

+ python

```python
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        ans = 0
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if board[i][j] != 'X':
                    continue 
                if self.get(i - 1, j, board) != 'X' and self.get(i, j - 1, board) != 'X':
                    ans += 1
        return ans 

    def get(self, i, j, board):
        if i < 0 or j < 0:
            return '.'
        else:
            return board[i][j]
```

+ JAVA

```java
class Solution {
    char[][] board;

    public int countBattleships(char[][] board) {
        this.board = board;
        int m = board.length;
        int n = board[0].length;
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] != 'X') {
                    continue;
                }
                if (get(i - 1, j) != 'X' && get(i, j - 1) != 'X') {
                    ans += 1;
                }
            }
        }
        return ans;
    }

    char get(int i, int j) {
        if (i < 0 || j < 0) {
            return '.';
        }else {
            return board[i][j];
        }
    }
}
```





## 12/17/2021

[1518. 换酒问题](https://leetcode-cn.com/problems/water-bottles/)

+ Python

```python
class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        ans = 0
        empty = 0
        while numBottles != 0:
            ans += numBottles
            empty += numBottles
            numBottles = empty // numExchange
            empty = empty % numExchange
        return ans 
    
```

+ C++

```c++
class Solution {
public:
    int numWaterBottles(int numBottles, int numExchange) {
        int ans = numBottles;
        int empty = numBottles;
        while (empty >= numExchange) {
            int temp = empty / numExchange;
            ans += temp;
            empty = empty % numExchange + temp;
        }
        return ans;
    }
};
```







## 12/15/2021

[851. 喧闹和富有](https://leetcode-cn.com/problems/loud-and-rich/)

```python
class Solution:
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
        n = len(quiet)
        hmap = dict()
        for x, y in richer:
            hmap.setdefault(y, []).append(x)
        self.ans = [-1] * n
        for i in range(n):
            self.DFS(i, hmap, quiet)
        return self.ans 

    def DFS(self, i, hmap, quiet):
        if self.ans[i] != -1:
            return
        candi = i
        for idx in hmap.get(i, []):
            self.DFS(idx, hmap, quiet)
            if quiet[self.ans[idx]] < quiet[candi]:
                candi = self.ans[idx]
        self.ans[i] = candi 
        return 
```



## 12/14/2021

[630. 课程表 III](https://leetcode-cn.com/problems/course-schedule-iii/)

```python
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses = sorted(courses, key=lambda x: x[1])
        heap = []
        days = 0
        ans = 0
        for c in courses:
            d, last = c
            if days + d <= last:
                days += d
                ans += 1
                heapq.heappush(heap, -d)
            else:
                if len(heap) == 0:
                    continue
                dmax = -heap[0]
                if dmax > d:
                    heapq.heappop(heap)
                    heapq.heappush(heap, -d)
                    days += (d - dmax)
        return ans 
```





## 12/13/2021

[807. 保持城市天际线](https://leetcode-cn.com/problems/max-increase-to-keep-city-skyline/)

```python
class Solution:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        n = len(grid)
        ans = 0
        row_max = [0] * n
        col_max = [0] * n
        for r in range(n):
            for c in range(n):
                val = grid[r][c]
                row_max[r] = max(row_max[r], val)
                col_max[c] = max(col_max[c], val)
        for r in range(n):
            for c in range(n):
                ans += min(row_max[r], col_max[c]) - grid[r][c]
        return ans 
```



## 12/12/2021

[709. 转换成小写字母](https://leetcode-cn.com/problems/to-lower-case/)

+ 大写字母 A - Z的 ASCII 码范围为[65,90]
+ 小写字母 a - z的 ASCII 码范围为[97,122]
+ 相差32

```python
class Solution:
    def toLowerCase(self, s: str) -> str:
        ans = []
        for ch in s:
            asc = ord(ch)
            if asc <= 90 and asc >= 65:
                ch = chr(asc + 32)
            ans.append(ch)
        return ''.join(ans)
```





## 12/11/2021

[911. 在线选举](https://leetcode-cn.com/problems/online-election/)

```python
class TopVotedCandidate:

    def __init__(self, persons: List[int], times: List[int]):
        votes = {}
        top = None
        max_vote = 0
        tops = []
        for p in persons:
            votes.setdefault(p, 0)
            votes[p] += 1
            if votes[p] >= max_vote:
                max_vote = votes[p]
                top = p 
            tops.append(top) 
        self.times = times
        self.tops = tops 

    def q(self, t: int) -> int:
        lo, hi = 0, len(self.times)
        while lo < hi:
            mid = (lo + hi) // 2
            if t >= self.times[mid]:
                lo = mid + 1
            else:
                hi = mid 
        return self.tops[lo - 1]



# Your TopVotedCandidate object will be instantiated and called as such:
# obj = TopVotedCandidate(persons, times)
# param_1 = obj.q(t)
```





## 12/10/2021

[748. 最短补全词](https://leetcode-cn.com/problems/shortest-completing-word/)

```python
class Solution:
    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        cnt_lic = [0] * 26
        for s in licensePlate:
            s = s.lower()
            idx = ord(s) - ord('a')
            if idx >= 0 and idx <= 25:
                cnt_lic[idx] += 1
        # print(cnt_lic)
        ans = ''
        min_len = 100
        for word in words:
            if len(word) >= min_len:
                continue 
            if self.is_comp(word, cnt_lic):
                ans = word 
                min_len = len(ans)
        return ans 

    def is_comp(self, word, cnt_lic):
        cnt_word = [0] * 26
        for s in word:
            idx = ord(s) - ord('a')
            cnt_word[idx] += 1
        for i in range(26):
            if cnt_lic[i] != 0 and cnt_word[i] < cnt_lic[i]:
                return False 
        return True 
```





## 12/9/2021

[794. 有效的井字游戏](https://leetcode-cn.com/problems/valid-tic-tac-toe-state/)

```python
class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        cnt1, cnt2 = 0, 0
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    continue
                if board[r][c] == 'O':
                    cnt2 += 1
                else:
                    cnt1 += 1
        if cnt1 != cnt2 and cnt1 - 1 != cnt2:
            return False 
        win1, win2 = 0, 0
        for i in range(3):
            if board[i][0] == board[i][1] and board[i][1] == board[i][2]:
                if board[i][0] == 'X':
                    win1 += 1
                if board[i][0] == 'O':
                    win2 += 1
            if board[0][i] == board[1][i] and board[1][i] == board[2][i]:
                # print(board[0][i],  board[1][i], board[2][i])
                if board[0][i] == 'X':
                    win1 += 1
                if board[0][i] == 'O':
                    win2 += 1
        # print(win1, win2)
        if board[0][0] == board[1][1] == board[2][2]:
            if board[0][0] == 'X':
                win1 += 1
            if board[0][0] == 'O':
                win2 += 1
        if board[0][2] == board[1][1] == board[2][0]:
            if board[0][2] == 'X':
                win1 += 1
            if board[0][2] == 'O':
                win2 += 1 
        # print(win1, win2)
        if win1 != 0 and win2 != 0:
            return False
        if win1 != 0 and cnt1 == cnt2:
            return False
        if win2 != 0 and cnt1 != cnt2:
            return False 
        return True
```



## 12/6/2021

[1816. 截断句子](https://leetcode-cn.com/problems/truncate-sentence/)

```python
class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        s += ' '
        ans = []
        word = ''
        for c in s:
            if k == 0:
                break
            if c == ' ':
                ans.append(word)
                word = ''
                k -= 1
            else:
                word += c 

        return ' '.join(ans)
```

+ 数空格

```python
class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        # 数空格
        s += ' '
        for i in range(len(s)):
            c = s[i]
            if c == ' ':
                k -= 1
                if k == 0:
                    break
        return s[:i]
```



## 12/4/2021

[383. 赎金信](https://leetcode-cn.com/problems/ransom-note/)

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        mag_cnt = collections.defaultdict(int)
        for s in magazine:
            mag_cnt[s] += 1
        for s in ransomNote:
            mag_cnt[s] -= 1
            if mag_cnt[s] < 0:
                return False 
        return True 
```







## 12/2/2021

[506. 相对名次](https://leetcode-cn.com/problems/relative-ranks/)

```python
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        n = len(score)
        ans = [''] * n 
        idx_sorted = list(range(n))
        idx_sorted = sorted(idx_sorted, key=lambda i: score[i], reverse=True)
        for i in range(n):
            medal = str(i + 1)
            if i == 0:
                medal = 'Gold Medal'
            elif i == 1:
                medal = 'Silver Medal'
            elif i == 2:
                medal = 'Bronze Medal'
            ans[idx_sorted[i]] = medal
        return ans
```



## 12/1/2021

[1446. 连续字符](https://leetcode-cn.com/problems/consecutive-characters/)

```python
class Solution:
    def maxPower(self, s: str) -> int:
        s += '#'
        char = s[0]
        max_len = 1
        i, j = 0, 0
        while j < len(s) - 1:
            j = j + 1
            if s[j] != char:
                max_len = max(max_len, j - i)
                char = s[j]
                i = j 
        
        return max_len
```



# 11/2021

## 11/30/2021

[400. 第 N 位数字](https://leetcode-cn.com/problems/nth-digit/)

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        s = 0
        m = 1
        while m < 100:
            if s < n <= s + 9 * (10 ** (m-1)) * m:
                break
            else:
                
                s += 9 * (10 ** (m-1)) * m
                m += 1
        # 是m位数
        #再确定是m位数中的第几个
        T = n - s
        if T % m == 0:
            number = int(T/m)
        else:
            number = int(T/m) + 1
        #计算第number个m位数是几
        X = 10 ** (m - 1) + number - 1
        #是X的第几位
        X = str(X)
        dig = T % m
        return int(X[dig - 1])
```





## 11/29/2021

[786. 第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)

+ 官方方法二：优先队列

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





## 11/28/2021

[438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

+ 双指针表示范围
+ diff维护差异

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(s) < len(p):
            return []
        ans = []
        len_s, len_p = len(s), len(p)
        cnt = [0] * 26
        for i in range(len_p):
            cnt[ord(p[i]) - ord('a')] += 1
            cnt[ord(s[i]) - ord('a')] -= 1
        diff = sum([1 for x in cnt if x != 0])
        if diff == 0:
            ans.append(0)
        i = 0
        j = len_p - 1
        while j < len_s - 1:
            # 去掉s[i]
            i_idx = ord(s[i]) - ord('a')
            if cnt[i_idx] == -1:
                diff -= 1
            if cnt[i_idx] == 0:
                diff += 1
            cnt[i_idx] += 1
            # 增加s[j+1]
            j_idx = ord(s[j+1]) - ord('a')
            if cnt[j_idx] == 1:
                diff -= 1
            if cnt[j_idx] == 0:
                diff += 1
            cnt[j_idx] -= 1
            i += 1
            j += 1
            # print(i, j, diff)
            if diff == 0:
                ans.append(i)
        return  ans 
```





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

