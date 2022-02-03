# 剑指 Offer（专项突击版）

## [剑指 Offer II 001. 整数除法](https://leetcode-cn.com/problems/xoh6Oh/)

```python
class Solution:
    def divide(self, a: int, b: int) -> int:
        max_int = 2147483647
        sign = 1
        if a * b < 0:
            sign = -1
        a = abs(a)
        b = abs(b)
        cum = 0
        ans = 0
        k = 1
        copy_b = b
        while cum < a:
            # print(b, k)
            if cum + b >= a + copy_b:
                b = copy_b
                k = 1
            cum += b 
            ans += k
            b = b + b
            k = k + k
        if cum > a:
            ans = sign * (ans - 1)
        else:
            ans = sign * ans

        return min(max_int, ans)
```



## [剑指 Offer II 002. 二进制加法](https://leetcode-cn.com/problems/JFETK5/)

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        if len(a) > len(b):
            a, b = b, a
        a = '0' * (len(b) - len(a)) + a
        d = {'0': 0, '1': 1}
        i = len(a) - 1
        add = 0
        ans = []
        while i >= 0:
            s = d[a[i]] + d[b[i]] + add
            ans.append(str(s % 2))
            add = s // 2
            i -= 1
        if add == 1:
            ans.append('1')
        
        return ''.join(reversed(ans))
```



## [剑指 Offer II 003. 前 n 个数字二进制中 1 的个数](https://leetcode-cn.com/problems/w3tCBm/)

+ n 的二进制1的个数 = n & (n - 1) 的个数 + 1

```java
class Solution {
    public int[] countBits(int n) {
        int[] dp = new int[n + 1];
        for (int num = 1; num <= n; num++) {
            int pre = num & (num - 1);
            dp[num] = dp[pre] + 1;
        }
        return dp;
    }
}
```



+ 检查最后一位 & 1，并右移

```java
class Solution {
    public int[] countBits(int n) {
        int[] dp = new int[n + 1];
        for (int num = 1; num <= n; num++) {
            int lowbit = num & 1;
            dp[num] = dp[num >> 1] + lowbit;
        }
        return dp;
    }
}
```



## [剑指 Offer II 004. 只出现一次的数字 ](https://leetcode-cn.com/problems/WGki4K/)

见 Offer 56 II

```java
class Solution {
    public int singleNumber(int[] nums) {
        int ones = 0, twos = 0;
        for (int num : nums) {
            ones = (ones ^ num) & (~twos);
            twos = (twos ^ num) & (~ones);
        }
        return ones; 
    }
}
```



## [剑指 Offer II 005. 单词长度的最大乘积](https://leetcode-cn.com/problems/aseY1I/)

+ 将判断两个单词是否有公共字母的时间复杂度降低到 O(1)
+ 只包含小写字母，用一个int的低26位表示

```java
class Solution {
    public int maxProduct(String[] words) {
        int n = words.length;
        int[] bits = new int[n];
        int max_pro = 0;
        for (int i = 0; i < n; i++) {
            String word = words[i];
            int bit = 0;
            for (int j = 0; j < word.length(); j++) {
                bit |= 1 << (word.charAt(j) - 'a');
            }
            bits[i] = bit;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((bits[i] & bits[j]) == 0) {
                    max_pro = Math.max(max_pro, words[i].length() * words[j].length());
                }
            }
        }
        return max_pro;
    }
}
```



## [剑指 Offer II 006. 排序数组中两个数字之和](https://leetcode-cn.com/problems/kLl5u1/)

```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int i = 0, j = numbers.length - 1;
        while (i < j) {
            int sum = numbers[i] + numbers[j];
            if (sum == target) {
                break;
            }else if (sum < target) {
                i++;
            }else {
                j--;
            }
        }
        return new int[]{i, j};
    }
}
```



## [剑指 Offer II 007. 数组中和为 0 的三个数](https://leetcode-cn.com/problems/1fGaJU/)

+ 三数之和

+ 排序，固定i
+ 移动i， j
+ 注意避免重复

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n - 2; i++) {
            // 找下一个不同的i， 除0之外
            if (i != 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            if (nums[i] > 0) break;
            // j, k
            int j = i + 1, k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum < 0) {
                    j += 1;
                }else if (sum > 0) {
                    k -= 1;
                }else {
                    ans.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    // 更新j, k (事实上只需要更新j到下一个不同的元素即可)
                    j += 1;
                    k -= 1;
                    while (j < k && nums[j] == nums[j - 1]) {
                        j += 1;
                    }
                    while (j < k && nums[k] == nums[k + 1]) {
                        k -= 1;
                    }
                }
            }
        }
        return ans;
    }
}
```



## [剑指 Offer II 008. 和大于等于 target 的最短子数组](https://leetcode-cn.com/problems/2VG8Kg/)

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int i = 0, j = -1;
        int n = nums.length;
        int min_len = nums.length + 1;
        int sum = 0;
        while (j < n) {
            if (sum < target) {
                j += 1;
                if (j < n) sum += nums[j];
            }else {
                min_len = Math.min(min_len, j - i + 1);
                sum -= nums[i];
                i += 1;
            }
        }
        if (min_len == n + 1) return 0;
        return min_len;
    }
}
```



## [剑指 Offer II 009. 乘积小于 K 的子数组(!!!!)](https://leetcode-cn.com/problems/ZVAVXX/)

+ 晕了

```java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k == 0) return 0;
        int i = 0, j = 0;
        int mul = 1;
        int n = nums.length;
        int cnt = 0;
        while (i < n) {
            while (j < n && mul < k) {
                mul *= nums[j];
                j += 1;
            }
            if (mul >= k) {
                cnt += Math.max(0, j - i - 1);
            }else {
                cnt += Math.max(0, j - i);
            }
            mul /= nums[i];
            i += 1;
        }
        return cnt;
    }
}
```



## [剑指 Offer II 010. 和为 k 的子数组](https://leetcode-cn.com/problems/QTMn0o/)

+ 前缀和
+ 哈希表储存前缀和的个数

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> cnt = new HashMap<>();
        int pre_sum = 0;
        cnt.put(0, 1);
        int ans = 0;
        for (int num : nums) {
            pre_sum += num;
            int target = pre_sum - k;
            ans += cnt.getOrDefault(target, 0);
            cnt.put(pre_sum, 1 + cnt.getOrDefault(pre_sum, 0));
        }
        return ans;
    }  
}
```



## [剑指 Offer II 011. 0 和 1 个数相同的子数组](https://leetcode-cn.com/problems/A1NYOS/)

+ 将0看作-1， 即找和为0的子数组

```java
class Solution {
    public int findMaxLength(int[] nums) {
        Map<Integer, Integer> sum2idx = new HashMap<>();
        sum2idx.put(0, -1);
        int pre_sum = 0;
        int max_len = 0;
        for (int i = 0; i < nums.length; i ++) {
            int num = nums[i];
            if (num == 0) num = -1;
            pre_sum += num;
            int pre_idx = sum2idx.getOrDefault(pre_sum, -2);
            if (pre_idx != -2) {
                max_len = Math.max(max_len, i - pre_idx);
            }else {
                sum2idx.put(pre_sum, i);
            }
        }
        return max_len;
    }
}
```



## [剑指 Offer II 012. 左右两边子数组的和相等](https://leetcode-cn.com/problems/tvdfij/)

+ 从左的前缀和 + 从右的前缀和

```java
class Solution {
    public int pivotIndex(int[] nums) {
        int n = nums.length;
        int[] left_sum = new int[n];
        int[] right_sum = new int[n];
        for (int i = n - 2; i >= 0; i--) {
            right_sum[i] = right_sum[i + 1] + nums[i + 1];
        }
        for (int i = 0; i < n; i++) {
            if (i != 0) {
                left_sum[i] = left_sum[i - 1] + nums[i - 1];
            }
            if (left_sum[i] == right_sum[i]) {
                return i;
            }
        }
        return -1;
    }
}
```

+ 不需要right_sum 直接total - pre_sum

```java
class Solution {
    public int pivotIndex(int[] nums) {
        int n = nums.length;
        int total = 0;
        for (int num : nums) {
            total += num;
        }
        int left_sum = 0;
        for (int i = 0; i < n; i++) {
            int right_sum = total - left_sum - nums[i];
            if (left_sum == right_sum) {
                return i;
            }
            left_sum += nums[i];
        }
        return -1;
    }
}
```



## [剑指 Offer II 013. 二维子矩阵的和](https://leetcode-cn.com/problems/O4NDxx/)

+ 二维前缀和

```java
class NumMatrix {
    private int[][] pre_sum;
    int m;
    int n;

    public NumMatrix(int[][] matrix) {
        this.m = matrix.length;
        this.n = matrix[0].length;
        pre_sum = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                pre_sum[i][j] = matrix[i][j] + get(i - 1, j) + get(i, j - 1) - get(i - 1, j - 1);
            }
        }
    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
        return get(row2, col2) - get(row2, col1 - 1) - get(row1 - 1, col2) + get(row1 - 1, col1 - 1);
    }

    int get(int row, int col) {
        if (row < 0 || col < 0) {
            return 0;
        }
        return pre_sum[row][col];
    }
}

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix obj = new NumMatrix(matrix);
 * int param_1 = obj.sumRegion(row1,col1,row2,col2);
 */
```



## [剑指 Offer II 014. 字符串中的变位词](https://leetcode-cn.com/problems/MPnaiL/)

+ 以s1长度作为滑动窗口的长度
+ 用diff记录两字符串的差异，当diff == 0，则找到一个变位词
+ 每加入一个字符
  + 对应字符的计数 - 1， 若从1 变成0， 则diff -= 1。 在此字符上两字符串无差异
  + 若从0 变成 - 1， 则diff += 1，  在此字符上两字符串本无差异，现在有差异了
  + 其他情况diff 不变
+ 每去掉一个字符
  + 对应字符的计数 + 1， 若从-1 变成0， 则diff -= 1。 在此字符上两字符串无差异
  + 若从0 变成  1， 则diff += 1，  在此字符上两字符串本无差异，现在有差异了
  + 其他情况diff 不变

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) return false;
        int n1 = s1.length();
        int n2 = s2.length();
        int i = 0, j = n1 - 1;
        Map<Character, Integer> cnt = new HashMap<>();
        for (int k = 0; k < n1; k++) {
            char s = s1.charAt(k);
            cnt.put(s, 1 + cnt.getOrDefault(s, 0));
        }
        int diff = cnt.size();
        for (int k = 0; k < n1; k++) {
            char s = s2.charAt(k);
            cnt.put(s, cnt.getOrDefault(s, 0) - 1);
            int val = cnt.get(s);
            if (val == 0) diff -= 1;
            if (val == -1) diff += 1;
        }
        while (j < n2) {
            if (diff == 0) return true;
            // 加入 j + 1
            if (j + 1 == n2) break;
            char s = s2.charAt(j + 1);
            cnt.put(s, cnt.getOrDefault(s, 0) - 1);
            int val = cnt.get(s);
            if (val == 0) diff -= 1;
            if (val == -1) diff += 1;
            // 移除 i
            s = s2.charAt(i);
            cnt.put(s, cnt.getOrDefault(s, 0) + 1);
            val = cnt.get(s);
            if (val == 0) diff -= 1;
            if (val == 1) diff += 1;
            i += 1;
            j += 1;
        }
        return false;
    }
}
```



## [剑指 Offer II 015. 字符串中的所有变位词](https://leetcode-cn.com/problems/VabMRr/)

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> ans = new ArrayList<>();
        if (s.length() < p.length()) return ans;
        int n1 = s.length();
        int n2 = p.length();
        int i = 0, j = -1;
        int diff = 0;
        int[] cnt = new int[26];
        // 加入p的字符
        for (int k = 0; k < n2; k++) {
            char ch = p.charAt(k);
            if (cnt[ch - 'a']++ == 0) {
                diff += 1;
            }
        }
        while (j < n1) {
            if (diff == 0) ans.add(i);
            char ch;
            if (j - i + 1 == n2) {
                // 移除i
                ch = s.charAt(i);
                cnt[ch - 'a'] += 1;
                if (cnt[ch - 'a'] == 0) diff -= 1;
                if (cnt[ch - 'a'] == 1) diff += 1;
                i += 1;
            }
            // 加入 j + 1
            if (j + 1 == n1) break;
            ch = s.charAt(j + 1);
            cnt[ch - 'a'] -= 1;
            if (cnt[ch - 'a'] == 0) diff -= 1;
            if (cnt[ch - 'a'] == -1) diff += 1;
            j += 1;
        }
        return ans;
    }
}
```



## [剑指 Offer II 016. 不含重复字符的最长子字符串](https://leetcode-cn.com/problems/wtcaE1/)

+ 滑动窗口
+ 右移j直到有重复
+ 更新max_len
+ 移除i，i+=1

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int i = 0, j = 0;
        Set<Character> hashset = new HashSet<>();
        int max_len = 0;
        int n = s.length();
        while (j < n) {
            while (j < n && !hashset.contains(s.charAt(j))) {
                hashset.add(s.charAt(j));
                j += 1;
            }
            max_len = Math.max(max_len, j - i);
            hashset.remove(s.charAt(i));
            i += 1;
        }
        return max_len;
    }
}
```



## [剑指 Offer II 017. 含有所有字符的最短字符串](https://leetcode-cn.com/problems/M1oyTv/)

+ diff 记录差异，不考虑t中没有的字符
+ diff != 0， 加入j
  + 对应字符的计数 - 1， 若从1 变成0， 则diff -= 1
+ diff == 0, 移除 i 
  + 对应字符的计数 +1， 若从0 变成1， 则diff += 1

```java
class Solution {
    public String minWindow(String s, String t) {
        if (t.length() > s.length()) return "";
        int n1 = s.length();
        int n2 = t.length();
        Map<Character, Integer> cnt = new HashMap<>();
        for (int i = 0; i < n2; i++) {
            cnt.put(t.charAt(i), 1 + cnt.getOrDefault(t.charAt(i), 0));
        }
        int diff = cnt.size();
        int left = -1, right = n1 + 2;
        int i = 0, j = 0;
        char ch;
        while (i < n1) {
            while (j < n1 && diff != 0) {
                ch = s.charAt(j);
                if (!cnt.containsKey(ch)) {
                    j += 1;
                    continue;
                }
                //加入 j 
                cnt.put(ch, cnt.get(ch) - 1);
                if (cnt.get(ch) == 0) diff -= 1;
                j += 1;
            }
            // 若diff == 0，更新left, right
            if (diff == 0 && right - left > j - i) {
                left = i;
                right = j;
            }
            // 移除 i
            ch = s.charAt(i);
            if (!cnt.containsKey(ch)) {
                i += 1;
                continue;
            }
            cnt.put(ch, cnt.get(ch) + 1);
            if (cnt.get(ch) == 1) diff += 1;
            i += 1;
        }
        if (left == -1) return "";
        return s.substring(left, right);
    }
}
```



## [剑指 Offer II 018. 有效的回文](https://leetcode-cn.com/problems/XltzEq/)

+ 双指针i ,j
+ 若不是字母或数字，跳过
+ 判断是否相等
+ Java 中 Character类的方法

```java
class Solution {
    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            while (i < j && !Character.isLetterOrDigit(s.charAt(i))) {
                i += 1;
            }
            while (i < j && !Character.isLetterOrDigit(s.charAt(j))) {
                j -= 1;
            }
            if (Character.toLowerCase(s.charAt(i)) != Character.toLowerCase(s.charAt(j))) {
                return false;
            }
            i += 1;
            j -= 1;
        }
        return true;
    }
}
```



```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i = 0
        j = len(s) - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            if s[i].lower() != s[j].lower():
                return False
            i += 1
            j -= 1
        return True 
```



## [剑指 Offer II 019. 最多删除一个字符得到回文](https://leetcode-cn.com/problems/RQku0D/)

+ 移动i，j判断字符是否相等
+ 若不相等，则判断 i + 1 - j 和 i - j  - 1是否有一个是回文串

```java
class Solution {
    public boolean validPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                return valid(s, i + 1, j) || valid(s, i, j - 1);
            }
            i += 1;
            j -= 1;
        }
        return true;
    }

    boolean valid(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left += 1;
            right -= 1;
        }   
        return true;
    }
}
```



## [剑指 Offer II 020. 回文子字符串的个数](https://leetcode-cn.com/problems/a7VOhD/)

+ 一个回文串去掉两头还是回文串
+ `dp[i][j]` 表示 该子串是否是回文
+ 需要 s[i] == s[j] 且 `dp[i - 1][j - 1]` == true
+ 初始化：
  + 长度为1的是回文串
  + 长度为2的判断两个字符是否相等
  + 可以用一个get()函数 使代码统一

```java
class Solution {
    boolean[][] dp;

    public int countSubstrings(String s) {
        int n = s.length();
        dp = new boolean[n][n];
        int ans = 0;
        // 初始化长度为1
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
            ans += 1;
        }
        // l 表示子串长度
        // j - i + 1 = l
        for (int l = 2; l <= n; l++) {
            for (int i = 0; i <= n - l; i++) {
                int j = l + i - 1;
                dp[i][j] = (s.charAt(i) == s.charAt(j)) && get(i + 1, j - 1);
                if (dp[i][j]) ans += 1;
            }
        }
        return ans;
    }

    private boolean get(int i, int j) {
        if (i > j) return true;
        return dp[i][j];
    }
}
```



## [剑指 Offer II 021. 删除链表的倒数第 n 个结点](https://leetcode-cn.com/problems/SLwz0R/)

+ 创建pre指向head
+ fast先走n + 1 步
+ fast指向null时， slow指向倒数n + 1个
+ 删除slow的下一个结点

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode pre = new ListNode(-1, head);
        ListNode slow = pre, fast = pre;
        for (int i = 0; i < n + 1; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return pre.next;
    }
}
```



## [剑指 Offer II 022. 链表中环的入口节点](https://leetcode-cn.com/problems/c32eOV/)

+ 以HashSet存结点

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        Set<ListNode> hset = new HashSet<>();
        while (head != null) {
            if (hset.contains(head)) return head;
            hset.add(head);
            head = head.next;
        }
        return null;
    }
}
```



+ 快慢指针
+ 分析快慢指针相遇的结点位置
+ https://leetcode-cn.com/problems/linked-list-cycle-ii/solution/linked-list-cycle-ii-kuai-man-zhi-zhen-shuang-zhi-/

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode pre = new ListNode(-1);
        pre.next = head;
        ListNode fast = pre, slow = pre;
        while (fast != null) {
            if (fast.next == null) return null;
            slow = slow.next;
            fast = fast.next.next;
            if (fast == slow) break;
        }
        if (fast == null) return null;
        while (pre != slow) {
            pre = pre.next;
            slow = slow.next;
        }
        return slow;
    }
}
```



## [剑指 Offer II 023. 两个链表的第一个重合节点](https://leetcode-cn.com/problems/3u1WK4/)

+ A走到底从B的头开始，B也一样
+ 若有交点一定在交点出汇合
+ 若没有交点，A == null， B == null 也会退出循环

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A_copy = headA, B_copy = headB;
        if (headA == null || headB == null) return null;
        while (headA != headB) {
            headA = headA == null ? B_copy : headA.next;
            headB = headB == null ? A_copy : headB.next;
        }
        return headA;
    }
}
```



## [剑指 Offer II 024. 反转链表](https://leetcode-cn.com/problems/UHnkqh/)

+ 迭代
+ pre, cur
+ cur 指向pre
+ cur.next 提前保存

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null, cur = head;
        ListNode next;
        while (cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
```



+ 递归

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return head;
        ListNode newhead = reverseList(head.next);
        // head.next 成为了链表尾
        head.next.next = head;
        head.next = null;
        return newhead;
    }
}
```



## [剑指 Offer II 025. 链表中的两数相加](https://leetcode-cn.com/problems/lMSNwu/)

+ 可以先翻转链表
+ 再按位相加
+ 也可以用stack

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        l1 = reverseList(l1);
        l2 = reverseList(l2);
        int carry = 0;
        int a, b;
        ListNode head = null;
        while (l1 != null || l2 != null || carry > 0) {
            if (l1 == null) {
                a = 0;
            }else {
                a = l1.val;
                l1 = l1.next;
            }
            if (l2 == null) {
                b = 0;
            }else {
                b = l2.val;
                l2 = l2.next;
            }
            int val = a + b + carry;
            carry = val / 10;
            val = val % 10;
            ListNode node = new ListNode(val);
            node.next = head;
            head = node;
        }
        return head;
    }

    ListNode reverseList(ListNode head) {
        ListNode pre = null, cur = head;
        ListNode next;
        while (cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
```



## [剑指 Offer II 026. 重排链表](https://leetcode-cn.com/problems/LGjMqU/)

+ 寻找链表中点
+ 后半链表逆序
+ 合并链表，原地合并(zigzag)

```java
class Solution {
    public void reorderList(ListNode head) {
        // 找链表中点，即左半边的尾部
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode l1 = head;
        ListNode l2 = slow.next;
        slow.next = null;
        // 翻转l2
        l2 = reverseList(l2);
        // 合并l1 l2， l1的长度 >= l2的长度
        ListNode l1_temp;
        ListNode l2_temp;
        while (l2 != null) {
            l1_temp = l1.next;
            l2_temp = l2.next;
            l1.next = l2;
            l2.next = l1_temp;
            l1 = l1_temp;
            l2 = l2_temp;
        }
    }

    public ListNode reverseList(ListNode head) {
        ListNode pre = null, cur = head;
        ListNode next;
        while (cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
```



## [剑指 Offer II 027. 回文链表](https://leetcode-cn.com/problems/aMhZSa/)

+ 能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题
+ 找到链表中点
+ 后半链表逆序
+ 判断是否相等
+ l1若多出一个结点，不用判断

```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        // 找链表中点，即左半边的尾部
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode l1 = head;
        ListNode l2 = slow.next;
        slow.next = null;
        // 翻转l2
        l2 = reverseList(l2);
        // 判断是否相等 l1 >= l2的长度
        while (l2 != null) {
            if (l1.val != l2.val) return false;
            l1 = l1.next;
            l2 = l2.next;
        }
        return true;
    }

    public ListNode reverseList(ListNode head) {
        ListNode pre = null, cur = head;
        ListNode next;
        while (cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
```



## [剑指 Offer II 028. 展平多级双向链表](https://leetcode-cn.com/problems/Qv1Da2/)

+ 先child后next的先序遍历
+ 递归DFS
+ 要先记录child和next 在往下递归

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node prev;
    public Node next;
    public Node child;
};
*/

class Solution {
    Node pre;

    public Node flatten(Node head) {
        if (head == null) return null;
        pre = new Node();
        pre.next = head;
        DFS(head);
        // head还是头结点，不过prev = pre, 需要设置成null
        head.prev = null;
        return head;
    }
    void DFS(Node head) {
        if (head == null) return;
        Node child = head.child;
        Node next = head.next;
        head.child = null;
        pre.next = head;
        head.prev = pre;
        pre = head;
        DFS(child);
        DFS(next);
    }
}
```



+ 递归
+ 用一个stack辅助

```java
class Solution {

    public Node flatten(Node head) {
        if (head == null) return null;
        Node pre = new Node();
        pre.next = head;
        LinkedList<Node> stack = new LinkedList<>();
        stack.addLast(head);
        while (!stack.isEmpty()) {
            Node node = stack.removeLast();
            while (node != null) {
                // node.next 入栈
                if (node.next != null) stack.addLast(node.next);
                // 记录child
                Node child = node.child;
                // 连接node
                node.child = null;
                node.prev = pre;
                pre.next = node;
                pre = node;
                node = child;
            }
        }
        head.prev = null;
        return head;
    }
}
```



## [剑指 Offer II 029. 排序的循环链表](https://leetcode-cn.com/problems/4ueAj6/)

+ 判断是否能插入在cur和next之间
+ 需要额外考虑插入值比最大值大或者比最小值小
+ 需要额外考虑所有元素相同，且与插入值不同，此时cur会回到head，break掉之后，随意插入即可

```java
class Solution {
    public Node insert(Node head, int insertVal) {
        Node newnode = new Node(insertVal);
        if (head == null) {
            newnode.next = newnode;
            return newnode;
        }
        if (head == head.next) {
            head.next = newnode;
            newnode.next = head;
            return head;
        }
        Node cur = head; 
        Node next = cur.next;
        while (true) {
            if (cur.val <= next.val) {
                if (cur.val <= insertVal && insertVal <= next.val) {
                    cur.next = newnode;
                    newnode.next = next;
                    return head;
                }
            }else {
                if (insertVal > cur.val || insertVal < next.val) {
                    cur.next = newnode;
                    newnode.next = next;
                    return head;
                }
            }
            cur = cur.next;
            next = next.next;
            if (cur == head) break;
        }
        cur.next = newnode;
        newnode.next = next;
        return head;
    }
}
```

```java
class Solution {
    public Node insert(Node head, int insertVal) {
        Node newnode = new Node(insertVal);
        if (head == null) {
            newnode.next = newnode;
            return newnode;
        }
        if (head == head.next) {
            helper(head, head, newnode);
            return head;
        }
        Node cur = head; 
        Node next = cur.next;
        while (true) {
            if (cur.val <= next.val) {
                if (cur.val <= insertVal && insertVal <= next.val) {
                    helper(cur, next, newnode);
                    return head;
                }
            }else {
                if (insertVal > cur.val || insertVal < next.val) {
                    helper(cur, next, newnode);
                    return head;
                }
            }
            cur = cur.next;
            next = next.next;
            if (cur == head) break;
        }
        helper(cur, next, newnode);
        return head;
    }

    private void helper(Node cur, Node next, Node newnode) {
        cur.next = newnode;
        newnode.next = next;
    }
}
```



## [剑指 Offer II 030. 插入、删除和随机访问都是 O(1) 的容器](https://leetcode-cn.com/problems/FortPu/)

+ 哈希表存储存储值到索引的映射
+ 动态数组存储元素值
+ 将要删除元素和最后一个元素交换, 将最后一个元素删除
+ Java 中 的 `Random` 实现返回随机val

```java
class RandomizedSet {
    Map<Integer, Integer> val2idx;
    List<Integer> list;
    Random random;

    /** Initialize your data structure here. */
    public RandomizedSet() {
        val2idx = new HashMap<>();
        list = new ArrayList<>();
        random = new Random();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if (val2idx.containsKey(val)) {
            return false;
        }
        // 加入到list, 加入val - list.length() - 1 到map
        list.add(val);
        val2idx.put(val, list.size() - 1);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if (!val2idx.containsKey(val)) {
            return false;
        }
        // 把val换到list末尾并删除
        int val_idx = val2idx.get(val);
        int last_idx = list.size() - 1;
        // 把当前末尾元素对应的idx改成val_idx， 交换位置，删除val(list, val2idx)
        int temp = list.get(last_idx);
        val2idx.put(temp, val_idx);
        // 不需要set，之后要删除  list.set(last_idx, val);
        list.set(val_idx, temp);
        list.remove(last_idx);
        val2idx.remove(val);
        return true;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        int idx = random.nextInt(list.size());
        return list.get(idx);
    }
}
```



```python
import random

class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.map = dict()
        self.list = []

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.map:
            return False
        self.list.append(val)
        self.map[val] = len(self.list) - 1
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.map:
            return False 
        val_idx = self.map.get(val)
        self.map[self.list[-1]] = val_idx 
        self.list[val_idx], self.list[-1] = self.list[-1], self.list[val_idx]
        self.list.pop()
        self.map.pop(val)
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return random.choice(self.list)
```





## [剑指 Offer II 031. 最近最少使用缓存](https://leetcode-cn.com/problems/OrIXps/)

+ 定长双向链表+哈希表
+ 需要四个方法
+ add_to_head
+ move_to_head
+ remove_tail
+ remove_node
+ 用两个dummy node 标记head, tail

```java
class LinkedNode {
    int key;
    int val;
    LinkedNode prev;
    LinkedNode next;

    public LinkedNode() {}
    public LinkedNode(int _key, int _val) {
        key = _key;
        val = _val;
    }
}

class LRUCache {
    int size;
    int capacity;
    LinkedNode head;
    LinkedNode tail;
    Map<Integer, LinkedNode> map;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        head = new LinkedNode();
        tail = new LinkedNode();
        head.next = tail;
        tail.prev = head;
        map = new HashMap<>();
    }
    
    public int get(int key) {
        if (map.containsKey(key)) {
            LinkedNode node = map.get(key);
            moveToHead(node);
            return node.val;
        }
        return -1;
    }
    
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            LinkedNode node = map.get(key);
            node.val = value;
            moveToHead(node);
        }else {
            LinkedNode newnode = new LinkedNode(key, value);
            if (++size > capacity) {
                // 删除尾结点，同时删除map中的key
                int tail_key = removeTail();
                map.remove(tail_key);
                size -= 1;
            }
            // 加入node，同时加入map
            map.put(key, newnode);
            addToHead(newnode);
            //System.out.println(map.toString());
        }

    }

    void addToHead(LinkedNode node) {
        head.next.prev = node;
        node.next = head.next;
        node.prev = head;
        head.next = node;
    }

    void removeNode(LinkedNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    int removeTail() {
        int key = tail.prev.key;
        removeNode(tail.prev);
        return key;
    }

    void moveToHead(LinkedNode node) {
        removeNode(node);
        addToHead(node);
    }
}
```





## [剑指 Offer II 032. 有效的变位词](https://leetcode-cn.com/problems/dKk3P7/)

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length() || s.equals(t)) return false;

        Map<Character, Integer> cnt = new HashMap<>(64);
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            cnt.put(ch, cnt.getOrDefault(ch, 0) + 1);
        }

        for (int i = 0; i < t.length(); i++) {
            char ch = t.charAt(i);
            if (!cnt.containsKey(ch)) return false;
            cnt.put(ch, cnt.get(ch) - 1);
            if (cnt.get(ch) < 0) return false;
        }
        return true;
    }
}
```



## [剑指 Offer II 033. 变位词组](https://leetcode-cn.com/problems/sfvd7V/)

+ 暴力遍历, 判断是否是变位词

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> ans = new ArrayList<>();
        List<String> head = new ArrayList<>();
        for (String word : strs) {
            boolean new_head = true;
            for (int i = 0; i < head.size(); i++) {
                if (!isAnagram(head.get(i), word)) continue;
                ans.get(i).add(word);
                new_head = false;
                break;
            }
            if (new_head) {
                head.add(word);
                List<String> temp = new ArrayList<>();
                temp.add(word);
                ans.add(temp);
            }
        }
        return ans;

    }

    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;

        Map<Character, Integer> cnt = new HashMap<>(64);
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            cnt.put(ch, cnt.getOrDefault(ch, 0) + 1);
        }

        for (int i = 0; i < t.length(); i++) {
            char ch = t.charAt(i);
            if (!cnt.containsKey(ch)) return false;
            cnt.put(ch, cnt.get(ch) - 1);
            if (cnt.get(ch) < 0) return false;
        }
        return true;
    }
}
```



+ 先对所有字符串排序
+ 哈希表记录排序后相同的字符串

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] arr = s.toCharArray();
            Arrays.sort(arr);
            String key = new String(arr);
            if (!map.containsKey(key)) {
                map.put(key, new ArrayList<String>());
            }
            map.get(key).add(s);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
```

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        Map = collections.defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            Map[key].append(s)
        
        return list(Map.values())
```



## [剑指 Offer II 034. 外星语言是否排序](https://leetcode-cn.com/problems/lwyVBB/)

```java
class Solution {
    Map<Character, Integer> map;

    public boolean isAlienSorted(String[] words, String order) {
        map = new HashMap<>(50);
        for (int i = 0; i < order.length(); i++) {
            map.put(order.charAt(i), i);
        }

        for (int i = 0; i < words.length - 1; i++) {
            if (!isordered(words[i], words[i + 1])) {
                return false;
            }
        }
        return true;
    }

    boolean isordered(String s1, String s2) {
        for (int i = 0; i < Math.min(s1.length(), s2.length()); i++) {
            char ch1 = s1.charAt(i);
            char ch2 = s2.charAt(i);
            if (map.get(ch1) < map.get(ch2)) return true;
            if (map.get(ch1) > map.get(ch2)) return false;
        }
        return s1.length() <= s2.length();
    }
}
```



## [剑指 Offer II 035. 最小时间差](https://leetcode-cn.com/problems/569nqc/)

+ 按时间排序，计算相邻时间差
+ 注意把最小的时间加1440放在最后

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





+ `1440`分钟，开一个`1440`长度的布尔数组模拟哈希表，把时间换算成`0~1439`之间的数值，将数值对应数组中的位置设置为`true`
+ 遍历数组，找离得最近的两个时间点, 就不需要排序了
+ 注意把最小的时间加1440放在最后





## [剑指 Offer II 036. 后缀表达式](https://leetcode-cn.com/problems/8Zf90G/)

逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。

+ 平常使用的算式则是一种中缀表达式，如 ( 1 + 2 ) * ( 3 + 4 ) 
+ 该算式的逆波兰表达式写法为 ( ( 1 2 + ) ( 3 4 + ) * ) 

逆波兰表达式主要有以下两个优点：

+ 去掉括号后表达式无歧义，上式即便写成 1 2 + 3 4 + * 也可以依据次序计算出正确结果。
+ 适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中。

```java
class Solution {
    public int evalRPN(String[] tokens) {
        LinkedList<Integer> stack = new LinkedList<>();
        for (String s : tokens) {
            if (is_ops(s)) {
                int sec = stack.removeLast();
                int fir = stack.removeLast();
                if (s.equals("+")) {
                    stack.addLast(fir + sec);
                }else if (s.equals("-")) {
                    stack.addLast(fir - sec);
                }else if (s.equals("*")) {
                    stack.addLast(fir * sec);
                }else {
                    stack.addLast(fir / sec);
                }
            }else {
                stack.addLast(Integer.parseInt(s));
            }
        }
        return stack.removeLast();
    }

    boolean is_ops(String s) {
        return s.equals("+") || s.equals("-") || s.equals("*") || s.equals("/");
    }
}
```



## [剑指 Offer II 037. 小行星碰撞](https://leetcode-cn.com/problems/XagZNi/)

+ 注释
+ 要想清楚碰撞的情况

```java
class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> stack = new LinkedList<>();

        for (int i = 0; i < asteroids.length; i++) {
            int ast = asteroids[i];
            // 大于0不会和stack中的ast碰撞，stack为空直接加入
            if (ast > 0 || stack.isEmpty()) {
                stack.addLast(ast);
            }else {
                while (!stack.isEmpty() && stack.getLast() > 0) {
                    int last = stack.removeLast();
                    if (last > -ast) { // ast消失，把last放回去，结束while
                        ast = 0;
                        stack.addLast(last);
                        break;
                    }else if (last == -ast) { // 都消失, 直接结束while
                        ast = 0;
                        break;
                    }
                    // 还有一种情况，last消失，继续while循环即可
                }
                // 若 ast != 0， 说明stack空，需要加入ast
                if (ast != 0) stack.addLast(ast);
            }
        }
        int[] ans = new int[stack.size()];
        int i = 0;
        for (int ast : stack) {
            ans[i] = ast;
            i += 1;
        }
        return ans;
    }
}
```



## [剑指 Offer II 038. 每日温度](https://leetcode-cn.com/problems/iIQa4I/)

+ 单调栈
+ 从后往前，若当前temp大于stack[-1]，则stack[-1]不需要留下了，因为有更高且更靠前的temp

```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        Deque<Integer> stack = new LinkedList<>();
        int[] ans = new int[temperatures.length];
        for (int i = temperatures.length - 1; i >= 0; i--) {
            int temp = temperatures[i];
            while (!stack.isEmpty() && temperatures[stack.getLast()] <= temp) {
                stack.removeLast();
            }
            if (stack.isEmpty()) {
                ans[i] = 0;
            }else {
                ans[i] = stack.getLast() - i;
            }
            stack.addLast(i);
        }
        return ans;
    }
}
```



## [剑指 Offer II 039. 直方图最大矩形面积](https://leetcode-cn.com/problems/0ynMMM/)

+ 需要从每个位置向左找到第一个更小的index
+ 向右找到第一个更小的index， 就确定了高和宽边
+ 两个单调stack，栈尾部元素最大，遇到大于等于的可以弹出，因为当前元素更小，会是更接近的小元素

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length; 
        int[] left = new int[n];
        int[] right = new int[n];
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            int h = heights[i];
            while (!stack.isEmpty() && heights[stack.getLast()] >= h) {
                stack.removeLast();
            }
            if (stack.isEmpty()) {
                left[i] = -1;
            }else {
                left[i] = stack.getLast();
            }
            stack.add(i);
        }
        stack.clear();
        for (int i = n - 1; i >= 0; i--) {
            int h = heights[i];
            while (!stack.isEmpty() && heights[stack.getLast()] >= h) {
                stack.removeLast();
            }
            if (stack.isEmpty()) {
                right[i] = n;
            }else {
                right[i] = stack.getLast();
            }
            stack.add(i);
        }
        
        int max_area = -1;
        for (int i = 0; i < n; i++) {
            max_area = Math.max(max_area, (right[i] - left[i] - 1) * heights[i]);
        }

        return max_area;
    }
}
```



+ 只遍历一次
+ 一个元素弹出时，就找到了它右边第一个小元素

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length; 
        int[] left = new int[n];
        int[] right = new int[n];
        Arrays.fill(right, n);
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            int h = heights[i];
            while (!stack.isEmpty() && heights[stack.getLast()] >= h) {
                int last = stack.removeLast();
                right[last] = i;
            }
            if (stack.isEmpty()) {
                left[i] = -1;
            }else {
                left[i] = stack.getLast();
            }
            stack.add(i);
        }

        int max_area = -1;
        for (int i = 0; i < n; i++) {
            max_area = Math.max(max_area, (right[i] - left[i] - 1) * heights[i]);
        }

        return max_area;
    }
}
```



## [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

+ 以每一行为底，看作柱状图，计算heights

```java
class Solution {
    public int maximalRectangle(String[] matrix) {
        int m = matrix.length;
        if (m == 0) return 0;
        int n = matrix[0].length();
        if (n == 0) return 0;
        int[] heights = new int[n];
        int max_area = 0;
        for (String row : matrix) {
            for (int j = 0; j < n; j++) {
                if (row.charAt(j) == '0') {
                    heights[j] = 0;
                }else {
                    heights[j] += 1;
                }
            }
            int area = largestRectangleArea(heights);
            max_area = Math.max(max_area, area);
        }
        return max_area;
    }

    int largestRectangleArea(int[] heights) {
        int n = heights.length; 
        int[] left = new int[n];
        int[] right = new int[n];
        Arrays.fill(right, n);
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            int h = heights[i];
            while (!stack.isEmpty() && heights[stack.getLast()] >= h) {
                int last = stack.removeLast();
                right[last] = i;
            }
            if (stack.isEmpty()) {
                left[i] = -1;
            }else {
                left[i] = stack.getLast();
            }
            stack.add(i);
        }

        int max_area = -1;
        for (int i = 0; i < n; i++) {
            max_area = Math.max(max_area, (right[i] - left[i] - 1) * heights[i]);
        }

        return max_area;
    }
}
```





## [剑指 Offer II 041. 滑动窗口的平均值](https://leetcode-cn.com/problems/qIsx9U/)

+ 维护一个deque
+ 超出size则removeleft

```python
class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.len = 0
        self.deque = collections.deque()
        self.sum = 0

    def next(self, val: int) -> float:
        self.len += 1
        self.deque.append(val)
        self.sum += val
        if self.len > self.size:
            self.len -= 1
            self.sum -= self.deque.popleft()
        return self.sum / self.len
```



```java
class MovingAverage {
    int size;
    int length = 0;
    double sum = 0.0;
    Deque<Integer> deque = new LinkedList<>();

    /** Initialize your data structure here. */
    public MovingAverage(int size) {
        this.size = size;
    }
    
    public double next(int val) {
        length += 1;
        sum += val;
        deque.addLast(val);
        if (length > size) {
            sum -= deque.removeFirst();
            length -= 1;
        }
        return sum / length;
    }
}
```



## [剑指 Offer II 042. 最近请求次数](https://leetcode-cn.com/problems/H8086Q/)

+ 维护deque
+ 每次去掉 < t - 3000 的头部元素

```python
class RecentCounter:

    def __init__(self):
        self.deque = collections.deque()

    def ping(self, t: int) -> int:
        self.deque.append(t)
        while self.deque[0] < t - 3000:
            self.deque.popleft()
        return len(self.deque)
```



```java
class RecentCounter {
    Deque<Integer> deque;

    public RecentCounter() {
        deque = new LinkedList<>();
    }
    
    public int ping(int t) {
        deque.addLast(t);
        while (deque.getFirst() < t - 3000) {
            deque.removeFirst();
        }
        return deque.size();
    }
}
```



## [剑指 Offer II 043. 往完全二叉树添加节点](https://leetcode-cn.com/problems/NaqhDT/)

+ 先深度优先遍历将子节点没填满的node按顺序插入deque
+ 将新节点作为deque第一个元素的子节点
+ 若填满则removefirst

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class CBTInserter {
    TreeNode root;
    Deque<TreeNode> deque;

    public CBTInserter(TreeNode root) {
        this.root = root;
        deque = new LinkedList<>();
        Deque<TreeNode> bfs_deque = new LinkedList<>();
        bfs_deque.addLast(root);
        while (!bfs_deque.isEmpty()) {
            TreeNode node = bfs_deque.removeFirst();
            if (node.left != null) bfs_deque.addLast(node.left);
            if (node.right != null) bfs_deque.addLast(node.right);
            if (node.right == null) deque.addLast(node);
        }
    }
    
    public int insert(int v) {
        TreeNode node = new TreeNode(v);
        TreeNode parent = deque.getFirst();
        if (parent.left == null) {
            parent.left = node;
        }else {
            parent.right = node;
            deque.removeFirst();
        }
        deque.addLast(node);
        return parent.val;
    } 
    
    public TreeNode get_root() {
        return root;
    }
}
```



## [剑指 Offer II 044. 二叉树每层的最大值](https://leetcode-cn.com/problems/hPov7L/)

+ 分层BFS
+ 记录每层最大值

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> largestValues(TreeNode root) {
        if (root == null) return new ArrayList();
        List<Integer> ans = new ArrayList<>();
        Deque<TreeNode> queue = new LinkedList<>();
        queue.addLast(root);
        while (!queue.isEmpty()) {
            int max = Integer.MIN_VALUE;
            int L = queue.size();
            for (int l = 0; l < L; l++) {
                TreeNode node = queue.removeFirst();
                max = Math.max(node.val, max);
                if (node.left != null) queue.addLast(node.left);
                if (node.right != null) queue.addLast(node.right);
            }
            ans.add(max);
        }
        return ans;
    }
}
```



```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        ans = []
        queue = collections.deque()
        queue.append(root)
        min_val = -1 << 31
        while queue:
            layer_max = min_val
            for _ in range(len(queue)):
                node = queue.popleft()
                layer_max = max(layer_max, node.val)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            ans.append(layer_max)
        return ans
```



## [剑指 Offer II 045. 二叉树最底层最左边的值](https://leetcode-cn.com/problems/LwUNpT/)

+ BFS
+ 遍历每一层时记录下第一个val

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int findBottomLeftValue(TreeNode root) {
        int ans = -1;
        Deque<TreeNode> queue = new LinkedList<>();
        queue.addLast(root);
        while (!queue.isEmpty()) {
            int L = queue.size();
            ans = queue.getFirst().val;
            for (int l = 0; l < L; l++) {
                TreeNode node = queue.removeFirst();
                if (node.left != null) queue.addLast(node.left);
                if (node.right != null) queue.addLast(node.right);
            }
        }
        return ans;
    }
}
```



+ DFS求二叉树深度的思路
+ 每一次遍历到最底层遇到的node为答案

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        self.depth = 0
        self.ans = -1
        self.dfs(root, 1)
        return self.ans 
    
    def dfs(self, node, level):
        if node is None:
            return 
        if level > self.depth:
            self.depth = level 
            self.ans = node.val
        self.dfs(node.left, level + 1)
        self.dfs(node.right, level + 1)
```



## [剑指 Offer II 046. 二叉树的右侧视图](https://leetcode-cn.com/problems/WNC0Lk/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) return new ArrayList();
        List<Integer> ans = new ArrayList<>();
        Deque<TreeNode> queue = new LinkedList<>();
        queue.addLast(root);
        while (!queue.isEmpty()) {
            int L = queue.size();
            for (int l = 0; l < L; l++) {
                TreeNode node = queue.removeFirst();
                if (node.left != null) queue.addLast(node.left);
                if (node.right != null) queue.addLast(node.right);
                if (l == L - 1) {
                    ans.add(node.val);
                }
            }
        }
        return ans;
    }
}
```



## [剑指 Offer II 047. 二叉树剪枝](https://leetcode-cn.com/problems/pOCWxh/)

+ DFS
+ 判断左右子树是否是全0，若是全0则改成null，再根据node值返回该节点是否为全0给上一层

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode pruneTree(TreeNode root) {
        boolean is_empty = dfs(root);
        if (is_empty) {
            return null;
        }
        return root;
    }

    // 检查node是否为全0子树
    boolean dfs(TreeNode node) {
        if (node == null) return true;
        boolean left = dfs(node.left);
        boolean right = dfs(node.right);
        if (left) {
            node.left = null;
        }
        if (right) {
            node.right = null;
        }
        if (left && right && node.val == 0) {
            return true;
        }
        return false;
    }
}
```



+ 改进
+ 若全0则返回nulll，否则返回node
+ 这样不需要单独写一个dfs

```java
class Solution {
    public TreeNode pruneTree(TreeNode root) {
        if (root == null) return null;
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);
        if (root.left == null && root.right == null && root.val == 0) {
            return null;
        }
        return root;
    }
}
```



## [剑指 Offer II 048. 序列化与反序列化二叉树](https://leetcode-cn.com/problems/h54YBf/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {
    StringBuilder data;

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        data = new StringBuilder();
        recurserialize(root);
        return data.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] list = data.split(",");
        Deque<String> datalist = new LinkedList<>(Arrays.asList(list));
        return recurdeserialize(datalist);
    }

    public void recurserialize(TreeNode root) {
        if (root == null) {
            data.append("None,");
            return;
        }
        data.append(String.valueOf(root.val) + ",");
        recurserialize(root.left);
        recurserialize(root.right);
    }

    public TreeNode recurdeserialize(Deque<String> datalist) {
        String fir = datalist.removeFirst();
        if ("None".equals(fir)) return null;
        TreeNode node = new TreeNode(Integer.parseInt(fir));
        node.left = recurdeserialize(datalist);
        node.right = recurdeserialize(datalist);
        return node;
    }

}
```



## [剑指 Offer II 049. 从根节点到叶节点的路径数字之和](https://leetcode-cn.com/problems/3Etpl5/)

+ DFS
+ 乘以10 加上当前结点
+ 遇到叶子结点累加到sum上

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int sum = 0;

    public int sumNumbers(TreeNode root) {
        dfs(root, 0);
        return sum;
    }

    private void dfs(TreeNode node, int cur_sum) {
        cur_sum = cur_sum * 10 + node.val;
        if (node.left == null && node.right == null) {
            sum += cur_sum;
            return;
        }
        if (node.left != null) dfs(node.left, cur_sum);
        if (node.right != null) dfs(node.right, cur_sum);
    }

}
```



## [剑指 Offer II 050. 向下的路径节点之和](https://leetcode-cn.com/problems/6eUYwP/)

+ 维护前缀和计数的map
+ dfs的性质，可以公用一个前缀map，注意遍历完node要删除node的prefix

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int cnt = 0;
    Map<Integer, Integer> prefix;

    public int pathSum(TreeNode root, int targetSum) {
        prefix = new HashMap<>();
        // 无前缀，0
        prefix.put(0, 1);
        dfs(root, targetSum, 0);
        return cnt;
    }

    public void dfs(TreeNode node, int targetSum, int sum) {
        if (node == null) return;
        sum += node.val;
        int pre_cnt = prefix.getOrDefault(sum - targetSum, 0);
        cnt += pre_cnt;
        //加入 prefix
        prefix.put(sum, prefix.getOrDefault(sum, 0) + 1);
        // 遍历left right
        dfs(node.left, targetSum, sum);
        dfs(node.right, targetSum, sum);
        // node遍历结束, 删除prefix
        prefix.put(sum, prefix.get(sum) - 1);
    }
}
```



## [剑指 Offer II 051. 节点之和最大的路径](https://leetcode-cn.com/problems/jC7MId/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int maxsum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        dfs(root);
        return maxsum;
    }

    int dfs(TreeNode node) {
        if (node == null) return 0;
        int left = Math.max(dfs(node.left), 0);
        int right = Math.max(dfs(node.right), 0);
        maxsum = Math.max(maxsum, node.val + left + right);
        return node.val + Math.max(left, right);
    }
}
```





## [剑指 Offer II 052. 展平二叉搜索树](https://leetcode-cn.com/problems/NYBBNL/)

+ 递归

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    TreeNode prehead;
    TreeNode res;
    public TreeNode increasingBST(TreeNode root) {
        prehead = new TreeNode(-1);
        res = prehead;
        dfs(root);
        return res.right;
    }

    void dfs(TreeNode node) {
        if (node == null) return;
        dfs(node.left);
        prehead.right = node;
        prehead = prehead.right;
        node.left = null;
        dfs(node.right);
    }
}

```



+ 迭代

```java
class Solution {
    TreeNode prehead;
    TreeNode res;
    public TreeNode increasingBST(TreeNode root) {
        prehead = new TreeNode(-1);
        res = prehead;
        Deque<TreeNode> stack = new LinkedList<>();

        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.addLast(root);
                root = root.left;
            }
            TreeNode head = stack.removeLast();
            head.left = null;
            prehead.right = head;
            prehead = prehead.right;
            root = head.right;
        }
        return res.right;
    }
}
```





## [剑指 Offer II 053. 二叉搜索树中的中序后继](https://leetcode-cn.com/problems/P5rCT8/)

+ DFS
+ 注意left之后要判断是否return

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    TreeNode pre_node = null;
    TreeNode ans = null;
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        dfs(root, p);
        return ans;
    }

    void dfs(TreeNode root, TreeNode p) {
        if (ans != null || root == null) return;
        dfs(root.left, p);
        // 注意
        if (ans != null) return;
        if (pre_node == p) {
            ans = root;
            return;
        }else {
            pre_node = root;
        }
        dfs(root.right, p);
    }
}
```



```java
class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        TreeNode pre_node = null;

        Deque<TreeNode> stack = new LinkedList<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.addLast(root);
                root = root.left;
            }
            TreeNode head = stack.removeLast();
            if (pre_node == p) {
                return head;
            }else {
                pre_node = head;
            }
            root = head.right;
        }
        return null;
    }
}
```



## [剑指 Offer II 054. 所有大于等于节点的值之和](https://leetcode-cn.com/problems/w6cpku/)

+ 二叉搜索数，右中左 则是倒序
+ 用sum记录，右中左 DFS

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int sum = 0;

    public TreeNode convertBST(TreeNode root) {
        dfs(root);
        return root;
    }

    void dfs(TreeNode node) {
        if (node == null) return;
        dfs(node.right);
        sum += node.val;
        node.val = sum;
        dfs(node.left);
    }
}
```



## [剑指 Offer II 055. 二叉搜索树迭代器](https://leetcode-cn.com/problems/kTOapQ/)

+ 迭代中序遍历的思路
+ 每次hasnext，出栈后将node.right 及所有左节点加入stack

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class BSTIterator {
    Deque<TreeNode> stack = new LinkedList<>();

    public BSTIterator(TreeNode root) {
        instack(root);
    }
    
    public int next() {
        TreeNode node = stack.removeLast();
        instack(node.right);
        return node.val;
    }
    
    public boolean hasNext() {
        return !stack.isEmpty();
    }

    private void instack(TreeNode node) {
        while (node != null) {
            stack.addLast(node);
            node = node.left;
        }
    }
}
```



## [剑指 Offer II 056. 二叉搜索树中两个节点之和](https://leetcode-cn.com/problems/opLdQZ/)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    Set<Integer> hset = new HashSet<>();
    boolean ans = false;

    public boolean findTarget(TreeNode root, int k) {
        dfs(root, k);
        return ans;
    }

    public void dfs(TreeNode node, int k) {
        if (ans == true || node == null) return;
        if (hset.contains(k - node.val)) {
            ans = true;
            return;
        }
        hset.add(node.val);
        dfs(node.left, k);
        if (ans == true) return;
        dfs(node.right, k);
    }
}
```



## [剑指 Offer II 057. 值和下标之差都在给定的范围内](https://leetcode-cn.com/problems/7WqeDu/)

![image-20220116152503753](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220116152503753.png)

![image-20220116152519323](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220116152519323.png)

如果当前有序集合中存在相同元素，那么此时程序将直接返回 true\。因此本题中的有序集合无需处理相同元素的情况。

```java
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        int n = nums.length;
        long longt = (long) t;
        TreeSet<Long> set = new TreeSet<>();
        for (int i = 0; i < n; i++) {
            long x = (long) nums[i];
            // 目标是范围是[x-t, x+t], 找到大于等于x-t的最小数, 再判断是否小于x+t
            Long ceil = set.ceiling(x - longt);
            if (ceil != null && ceil <= x + longt) {
                return true;
            };
            set.add(x);
            if (set.size() > k) {
                set.remove((long) nums[i - k]);
            }
        }
        return false;
    }
}
```



## [剑指 Offer II 058. 日程表](https://leetcode-cn.com/problems/fi9suh/)

+ TreeMap start:key
+ 二分查找日程安排的情况来检查新日常安排是否可以预订

```java
class MyCalendar {
    TreeMap<Integer, Integer> map;

    public MyCalendar() {
        map = new TreeMap<>();
    }
    
    public boolean book(int start, int end) {
        Integer s;
        // 先找<=start 的最大s
        s = map.floorKey(start);
        if (s != null && start < map.get(s)) {
            // start >= e 才不会重合
            return false;
        }
        // 找>=start 的最小s
        s = map.ceilingKey(start);
        if (s != null && end > s) {
            // end <= s 才不会重合
            return false;
        }
        // 可以添加，加入map后return true
        map.put(start, end);
        return true;
    }
}
```





## [剑指 Offer II 059. 数据流的第 K 大数值](https://leetcode-cn.com/problems/jBjn9C/)

+ 维护长度为k的小根堆

```java
class KthLargest {
    PriorityQueue<Integer> pq;
    int k;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        pq = new PriorityQueue<>();
        for (int val : nums) {
            pq.add(val);
            if (pq.size() > k) pq.poll();
        }
    }
    
    public int add(int val) {
        pq.add(val);
        if (pq.size() > k) pq.poll();
        return pq.peek();
    }
}
```



## [剑指 Offer II 060. 出现频率最高的 k 个数字](https://leetcode-cn.com/problems/g5c51o/)

+ 先map记录每个数字的出现此次数
+ 维护长度为k的小根堆
+ 自定义一个pair类，重写compareTo 方法

```java
class Pair implements Comparable<Pair>{
    public int key;
    public int value;

    public Pair(int k, int v) {
        key = k;
        value = v;
    }

    @Override 
    public int compareTo​(Pair o) {
        return value - o.value;
    }
}

class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int num : nums) {
            cnt.put(num, 1 + cnt.getOrDefault(num, 0));
        }

        for (int num : cnt.keySet()) {
            Pair pair = new Pair(num, cnt.get(num));
            if (pq.size() < k) {
                pq.add(pair);
                continue;
            }
            if (pair.value > pq.peek().value) {
                pq.poll();
                pq.add(pair);
            }
        }
        int[] ans = new int[k];
        int i = 0;
        for (Pair pair : pq) {
            ans[i] = pair.key;
            i += 1;
        }
        return ans;
    }
}
```



## [剑指 Offer II 061. 和最小的 k 个数对](https://leetcode-cn.com/problems/qn8gGX/)

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
        for (int i = 0; i < Math.min(n, k); i++) {
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



## [剑指 Offer II 062. 实现前缀树](https://leetcode-cn.com/problems/QC3q1f/)

+ 向子节点的指针数组 children
+ 布尔字段 isEnd，表示该节点是否为字符串的结尾

```java
class Trie {
    private Trie[] children;
    private boolean isEnd;

    /** Initialize your data structure here. */
    public Trie() {
        children = new Trie[26];
        isEnd = false;
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                node.children[idx] = new Trie();
            }
            node = node.children[idx];
        }
        // 插入结束后会停在最后一个char处
        node.isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                return false;
            }
            node = node.children[idx];
        }
        return node.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        Trie node = this;
        for (int i = 0; i < prefix.length(); i++) {
            char ch = prefix.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                return false;
            }
            node = node.children[idx];
        }
        return true;
    }
}
```



+ 重复结构封装

```java
class Trie {
    private Trie[] children;
    private boolean isEnd;

    /** Initialize your data structure here. */
    public Trie() {
        children = new Trie[26];
        isEnd = false;
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                node.children[idx] = new Trie();
            }
            node = node.children[idx];
        }
        // 插入结束后会停在最后一个char处
        node.isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        Trie node = prefix(word);
        return node != null && node.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        Trie node = prefix(prefix);
        return node != null;
    }

    // 返回 最后一个word最后一个char对应的结点
    private Trie prefix(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                return null;
            }
            node = node.children[idx];
        }
        return node;   
    }
}
```



## [剑指 Offer II 063. 替换单词](https://leetcode-cn.com/problems/UhWRSj/)

+ 建立前缀树加入词根
+ 每个word找最短的前缀,不存在则加入原word

```java
class Trie {
    private Trie[] children;
    private boolean isEnd;

    /** Initialize your data structure here. */
    public Trie() {
        children = new Trie[26];
        isEnd = false;
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                node.children[idx] = new Trie();
            }
            node = node.children[idx];
        }
        // 插入结束后会停在最后一个char处
        node.isEnd = true;
    }

    // 返回 最短prefix
    public String shortestPrefix(String word) {
        Trie node = this;
        StringBuilder prefix = new StringBuilder();
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                return null;
            }
            prefix.append(ch);
            node = node.children[idx];
            if (node.isEnd) {
                return prefix.toString();
            }
        }
        return null;
    }
}

class Solution {
    public String replaceWords(List<String> dictionary, String sentence) {

        // 建立Trie
        Trie mytrie = new Trie();
        for (String root : dictionary) {
            mytrie.insert(root);
        }
        String[] words = sentence.split(" ");
        String[] ans = new String[words.length];
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            //System.out.println(word);
            String prefix = mytrie.shortestPrefix(word);
            if (prefix == null) {
                ans[i] = word;
            }else {
                ans[i] = prefix;
            }
        }
        return String.join(" ", ans);
    }
}
```



## [剑指 Offer II 064. 神奇的字典](https://leetcode-cn.com/problems/US1pGT/)

+ 建立map 长度 - 》 word list
+ 根据searchword的长度，注意判断是否只差一个单词

```java
class MagicDictionary {
    Map<Integer, List<String>> map;

    /** Initialize your data structure here. */
    public MagicDictionary() {
        map = new HashMap<>();
    }
    
    public void buildDict(String[] dictionary) {
        for (String word : dictionary) {
            int l = word.length();
            if (!map.containsKey(l)) {
                map.put(l, new ArrayList<String>());
            }
            map.get(l).add(word);
        }
    }
    
    public boolean search(String searchWord) {
        List<String> target = map.get(searchWord.length());
        if (target == null) return false;
        for (String t : target) {
            if (distOne(searchWord, t)) {
                return true;
            }
        }
        return false;
    }

    private boolean distOne(String s1, String s2) {
        int cnt = 0;
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) != s2.charAt(i)) {
                cnt += 1;
            }
            if (cnt > 1) return false;
        }
        return cnt == 1;
    }
}
```



## [剑指 Offer II 065. 最短的单词编码](https://leetcode-cn.com/problems/iSwD2y/)

+ 倒序后插入字典树
+ 然后计算所有叶结点对应的string的长度，再加上叶结点的个数
+ 每次插入，记录最后一个节点的到map
+ 检查叶结点是否有child

```java
class Trie {
    public Trie[] children;
    public boolean hasChild;

    /** Initialize your data structure here. */
    public Trie() {
        children = new Trie[26];
        hasChild = false;
    }
    
    /** Inserts a word into the trie. */
    public Trie insert(String word) {
        Trie node = this;
        // 倒序字符串
        for (int i = word.length() - 1; i >= 0; i--) {
            char ch = word.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                node.children[idx] = new Trie();
                node.hasChild = true;
            }
            node = node.children[idx];
        }
        // 插入结束后会停在最后一个char处
        return node;
    }
}

class Solution {
    public int minimumLengthEncoding(String[] words) {
        Trie mytrie = new Trie();
        Map<Trie, Integer> len = new HashMap<>();
        int ans = 0;
        for (String word : words) {
            Trie node = mytrie.insert(word);
            len.put(node, word.length() + 1);
        }
        
        for (Trie node : len.keySet()) {
            if (node.hasChild == false) {
                ans += len.get(node);
            }
        }
        return ans;
    }
}
```



## [剑指 Offer II 066. 单词之和](https://leetcode-cn.com/problems/z1R5dt/)

+ Trie tree
+ insert 时每路过一个结点使其cnt += val
+ 更新key value时，应该为 val - old_value
+ 返回prefix对应最后一个node的cnt

```java
class MapSum {
    Trie mytrie;
    Map<String, Integer> map;
    /** Initialize your data structure here. */
    public MapSum() {
        mytrie = new Trie();
        map = new HashMap<>();
    }
    
    public void insert(String key, int val) {
        int pre_val = map.getOrDefault(key, 0);
        mytrie.insert(key, val - pre_val);
        map.put(key, val);
    }
    
    public int sum(String prefix) {
        return mytrie.sum(prefix);
    }
}

class Trie {
    private Trie[] children;
    private int pre_cnt;

    /** Initialize your data structure here. */
    public Trie() {
        children = new Trie[26];
        pre_cnt = 0;
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word, int val) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                node.children[idx] = new Trie();
            }
            node = node.children[idx];
            node.pre_cnt += val;
        }
    }

    // 返回 prefix 最后一个结点的cnt
    public int sum(String prefix) {
        Trie node = this;
        for (int i = 0; i < prefix.length(); i++) {
            char ch = prefix.charAt(i);
            int idx = ch - 'a';
            if (node.children[idx] == null) {
                return 0;
            }
            node = node.children[idx];
        }
        return node.pre_cnt;
    }
}
```



## [剑指 Offer II 067. 最大的异或](https://leetcode-cn.com/problems/ms70jA/)

```java
class Solution {
    final int HIGH_BIT = 30;
    BiTrie mytrie;

    public int findMaximumXOR(int[] nums) {
        mytrie = new BiTrie();
        int maximum = 0;
        for (int num : nums) {
            insert(num);
            maximum = Math.max(maximum, maxXOR(num));
        }
        return maximum;
    }

    void insert(int num) {
        BiTrie node = mytrie;
        for (int k = HIGH_BIT; k >= 0; k--) {
            int digit = (num >> k) & 1;
            if (digit == 0 && node.zero == null) {
                node.zero = new BiTrie();
            }
            if (digit == 1 && node.one == null) {
                node.one = new BiTrie();
            }
            if (digit == 0) {
                node = node.zero;
            }else {
                node = node.one;
            }
        }
    }

    int maxXOR(int num) {
        int max = 0;
        BiTrie node = mytrie;
        for (int k = HIGH_BIT; k >= 0; k--) {
            int digit = (num >> k) & 1;
            if (digit == 0) {
                // 找1
                if (node.one != null) {
                    max += (1 << k);
                    node = node.one;
                }else {
                    node = node.zero;
                }
            }else {
                // 找0
                if (node.zero != null) {
                    max += (1 << k);
                    node = node.zero;
                }else {
                    node = node.one;
                }
            }
        }
        return max;
    }
}

class BiTrie {
    BiTrie zero = null;
    BiTrie one = null;
}
```



## [剑指 Offer II 068. 查找插入位置](https://leetcode-cn.com/problems/N6YdxV/)

```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int i = 0, j = nums.length;
        int mid;
        while (i < j) {
            mid = (i + j) / 2;
            if (nums[mid] == target) {
                return mid;
            }else if (nums[mid] < target) {
                i = mid + 1;
            }else {
                j = mid;
            }
        }
        return i;
    }
}
```



## [剑指 Offer II 069. 山峰数组的顶部](https://leetcode-cn.com/problems/B1IidL/)

```java
class Solution {
    public int peakIndexInMountainArray(int[] arr) {
        int i = 0;
        int j = arr.length - 1;
        while (i < j) {
            int mid = (i + j) / 2;
            //System.out.println(arr[mid]);
            if (arr[mid] < arr[mid - 1]) {
                j = mid;
            }else if (arr[mid] > arr[mid + 1]) {
                return mid;
            }else {
                i = mid + 1;
            }
        }
        return -1;
    }
}
```



## [剑指 Offer II 070. 排序数组中只出现一次的数字](https://leetcode-cn.com/problems/skFtm2/)

+ 异或O(n)

```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        int ans = 0;
        for (int num : nums) {
            ans = ans ^ num;
        }
        return ans;
    }
}
```



+ 二分
+ 包含单个出现元素的长度为奇数

```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        int i = 0, j = nums.length - 1;
        while (i < j) {
            int mid = (i + j) / 2;
            // i - j 至少有三个数，所以mid必然左右都有数，不会越界
            if (nums[mid] == nums[mid - 1]) {
                // i -- mid-2, mid+1 -- j
                if (((mid - i - 1) & 1) == 1) {
                    j = mid - 2;
                }else {
                    i = mid + 1;
                }
            }else if (nums[mid] == nums[mid + 1]) {
                // i -- mid-1, mid+2 -- j
                if (((mid - i) & 1) == 1) {
                    j = mid - 1;
                }else {
                    i = mid + 2;
                }
            }else {
                return nums[mid];
            }
        }
        return nums[i];
    }
}
```





## [剑指 Offer II 071. 按权重生成随机数](https://leetcode-cn.com/problems/cuyjEf/)

+ 前缀和
+ 二分查找

```java
class Solution {
    Random rand = new Random();
    int[] presum;
    int sum;

    public Solution(int[] w) {
        presum = new int[w.length];
        sum = 0;
        for (int i = 0; i < w.length; i++) {
            sum += w[i];
            presum[i] = sum;
        }
    }
    
    public int pickIndex() {
        int r = rand.nextInt(sum) + 1;
        return bisect(r);
    }

    int bisect(int num) {
        int lo = 0;
        int hi = presum.length - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (num <= presum[mid]) {
                hi = mid;
            }else {
                lo = mid + 1;
            }
        }
        return lo;
    }
}
```





## [剑指 Offer II 072. 求平方根](https://leetcode-cn.com/problems/jJ0w9p/)

```java
class Solution {
    public int mySqrt(int x) {
        if (x <= 1) return x;
        int lo = 1;
        int hi = Math.min(x, 46340);
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2 + 1;
            int mul = mid * mid;
            if (mul == x) {
                return mid;
            }else if (mul > x) {
                hi = mid - 1;
            }else {
                lo = mid;
            }
        }
        return lo;
    }
}
```



## [剑指 Offer II 073. 狒狒吃香蕉](https://leetcode-cn.com/problems/nZZqjQ/)

+ 二分
+ O(N)判断是否能吃完

```java
class Solution {
    public int minEatingSpeed(int[] piles, int h) {
        int lo = 1;
        int hi = 1000000000;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (valid(mid, h, piles)) {
                hi = mid;
            }else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private boolean valid(int k, int h, int[] piles) {
        int cnt = 0;
        for (int pile : piles) {
            cnt += (pile - 1) / k + 1;
        }
        return cnt <= h;
    }
}
```



## [剑指 Offer II 074. 合并区间](https://leetcode-cn.com/problems/SsGoHC/)

+ 排序后逐个检查是否相邻

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] l1, int[] l2) {
                return l1[0] - l2[0];
            }
        });
        List<int[]> ans = new ArrayList<>();
        for (int i = 0; i < intervals.length; i++) {
            int l = intervals[i][0];
            int r = intervals[i][1];
            if (ans.isEmpty() || l > ans.get(ans.size() - 1)[1]) {
                ans.add(intervals[i]);
            }else {
                ans.get(ans.size() - 1)[1] = Math.max(r, ans.get(ans.size() - 1)[1]);
            }
        }
        return ans.toArray(new int[0][]);
    }
}
```



```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals = sorted(intervals, key=lambda x : x[0])
        ans = []
        for l, r in intervals:
            if len(ans) == 0 or l > ans[-1][1]:
                ans.append([l, r])
            else:
                ans[-1][1] = max(ans[-1][1], r)
        return ans
```



## [剑指 Offer II 075. 数组相对排序](https://leetcode-cn.com/problems/0H97ZC/)

+ 自定义排序
+ 注意java中Integer 不能对int 进行排序

```java
import java.util.Comparator;

class Solution {
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        int n = arr2.length;
        Map<Integer, Integer> order = new HashMap<>();
        for (int i = 0; i < n; i++) {
            order.put(arr2[i], i);
        }
        List<Integer> arr = new ArrayList<>();
        for (int num : arr1) {
            arr.add(num);
        }
        Comparator<Integer> comp = new Comparator<>() {
            @Override
            public int compare(Integer x1, Integer x2) {
                int o1 = order.getOrDefault(x1, n + x1);
                int o2 = order.getOrDefault(x2, n + x2);
                return o1 - o2;
            }
        };
        Collections.sort(arr, comp);
        for (int i = 0; i < arr1.length; i++) {
            arr1[i] = arr.get(i);
        }
        return arr1;
    }
}
```



```python
from functools import cmp_to_key

class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        def comp(x1, x2):
            o1 = order.get(x1, x1 + n)
            o2 = order.get(x2, x2 + n)
            return o1 - o2

        n = len(arr2)
        order = dict()
        for i, num in enumerate(arr2):
            order[num] = i
        arr1 = sorted(arr1, key=cmp_to_key(comp))
        return arr1
```



## [剑指 Offer II 076. 数组中的第 k 大的数字](https://leetcode-cn.com/problems/xx4gT2/)

+ 小根堆

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(k * 2);
        for (int num : nums) {
            if (pq.size() < k) {
                pq.add(num);
            }else if (num > pq.peek()) {
                pq.poll();
                pq.add(num);
            }
        }
        return pq.peek();
    }
}
```

+ 快排思想


```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        qsort(nums, 0, n - 1, k);
        return nums[n - k];

    }

    void qsort(int[] nums, int l, int r, int k) {
        if (r <= l) return;
        int i = l, j = r;
        int temp = nums[l];
        while (i < j) {
            while (i < j && nums[j] >= temp) {
                j -= 1;
            }
            nums[i] = nums[j];
            while (i < j && nums[i] <= temp) {
                i += 1;
            }
            nums[j] = nums[i];
        }
        nums[i] = temp;
        if (r - i + 1 == k) {
            return;
        }else if (r - i + 1 < k) {
            qsort(nums, l, i - 1, k - r + i - 1);
        }else {
            qsort(nums, i + 1, r, k);
        }
    }
}
```


+ 快排，随机取pivot
+ 可以防止极端用例速度过慢

```java
class Solution {
    Random rand = new Random();

    public int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        qsort(nums, 0, n - 1, k);
        return nums[n - k];

    }

    void qsort(int[] nums, int l, int r, int k) {
        if (r <= l) return;
        int i = l, j = r;
        // 随机将 l 处的元素与范围中的一个元素交换
        int randidx = l + rand.nextInt(r - l + 1);
        swap(nums, l, randidx);
        int temp = nums[l];
        while (i < j) {
            while (i < j && nums[j] >= temp) {
                j -= 1;
            }
            nums[i] = nums[j];
            while (i < j && nums[i] <= temp) {
                i += 1;
            }
            nums[j] = nums[i];
        }
        nums[i] = temp;
        if (r - i + 1 == k) {
            return;
        }else if (r - i + 1 < k) {
            qsort(nums, l, i - 1, k - r + i - 1);
        }else {
            qsort(nums, i + 1, r, k);
        }
    }

    void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```





## [剑指 Offer II 078. 合并排序链表](https://leetcode-cn.com/problems/vvXgSW/)

+ 小根堆 + 多路归并

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode n1, ListNode n2) {
                return n1.val - n2.val;
            }
        });
        for (ListNode node : lists) {
            if (node != null) pq.add(node);
        }
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;

        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            cur.next = node;
            cur = cur.next;
            if (node.next != null) pq.add(node.next);
        } 
        return dummy.next;
    }
}
```



+ python中注意要给ListNode 类写一个\_\_le\_\_ 方法

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # le方法
        def __lt__(self, other):
            return self.val < other.val
        ListNode.__lt__ = __lt__
        pq = []
        for node in lists:
            if node is not None:
                heapq.heappush(pq, node)
        dummy = ListNode(-1)
        cur = dummy

        while pq:
            node = heapq.heappop(pq)
            cur.next = node
            cur = cur.next
            if node.next is not None:
                heapq.heappush(pq, node.next) 
        return dummy.next
```





## [剑指 Offer II 079. 所有子集](https://leetcode-cn.com/problems/TVdhkn/)

+ 无重复元素
+ dfs回溯，要求元素下标要递增
+ 不需要visited

```java
class Solution {
    List<List<Integer>> ans;
    List<Integer> cur;

    public List<List<Integer>> subsets(int[] nums) {
        ans = new ArrayList<>();
        cur = new ArrayList<>();
        dfs_backtrack(nums, -1);
        return ans;
    }

    void dfs_backtrack(int[] nums, int last_idx) {
        ans.add(new ArrayList<Integer>(cur));
        for (int i = last_idx + 1; i < nums.length; i++) {
            cur.add(nums[i]);
            dfs_backtrack(nums, i);
            cur.remove(cur.size() - 1);
        }
    }
}
```



```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        self.ans = []
        self.cur = []
        self.dfs_backtrack(nums, -1)
        return self.ans 

    def dfs_backtrack(self, nums, last_idx):
        self.ans.append(self.cur.copy())
        for i in range(last_idx + 1, len(nums)):
            self.cur.append(nums[i])
            self.dfs_backtrack(nums, i)
            self.cur.pop()
```



## [剑指 Offer II 080. 含有 k 个元素的组合](https://leetcode-cn.com/problems/uUsW3B/)

+ dfs回溯
+ 要求数字递增
+ 长度为k时停止
+ 可以加入剪枝

```java
class Solution {
    List<List<Integer>> ans;
    List<Integer> track;

    public List<List<Integer>> combine(int n, int k) {
        ans = new ArrayList<>();
        track = new ArrayList<>();
        dfs_backtrack(n, k, 0);
        return ans;
    }

    void dfs_backtrack(int n, int k, int last) {
        if (track.size() == k) {
            ans.add(new ArrayList<>(track));
            return;
        }
        // 剪枝
        if (n - last < k - track.size()) {
            return;
        }
        for (int next = last + 1; next <= n; next++) {
            track.add(next);
            dfs_backtrack(n, k, next);
            track.remove(track.size() - 1);
        }
    }
}
```



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



## [剑指 Offer II 081. 允许重复选择元素的组合](https://leetcode-cn.com/problems/Ygoe9J/)

+ dfs回溯
+ sum超出时停止
+ 控制track中元素递增来保证唯一性

```java
class Solution {
    List<List<Integer>> ans;
    List<Integer> track;
    int[] candidates;
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        ans = new ArrayList<>();
        track = new ArrayList<>();
        this.candidates = candidates;
        dfs_backtrack(0, target, -1);
        return ans;
    }

    void dfs_backtrack(int sum, int target, int last) {
        if (sum > target) return;
        if (sum == target) {
            ans.add(new ArrayList<>(track));
            return;
        }
        for (int next : candidates) {
            if (next < last) continue;
            track.add(next);
            dfs_backtrack(sum + next, target, next);
            track.remove(track.size() - 1);
        }
    }
}
```



+ 先排序，可以更有效率的剪枝

```java
class Solution {
    List<List<Integer>> ans;
    List<Integer> track;
    int[] candidates;
    
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        ans = new ArrayList<>();
        track = new ArrayList<>();
        Arrays.sort(candidates);
        this.candidates = candidates;
        dfs_backtrack(0, target, 0);
        return ans;
    }

    void dfs_backtrack(int sum, int target, int last_idx) {
        if (sum == target) {
            ans.add(new ArrayList<>(track));
            return;
        }
        for (int idx = last_idx; idx < candidates.length; idx++) {
            int val = candidates[idx];
            if (sum + val > target) break;
            track.add(val);
            dfs_backtrack(sum + val, target, idx);
            track.remove(track.size() - 1);
        }
    }
}
```



```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.track = []
        self.ans = []
        self.candidates = sorted(candidates)
        self.dfs_backtrack(0, target, 0)
        return self.ans 

    def dfs_backtrack(self, sum, target, last_idx):
        if sum == target:
            self.ans.append(self.track.copy())
            return 
        # 剪枝
        for idx in range(last_idx, len(self.candidates)):
            val = self.candidates[idx]
            if (val + sum > target):
                break 
            self.track.append(val)
            self.dfs_backtrack(sum + val, target, idx)
            self.track.pop()
        return 
```



## [剑指 Offer II 082. 含有重复元素集合的组合](https://leetcode-cn.com/problems/4sjJUc/)

+ 会有重复元素，且每个元素只能选择一次
+ map记录每个元素的剩余可用次数

+ 寻找下一个数字时要遍历dict.keys()  即无重复数组

```java
class Solution {
    List<List<Integer>> ans;
    List<Integer> track;
    Map<Integer, Integer> cnt;
    int[] unicand;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        ans = new ArrayList<>();
        track = new ArrayList<>();
        cnt = new HashMap<>();

        for (int val : candidates) {
            cnt.put(val, cnt.getOrDefault(val, 0) + 1);
        }

        unicand = new int[cnt.size()];
        int i = 0;
        for (int cand : cnt.keySet()) {
            unicand[i++] = cand;
        }
        Arrays.sort(unicand);

        dfs_backtrack(0, target, 0);
        return ans;
    }

    void dfs_backtrack(int sum, int target, int last_idx) {
        if (sum == target) {
            ans.add(new ArrayList<>(track));
            return;
        }
        for (int idx = last_idx; idx < unicand.length; idx++) {
            int val = unicand[idx];
            int rest = cnt.get(val);
            if (rest == 0) continue;
            if (sum + val > target) break;
            track.add(val);
            cnt.put(val, rest - 1);
            dfs_backtrack(sum + val, target, idx);
            track.remove(track.size() - 1);
            cnt.put(val, rest);
        }
    }
}
```



## [剑指 Offer II 083. 没有重复元素集合的全排列](https://leetcode-cn.com/problems/VvJkup/)

+ 无重复元素的全排列
+ dfs_backtrack
+ 需要visited记录

```java
class Solution {
    List<List<Integer>> ans;
    List<Integer> track;
    boolean[] visited;

    public List<List<Integer>> permute(int[] nums) {
        ans = new ArrayList<>();
        track = new ArrayList<>();
        int n = nums.length;
        visited = new boolean[n];
        dfs_backtrack(nums, n);
        return ans;
    }

    private void dfs_backtrack(int[] nums, int n) {
        if (track.size() == n) {
            ans.add(new ArrayList<Integer>(track));
            return;
        }
        for (int i = 0; i < n; i++) {
            if (visited[i] == false) {
                track.add(nums[i]);
                visited[i] = true;
                dfs_backtrack(nums, n);
                track.remove(track.size() - 1);
                visited[i] = false;
            }
        }
    }
}
```



```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.res = []
        self.visited = set()
        self.track = []
        self.backtrack(nums)
        return self.res 
    
    def backtrack(self, nums):
        if len(self.track) == len(nums):
            self.res.append(self.track.copy())
            return 
        for num in nums:
            if num not in self.visited:
                self.track.append(num)
                self.visited.add(num)
                self.backtrack(nums)
                self.track.pop()
                self.visited.remove(num)
```



## [剑指 Offer II 084. 含有重复元素集合的全排列 ](https://leetcode-cn.com/problems/7p8L0Z/)

+ 有重复元素的全排列
+ dfs_backtrack
+ map记录个数, 每次遍历的时候遍历map的key 即不重复的元素

```java
class Solution {
    List<List<Integer>> ans;
    List<Integer> track;
    Map<Integer, Integer> cnt;
    Set<Integer> unival;

    public List<List<Integer>> permuteUnique(int[] nums) {
        ans = new ArrayList<>();
        track = new ArrayList<>();
        cnt = new HashMap<>();
        int n = nums.length;
        for (int num : nums) {
            cnt.put(num, cnt.getOrDefault(num, 0) + 1);
        }
        unival = cnt.keySet();
        dfs_backtrack(nums, n);
        return ans;
    }

    private void dfs_backtrack(int[] nums, int n) {
        if (track.size() == n) {
            ans.add(new ArrayList<Integer>(track));
            return;
        }
        for (int num : unival) {
            if (cnt.get(num) > 0) {
                track.add(num);
                cnt.put(num, cnt.get(num) - 1);
                dfs_backtrack(nums, n);
                track.remove(track.size() - 1);
                cnt.put(num, cnt.get(num) + 1);
            }
        }
    }
}
```



```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        self.res = []
        self.cnt = collections.defaultdict(int)
        for num in nums:
            self.cnt[num] += 1
        self.track = []
        self.backtrack(len(nums))
        return self.res 
        
    def backtrack(self, n):
        if len(self.track) == n:
            self.res.append(self.track.copy())
            return 
        for num in self.cnt:
            if self.cnt[num] > 0:
                self.track.append(num)
                self.cnt[num] -= 1
                self.backtrack(n)
                self.track.pop()
                self.cnt[num] += 1
```



## [剑指 Offer II 085. 生成匹配的括号](https://leetcode-cn.com/problems/IDBivT/)

```java
class Solution {
    List<String> ans;
    StringBuilder track;

    public List<String> generateParenthesis(int n) {
        ans = new ArrayList<>();
        track = new StringBuilder();
        dfs_backtrack(0, 0, n);
        return ans;
    }

    private void dfs_backtrack(int left, int right, int n) {
        if (left < right) return;
        if (right == n) {
            ans.add(track.toString());
            return;
        }
        if (left < n) {
            track.append("(");
            dfs_backtrack(left + 1, right, n);
            track.deleteCharAt(track.length() - 1);
        }
        track.append(")");
        dfs_backtrack(left, right + 1, n);
        track.deleteCharAt(track.length() - 1);        
    }
}
```



```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []
        track = []
        self.dfs_backtrack(0, 0 ,track, ans, n)
        return ans

    def dfs_backtrack(self, left, right, track, ans, n):
        if left < right:
            return
        if right == n:
            ans.append(''.join(track))
            return
        if (left < n):
            track.append('(')
            self.dfs_backtrack(left + 1, right, track, ans, n)
            track.pop()
        if (right < n):
            track.append(')')
            self.dfs_backtrack(left, right + 1, track, ans, n)
            track.pop()
```



## [剑指 Offer II 086. 分割回文子字符串](https://leetcode-cn.com/problems/M99OJA/)

+ 先用动态规划建立isPal， i - j是否是回文串
+ 再dfs_backtrack
+ 起始点为i，找遍历j ，是回文串则作为下一个

```java
class Solution {
    List<String[]> ans;
    List<String> track;
    boolean[][] isPal;

    public String[][] partition(String s) {
        ans = new ArrayList<>();
        track = new ArrayList<>();
        int n = s.length();
        isPal = new boolean[n][n];
        constructPal(s);
        dfs_backtrack(0, n, s);
        
        return ans.toArray(new String[ans.size()][]);
    }

    private void dfs_backtrack(int idx, int n, String s) {
        if (idx == n) {
            ans.add(track.toArray(new String[track.size()]));
            return;
        }
        for (int next = idx; next < n; next++) {
            if (isPal[idx][next]) {
                track.add(s.substring(idx, next + 1));
                dfs_backtrack(next + 1, n , s);
                track.remove(track.size() - 1);
            }
        }
    }   

    private void constructPal(String s) {
        int n = s.length();
        for (int i = 0; i < n; i++) {
            Arrays.fill(isPal[i], true);
        }
        for (int l = 2; l <= n; l++) {
            for (int i = 0; i + l - 1 < n; i++) {
                int j = i + l - 1;
                isPal[i][j] = (s.charAt(i) == s.charAt(j)) && isPal[i + 1][j - 1];
            }
        }
    }

}
```



## [剑指 Offer II 087. 复原 IP ](https://leetcode-cn.com/problems/0on3uN/)

+ dfs_backtrack
+ 判断是否再 0 - 255之间
+ 注意java中track生成字符串
+ 注意java中用了一个数组，不需要在回溯时改变最后位置的元素
+ dfs会自动修改之前的元素

```java
class Solution {
    List<String> ans;
    String[] track;

    public List<String> restoreIpAddresses(String s) {
        ans = new ArrayList<>();
        track = new String[4];
        dfs_backtrack(0, 0, s.length(), s);
        return ans;
    }

    private void dfs_backtrack(int len, int idx, int n, String s) {
        if (len == 4) {
            if (idx == n) {
                ans.add(String.join(".", track));
            }
            return;
        }
        int remain = n - idx; // 2
        int need = 4 - len; // 1
        if (need > remain || need * 3 < remain) return;
        for (int next_idx = idx + 1; next_idx <= Math.min(idx + 3, n); next_idx++) {

            String code = s.substring(idx, next_idx);
            if (inRange(code)) {
                track[len] = code;
                dfs_backtrack(len + 1, next_idx, n, s);
            }
        }
    }

    private boolean inRange(String num) {
        int n = num.length();
        if (n == 1) return true;
        if (num.charAt(0) == '0') return false;
        if (n == 2) return true;
        return num.compareTo("255") <= 0;
    }
}
```



```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        ans = []
        track = []
        self.dfs_backtrack(0, 0, track, ans, len(s), s)
        return ans 

    def dfs_backtrack(self, len, idx, track, ans, n, s):
        if len == 4:
            if idx == n:
                ans.append('.'.join(track))
            return
        for next_idx in range(idx + 1, min(n, idx + 3) + 1):
            code = s[idx: next_idx]
            if (self.inRange(code)):
                track.append(code)
                self.dfs_backtrack(len + 1, next_idx, track, ans, n, s)
                track.pop()

    def inRange(self, s):
        n = len(s)
        if n == 1:
            return True
        if s[0] == '0':
            return False
        if n == 2:
            return True
        return s <= '255'
```



## [剑指 Offer II 088. 爬楼梯的最少成本](https://leetcode-cn.com/problems/GzCJIP/)

+ dp
+ dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

```java
class Solution {
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        if (n <= 1) return 0;
        int[] dp = new int[n + 1];
        for (int i = 2; i < n + 1; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
    }
}
```



## [剑指 Offer II 089. 房屋偷盗](https://leetcode-cn.com/problems/Gu0c2T/)

+ dp[i]表示前 i 间房屋能偷窃到的最高总金额
+ dp[i] = Math.max(dp[i - 1], nums[i - 1] + dp[i - 2])

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        dp[1] = nums[0];
        for (int i = 2; i < n + 1; i++) {
            dp[i] = Math.max(dp[i - 1], nums[i - 1] + dp[i - 2]);
        }
        return dp[n];
    }
}
```

+ 只依赖前两个，可以省去数组的空间开销

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        int pre = 0, cur = nums[0];
        int temp;
        for (int i = 1; i < n; i++) {
            temp = Math.max(cur, pre + nums[i]);
            pre = cur;
            cur = temp;
        }
        return cur;
    }
}
```



```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        pre, cur = 0, nums[0]
        for i in range(1, len(nums)):
            pre, cur = cur, max(cur, pre + nums[i])
        return cur
```



## [剑指 Offer II 090. 环形房屋偷盗](https://leetcode-cn.com/problems/PzWKhm/)

+ 分成偷不偷最后一个房子
+ 分别是两个无环问题
+ 封装一个helper函数

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }
        return Math.max(helper(0, n - 1, nums), helper(1, n, nums));
    }

    private int helper(int start, int end, int[] nums) {
        int pre = 0, cur = nums[start];
        int temp;
        for (int i = start + 1; i < end; i++) {
            temp = Math.max(cur, pre + nums[i]);
            pre = cur;
            cur = temp;
        }
        return cur;
    }
}
```



```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        return max(self.helper(0, n - 1, nums), self.helper(1, n, nums))

    def helper(self, start, end, nums):
        pre, cur = 0, nums[start]
        for i in range(start + 1, end):
            pre, cur = cur, max(cur, pre + nums[i])
        return cur
```



## [剑指 Offer II 091. 粉刷房子](https://leetcode-cn.com/problems/JEj789/)

+ 当前房子有三种情况，根据前一间房子的情况计算出三种粉刷的最小花费
+ newred = Math.min(blue, green) + r
+ newblue = Math.min(red, green) + b
+ newgreen = Math.min(red, blue) + g

```java
class Solution {
    public int minCost(int[][] costs) {
        int red = costs[0][0];
        int blue = costs[0][1];
        int green = costs[0][2];
        int newred, newblue, newgreen, r, b, g;
        for (int i = 1; i < costs.length; i++) {
            r = costs[i][0];
            b = costs[i][1];
            g = costs[i][2];
            newred = Math.min(blue, green) + r;
            newblue = Math.min(red, green) + b;
            newgreen = Math.min(red, blue) + g;
            red = newred; blue = newblue; green = newgreen;
        }
        return Math.min(Math.min(red, blue), green);
    }
}
```



```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        red, green, blue = costs[0][0], costs[0][1], costs[0][2]
        for i in range(1, len(costs)):
            r, g, b = costs[i][0], costs[i][1], costs[i][2]
            red, green, blue = min(green, blue) + r, min(red, blue) + g, min(green, red) + b
        return min(red, green, blue)
```



## [剑指 Offer II 092. 翻转字符](https://leetcode-cn.com/problems/cyJERH/)

+ dp
+ 记录前一个位置以0结尾的最小翻转次数，和以1结尾的最小翻转次数
+ 根据当前位置更新

```java
class Solution {
    public int minFlipsMonoIncr(String s) {
        int zero = 0;
        int one = 1;
        if (s.charAt(0) == '1') {
            zero = 1;
            one = 0;
        }
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == '0') {
                one = 1 + Math.min(zero, one);
            }else {
                int tmpzero = 1 + zero;
                one = Math.min(zero, one);
                zero = tmpzero;
            }
        }
        return Math.min(zero, one);
    }
}
```



```python
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        zero, one = 0, 1
        if s[0] == '1':
            zero, one = 1, 0
        for i in range(1, len(s)):
            if s[i] == '0':
                one = 1 + min(zero, one)
            else:
                zero, one = 1 + zero, min(zero, one)
        return min(zero, one)
```



+ 记录前缀和
+ 遍历每一个位置，以该位置为最后一个0
+ 计算左边的1的个数和右边的0的个数，加起来就是翻转次数
+ 注意可以全是1，presume要多留一个位置

```java
class Solution {
    public int minFlipsMonoIncr(String s) {
        int n = s.length();
        int[] presum = new int[n + 1];
        for (int i = 1; i < n + 1; i++) {
            presum[i] = presum[i - 1];
            if (s.charAt(i - 1) == '1') {
                presum[i] += 1;
            }
        }
        int min = n + 1;
        for (int i = 0; i < n + 1; i++) {
            int leftone = presum[i];
            int rightzero = n - i - (presum[n] - presum[i]);
            min = Math.min(min, leftone + rightzero);
        }
        return min;
    }
}
```



## [剑指 Offer II 093. 最长斐波那契数列](https://leetcode-cn.com/problems/Q91FMA/)

+ dp\[i\]\[j\] ：以arr i 和 arr j 为最后两个元素的最长 *斐波那契式*  长度
+ hashmap计算上一个数的下标，z， 若存在，则dp\[i\]\[j\] = dp\[z\]\[i\] + 1
+ 不存在则设置为2，即为初始化 斐波那契

```java
class Solution {
    public int lenLongestFibSubseq(int[] arr) {
        Map<Integer, Integer> val2idx = new HashMap<>();
        int n = arr.length;
        int[][] dp = new int[n][n];
        int longest = 2;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                int pre_val = arr[j] - arr[i];
                int pre_idx = val2idx.getOrDefault(pre_val, -1);
                if (pre_idx != -1) {
                    dp[i][j] = 1 + dp[pre_idx][i];
                }else {
                    dp[i][j] = 2;
                }
                longest = Math.max(longest, dp[i][j]);
            }
            // 将i 处加入hashmap，可以被之后的循环查找
            val2idx.put(arr[i], i);
        }

        if (longest < 3) return 0;
        return longest;
    }
}
```









## [剑指 Offer II 094. 最少回文分割](https://leetcode-cn.com/problems/omKAoA/)

+ 动态规划简历isPal二维数组，判断两下标之间是否是回



![image-20220125203837575](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220125203837575.png)

```java
class Solution {
    boolean[][] isPal;

    public int minCut(String s) {
        int n = s.length();
        isPal = new boolean[n][n];
        constructPal(s);
        int[] dp = new int[n];
        for (int i = 1; i < n; i++) {
            if (isPal[0][i]) {
                dp[i] = 0;
                continue;
            }
            int min = i + 1;
            for (int st = 1; st <= i; st++) {
                if (isPal[st][i]) {
                    min = Math.min(min, dp[st - 1] + 1);
                }
            }
            dp[i] = min;
        }
        return dp[n - 1];

    }

    private void constructPal(String s) {
        int n = s.length();
        for (int i = 0; i < n; i++) {
            Arrays.fill(isPal[i], true);
        }
        for (int l = 2; l <= n; l++) {
            for (int i = 0; i + l - 1 < n; i++) {
                int j = i + l - 1;
                isPal[i][j] = (s.charAt(i) == s.charAt(j)) && isPal[i + 1][j - 1];
            }
        }
    }
}
```



## [剑指 Offer II 095. 最长公共子序列](https://leetcode-cn.com/problems/qJnOS7/)

![image-20220126112723390](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220126112723390.png)

```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i < m + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
}
```



```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]
```



## [剑指 Offer II 096. 字符串交织](https://leetcode-cn.com/problems/IY6buf/)

![image-20220126120451255](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220126120451255.png)

+ 注意i j 等于0 时，代码的技巧

```java
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length(), n = s2.length();
        if (m + n != s3.length()) return false;
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i < m + 1; i++) {
            for (int j = 0; j < n + 1; j++) {
                if (i > 0) {
                    dp[i][j] = s1.charAt(i - 1) == s3.charAt(i + j - 1) && dp[i - 1][j];
                }
                if (j > 0) {
                    dp[i][j] = dp[i][j] || s2.charAt(j - 1) == s3.charAt(i + j - 1) && dp[i][j - 1];
                }
            }
        }
        return dp[m][n];
    }
}
```



## [剑指 Offer II 097. 子序列的数目](https://leetcode-cn.com/problems/21dk04/)

+ `dp[i][j]`表示 s前i子串中有多少个t前j子串
+ 注意初始化，`dp[i][0] = 1`
+ `dp[i][j] = dp[i-1][j]`， 若`s[i-1] == t[j - 1]` , 则还要加上`dp[i-1][j-1]`
+ `j <= i` 时合理，否则直接是0

```java
class Solution {
    public int numDistinct(String s, String t) {
        int n1 = s.length();
        int n2 = t.length();
        if (n1 < n2) return 0;
        int[][] dp = new int[n1 + 1][n2 + 1];
        for (int i = 0; i < n1 + 1; i++) {
            dp[i][0] = 1;
        }

        for (int i = 1; i < n1 + 1; i++) {
            for (int j = 1; j <= Math.min(i, n2); j++) {
                dp[i][j] = dp[i - 1][j];
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] += dp[i - 1][j - 1];
                }
            }
        }
        return dp[n1][n2];
    }
}
```



```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        n1 = len(s)
        n2 = len(t)
        if n1 < n2:
            return 0
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(n1 + 1):
            dp[i][0] = 1
        for i in range(1, n1 + 1):
            for j in range(1, min(i, n2) + 1):
                dp[i][j] = dp[i - 1][j]
                if (s[i - 1] == t[j - 1]):
                    dp[i][j] += dp[i - 1][j - 1]
        return dp[-1][-1]
```



## [剑指 Offer II 098. 路径的数目](https://leetcode-cn.com/problems/2AoeFn/)

+ 动态规划

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        dp[0][1] = 1;
        for (int i = 1; i < m + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m][n];
    }
}
```



+ 组合数，一共要走`m + n - 2` 步，其中`m - 1` 步向下
+ `C(m + n - 2, m - 1)`
+ 不能直接用阶乘算，会溢出

```java
class Solution {
    public int uniquePaths(int m, int n) {
        return comb(m + n - 2, Math.min(m, n) - 1);
    }

    int comb(int N, int k) {
        long ans = 1;
        for (int u = N - k + 1, d = 1; d <= k; u++, d++) {
            ans = ans * u / d;
        }
        return (int)ans;
    }
}
```



## [剑指 Offer II 099. 最小路径之和](https://leetcode-cn.com/problems/0i0mDW/)

+ dp
+ 可以在原数组上dp，节省空间

```java
class Solution {
    
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int j = 1; j < n; j++) {
            grid[0][j] += grid[0][j - 1];
        }
        for (int i = 1; i < m; i++) {
            grid[i][0] += grid[i - 1][0];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[m - 1][n - 1];
    }
}
```



## [剑指 Offer II 100. 三角形中最小路径之和](https://leetcode-cn.com/problems/IlPe0q/)

+ 自顶向下dp
+ 原数组上直接修改

```java
class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        for (int i = 1; i < n; i++) {
            List<Integer> cur = triangle.get(i);
            List<Integer> up = triangle.get(i - 1);
            for (int j = 0; j < cur.size(); j++) {
                if (j == 0) {
                    cur.set(j, cur.get(j) + up.get(j));
                }else if (j == cur.size() - 1) {
                    cur.set(j, cur.get(j) + up.get(j - 1));
                }else {
                    cur.set(j, cur.get(j) + Math.min(up.get(j), up.get(j - 1)));
                }
            }
        }
        int min = Integer.MAX_VALUE;
        for (int val : triangle.get(n - 1)) {
            min = Math.min(min, val);
        }
        return min;
    }
}
```



+ 自底向上dp

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        for i in range(n - 2, -1, -1):
            for j in range(0, len(triangle[i])):
                triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])
        return triangle[0][0]
```



## [剑指 Offer II 101. 分割等和子集](https://leetcode-cn.com/problems/NUPfPr/)

+ 0 - 1背包
+ 先计算sum，不是偶数则false
+ 之后需要在nums中找和为 `sum / 2` 的子序列
+ 以`sum / 2` 为背包size，看最大的和是否为`sum / 2` 
+ 要注意数组的交换

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        int max = -1;
        int n = nums.length;
        for (int num : nums) {
            sum += num;
            max = Math.max(max, num);
        }
        int target = sum >> 1;
        if ((sum & 1) == 1 || target < max) return false;
        if (target == max) return true;
        // 初始化第一个元素
        int[] pre = new int[target + 1];
        for (int j = 0; j < target + 1; j++) {
            pre[j] = nums[0] <= j ? nums[0] : 0;
        }
        int[] cur = new int[target + 1];
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < target + 1; j++) {
                cur[j] = pre[j];
                if (j >= nums[i]) {
                    cur[j] = Math.max(cur[j], pre[j - nums[i]] + nums[i]);
                }
            }
            int[] tmp = pre;
            pre = cur;
            cur = tmp;
        }
        return pre[target] == target;
    }
}
```



+ 其实应该用boolean数组
+ `dp[i][j]` 表示用前i个数字是否能正好凑出 sum j

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        sum_ = sum(nums)
        max_ = max(nums)
        n = len(nums)
        target = sum_ >> 1
        if (sum_ & 1) == 1 or target < max_:
            return False
        if target == max_:
            return True
        
        pre = [False] * (target + 1)
        cur = [False] * (target + 1)
        pre[0] = True
        for num in nums:
            for j in range(target + 1):
                if j < num:
                    cur[j] = pre[j]
                else:
                    cur[j] = pre[j] or pre[j - num]
            cur, pre = pre, cur  
        return pre[-1]
```



## [剑指 Offer II 102. 加减的目标值](https://leetcode-cn.com/problems/YaVDxD/)

![image-20220128175414482](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220128175414482.png)

+ 计算sum，选取若干元素总和neg
+ 转换为背包dp问题
+ `dp[i][j]`表示前 i 个数中选取元素，使得这些元素之和等于 j 的方案数

```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        int diff = sum - target;
        if (diff < 0) return 0;
        if ((diff & 1) == 1) return 0;
        int negsum = (sum - target) / 2;
        int n = nums.length;
        int[][] dp = new int[n + 1][negsum + 1];
        dp[0][0] = 1;

        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j < negsum + 1; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= nums[i - 1]) {
                    dp[i][j] += dp[i - 1][j - nums[i - 1]];
                }
            }
        }
        return dp[n][negsum];
    }
}
```



+ 省一些空间

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        sum_ = sum(nums)
        diff = sum_ - target
        if diff < 0 or diff % 2 == 1:
            return 0 
        negsum = diff // 2
        n = len(nums)
        pre = [0] * (negsum + 1)
        pre[0] = 1
        cur = [0] * (negsum + 1)
        for i in range(n):
            num = nums[i]
            for j in range(negsum + 1):
                cur[j] = pre[j]
                if j >= num:
                    cur[j] += pre[j - num]
            pre, cur = cur, pre 
        return pre[-1]
```



## [剑指 Offer II 103. 最少的硬币数目](https://leetcode-cn.com/problems/gaM7Ch/)

+ 完全背包dp
+ `dp[i][j]`表示前i个coin，组成金额j的最小个数
+ `dp[i][j] = min(dp[i - 1][j], dp[i][j - nums[i]] + 1)`
+ 初始条件 `dp[0][0] = 1, dp[0][j] = -1`

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[][] dp = new int[n + 1][amount + 1];
        Arrays.fill(dp[0], -1);
        dp[0][0] = 0;
        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j < amount + 1; j++) {
                if (j - coins[i - 1] < 0) {
                    dp[i][j] = dp[i - 1][j];
                    continue;
                }
                int without = dp[i - 1][j]; 
                int with = dp[i][j - coins[i - 1]];
                if (without == -1 && with == -1) {
                    dp[i][j] = -1;
                }else if (without == -1) {
                    dp[i][j] = with + 1;
                }else if (with == -1) {
                    dp[i][j] = without;
                }else {
                    dp[i][j] = Math.min(without, with + 1);
                }
            }
        }
        return dp[n][amount];
    }
}
```



+ 省空间

```java
class Solution {
    static final int INF = Integer.MAX_VALUE;

    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, INF);
        dp[0] = 0;
        for (int coin : coins) {
            // System.out.println(Arrays.toString(dp));
            for (int j = coin; j < amount + 1; j++) {
                if (dp[j - coin] != INF) {
                    dp[j] = Math.min(dp[j], dp[j - coin] + 1);
                }
            }
        }
        return dp[amount] == INF ? -1 : dp[amount];
    }
}
```



## [剑指 Offer II 104. 排列的数目](https://leetcode-cn.com/problems/D0F0SV/)

![image-20220129165325573](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220129165325573.png)

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i < target + 1; i++) {
            for (int num : nums) {
                if (i >= num) {
                    dp[i] += dp[i - num];
                }
            }            
        }
        return dp[target];
    }
}
```



## [剑指 Offer II 105. 岛屿的最大面积](https://leetcode-cn.com/problems/ZL6zAn/)

+ DFS

```java
class Solution {
    int cnt;
    int[][] visited;

    public int maxAreaOfIsland(int[][] grid) {
        int maxarea = 0;
        int m = grid.length, n = grid[0].length;
        visited = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (visited[i][j] == 0 && grid[i][j] == 1) {
                    cnt = 0;
                    dfs(grid, i, j, m, n);
                    maxarea = Math.max(maxarea, cnt);
                }
            }
        }
        return maxarea;
    }

    private void dfs(int[][] grid, int i, int j, int m, int n) {
        if (i < 0 || j < 0 || i >= m || j >= n) return;
        if (visited[i][j] == 1 || grid[i][j] == 0) return;
        visited[i][j] = 1;
        cnt += 1;
        dfs(grid, i - 1, j, m, n);
        dfs(grid, i + 1, j, m, n);
        dfs(grid, i, j - 1, m, n);
        dfs(grid, i, j + 1, m, n);
    }
}
```



+ 遍历过1后将其改成0，可以省去visited数组的空间

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        max_area = 0
        self.area = 0
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    self.area = 0
                    self.dfs(grid, i, j, m, n)
                    max_area = max(max_area, self.area)
        return max_area

    def dfs(self, grid, i, j, m, n):
        if i < 0 or j < 0 or i >= m or j >= n:
            return
        if grid[i][j] == 0:
            return 
        self.area += 1
        grid[i][j] = 0
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            self.dfs(grid, ni, nj, m, n)
```





## [剑指 Offer II 106. 二分图](https://leetcode-cn.com/problems/vEAB3K/)

+ 要注意 每个连通分支 是二分图才行
+ 类似广度优先

```java
class Solution {
    public boolean isBipartite(int[][] graph) {
        int n = graph.length;
        int[] color = new int[n];
        Arrays.fill(color, -1);
        Deque<Integer> deque = new LinkedList<>();
        for (int start = 0; start < n; start++) {
            if (color[start] != -1) {
                continue;
            }
            color[start] = 0;
            deque.add(start);
            while (!deque.isEmpty()) {
                int node = deque.removeFirst();
                int node_col = color[node];
                for (int adj : graph[node]) {
                    if (color[adj] == -1) {
                        color[adj] = 1 - node_col;
                        deque.add(adj);
                    }else {
                        if (1 - node_col != color[adj]) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
}
```



## [剑指 Offer II 107. 矩阵中的距离](https://leetcode-cn.com/problems/2bCMpM/)

+ BFS
+ 从所有0点开始遍历，加入adjacent

```java
class Solution {
    boolean[][] visited;
    Deque<int[]> deque;
    int m;
    int n;

    public int[][] updateMatrix(int[][] mat) {
        m = mat.length;
        n = mat[0].length;
        int[][] dist = new int[m][n];
        visited = new boolean[m][n];
        deque = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 0) {
                    deque.add(new int[]{i, j, 0});
                    visited[i][j] = true;
                }
            }
        }
        // bfs
        while (!deque.isEmpty()) {
            int[] pair = deque.removeFirst();
            int i = pair[0], j = pair[1], val = pair[2];
            dist[i][j] = val;
            addAdj(i, j, val);
        }
        return dist;
    }

    private void addAdj(int i, int j, int cur_val) {
        int[][] adj = {{i-1, j}, {i+1, j}, {i, j-1}, {i, j+1}};
        for (int[] idx : adj) {
            int ni = idx[0], nj = idx[1];
            if (ni < 0 || nj < 0 || ni >= m || nj >= n) continue;
            if (!visited[ni][nj]) {
                visited[ni][nj] = true;
                deque.add(new int[]{ni, nj, cur_val + 1});
            }
        }
    }
}
```



+ python

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        dist = [[0] * n for _ in range(m)]
        visited = [[False] * n for _ in range(m)]
        deque = collections.deque()
        for i in range(m):
            for j in range(n):
                if (mat[i][j] == 0):
                    deque.append((i, j))
                    visited[i][j] = True 
        
        dire = ((-1, 0), (1, 0), (0, -1), (0, 1))
        step = 0
        while deque:
            for _ in range(len(deque)):
                i, j = deque.popleft()
                dist[i][j] = step
                # 加入 adj
                for dx, dy in dire:
                    ni, nj = i + dx, j + dy 
                    if ni < 0 or nj < 0 or ni >= m or nj >= n:
                        continue
                    if visited[ni][nj] is False:
                        visited[ni][nj] = True
                        deque.append((ni, nj))
            step += 1
        return dist 
```



## [剑指 Offer II 108. 单词演变](https://leetcode-cn.com/problems/om3reC/)



![image-20220131154508718](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20220131154508718.png)

```java
class Solution {
    Map<String, List<String>> edge;

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        edge = new HashMap<>();
        for (String word : wordList) {
            addEdge(word);
        }
        addEdge(beginWord);
        if (!edge.containsKey(endWord)) return 0;

        //System.out.println(edge.toString());
        Deque<Object[]> deque = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        deque.add(new Object[]{beginWord, 1});
        visited.add(beginWord);
        while (!deque.isEmpty()) {
            Object[] pair = deque.removeFirst();
            String word = (String)pair[0];
            int len = (int)pair[1];
            if (word.equals(endWord)) {
                return (len + 1) / 2;
            }
            for (String next : edge.get(word)) {
                if (!visited.contains(next)) {
                    visited.add(next);
                    deque.add(new Object[]{next, len + 1});
                }
            }
        }
        return 0;
    }

    void addEdge(String word) {
        if (edge.containsKey(word)) return;
        edge.put(word, new ArrayList<>());
        char[] arr = word.toCharArray();
        for (int i = 0; i < arr.length; i++) {
            char temp = arr[i];
            arr[i] = '#';
            String mid = new String(arr);
            if (!edge.containsKey(mid)) edge.put(mid, new ArrayList<>());
            edge.get(word).add(mid);
            edge.get(mid).add(word);
            arr[i] = temp;
        }
    }
}
```



```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        self.edge = collections.defaultdict(list)
        for word in wordList:
            self.addEdge(word)
        self.addEdge(beginWord)
        if endWord not in self.edge:
            return 0 
        
        deque = collections.deque()
        visited = set()
        deque.append((beginWord, 1))
        visited.add(beginWord)
        while deque:
            word, llen = deque.popleft()
            if word == endWord:
                return (llen + 1) // 2
            for next_word in self.edge[word]:
                if next_word not in visited:
                    visited.add(next_word)
                    deque.append((next_word, llen + 1))
        return 0

    def addEdge(self, word):
        if word in self.edge:
            return
        char_lst = [s for s in word]
        for i in range(len(word)):
            temp = char_lst[i]
            char_lst[i] = '#'
            mid = ''.join(char_lst)
            self.edge[word].append(mid)
            self.edge[mid].append(word)
            char_lst[i] = temp
```



## [剑指 Offer II 110. 所有路径](https://leetcode-cn.com/problems/bP4bmD/)

+ dfs
+ 无环图, 不需要visited 记录

```java
class Solution {
    List<List<Integer>> ans;
    List<Integer> track;

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        int n = graph.length;
        ans = new ArrayList<>();
        track = new ArrayList<>();
        track.add(0);
        dfs_backtrack(graph, 0, n);
        return ans;
    }

    private void dfs_backtrack(int[][] graph, int pre_posi, int n) {
        if (pre_posi == n - 1) {
            ans.add(new ArrayList<Integer>(track));
            return;
        }
        for (int next_posi : graph[pre_posi]) {
            // if (visited[next_posi]) continue;
            // 无环图, 不需要visited
            track.add(next_posi);
            dfs_backtrack(graph, next_posi, n);
            track.remove(track.size() - 1);
        }
    }
}
```



+ python

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        self.ans = []
        self.track = [0]
        n = len(graph)
        self.dfs_backtrack(graph, 0, n)
        return self.ans


    def dfs_backtrack(self, graph, pre_posi, n):
        if pre_posi == n - 1:
            self.ans.append(self.track.copy())
            return 
        for next_posi in graph[pre_posi]:
            self.track.append(next_posi)
            self.dfs_backtrack(graph, next_posi, n)
            self.track.pop()
```



## [剑指 Offer II 111. 计算除法](https://leetcode-cn.com/problems/vlzXQL/)

+ 建立map

```java
class Solution {
    Map<String, Map<String, Double>> map;
    Set<String> visited;
    double div;

    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int n = equations.size();
        map = new HashMap<>();
        visited = new HashSet<>();
        for (int i = 0; i < n; i++) {
            String s1 = equations.get(i).get(0);
            String s2 = equations.get(i).get(1);
            double val = values[i];
            insert(s1, s2, val);
            insert(s2, s1, 1.0 / val);
        }
        //System.out.println(map.get("b"));
        double[] ans = new double[queries.size()];
        for (int i = 0; i < ans.length; i++) {
            String nume = queries.get(i).get(0);
            String deno = queries.get(i).get(1);
            if (!map.containsKey(nume) || !map.containsKey(deno)) {
                ans[i] = -1.0;
            }else {
                ans[i] = calc(queries.get(i).get(0), queries.get(i).get(1));
            }
        }

        return ans;
    }


    private double calc(String start, String end) {
        div = -1.0;
        visited.clear();
        dfs(start, end, 1);
        visited.add(start);
        return div;
    }

    private void dfs(String pre, String end, double mul) {
        if (div != -1.0) return;
        if (pre.equals(end)) {
            div = mul;
            return; 
        }
        for (Map.Entry<String, Double> entry : map.get(pre).entrySet()) {
            String next = entry.getKey();
            if (visited.contains(next)) continue;
            double val = entry.getValue();
            visited.add(next);
            dfs(next, end, mul * val);
            visited.remove(next);
        }
    }
   
    
    private void insert(String s1, String s2, double val) {
        if (!map.containsKey(s1)) {
            map.put(s1, new HashMap<>());
        }
        map.get(s1).put(s2, val);
    }
}
```



## [剑指 Offer II 112. 最长递增路径](https://leetcode-cn.com/problems/fpTFWP/)

+ dfs
+ 记忆化搜索
+ dfs 返回i， j 为起点的最长递增路径长度

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        self.dire = ((-1, 0), (1, 0), (0, -1), (0, 1))
        self.memo = [[-1] * n for _ in range(m)]
        ans = 1
        for i in range(m):
            for j in range(n):
                ans = max(ans, self.dfs(matrix, i, j, m, n))
        # print(self.dfs(matrix, 0, 0, m, n))
        return ans 

    def dfs(self, matrix, i, j, m, n):
        if self.memo[i][j] != -1:
            return self.memo[i][j]
        max_step = 1
        for di, dj in self.dire:
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or ni >= m or nj >= n:
                continue 
            if matrix[ni][nj] > matrix[i][j]:
                max_step = max(max_step, self.dfs(matrix, ni, nj, m, n) + 1)
        self.memo[i][j] = max_step
        return max_step
        
```



## [剑指 Offer II 113. 课程顺序](https://leetcode-cn.com/problems/QA2IGt/)

+ 拓扑排序, bfs
+ 记录每个点的入度
+ 入度为0说明可以选择了，加入deque 进行dfs
+ 遍历到某一个点，将其连接的点的入度 -1 即可

```java
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> edge = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            edge.add(new ArrayList<>());
        }
        int[] in_cnt = new int[numCourses];
        for (int[] pair : prerequisites) {
            int fir = pair[1];
            int sec = pair[0];
            in_cnt[sec] += 1;
            edge.get(fir).add(sec);
        }
        Deque<Integer> deque = new LinkedList<>();
        int[] ans = new int[numCourses];
        int idx = 0;
        for (int i = 0; i < numCourses; i++) {
            if (in_cnt[i] == 0) {
                deque.add(i);
            }
        }
        while (!deque.isEmpty()) {
            int top = deque.removeFirst();
            ans[idx++] = top;
            for (int next : edge.get(top)) {
                in_cnt[next] -= 1;
                if (in_cnt[next] == 0) {
                    deque.add(next);
                }
            }
        }
        if (idx != numCourses) return new int[0];
        return ans;
    }
}
```



