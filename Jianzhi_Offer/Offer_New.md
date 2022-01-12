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

