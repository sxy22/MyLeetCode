# 程序员面试金典（第 6 版）

## [面试题 01.02. 判定是否互为字符重排](https://leetcode.cn/problems/check-permutation-lcci/)

+ hashtable 存 字符cnt
+ cnt < 0 返回false

**Python**

```python
class Solution:
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False
        cnt = collections.defaultdict(int)
        for ch in s1:
            cnt[ch] += 1
        for ch in s2:
            cnt[ch] -= 1
            if cnt[ch] < 0:
                return False
        return True 
```

**Java**

```java
class Solution {
    public boolean CheckPermutation(String s1, String s2) {
        if (s1.length() != s2.length()) {
            return false;
        }
        Map<Character, Integer> cnt = new HashMap<>();
        for (int i = 0; i < s1.length(); i++) {
            char ch = s1.charAt(i);
            cnt.put(ch, cnt.getOrDefault(ch, 0) + 1);
        }
        for (int i = 0; i < s2.length(); i++) {
            char ch = s2.charAt(i);
            cnt.put(ch, cnt.getOrDefault(ch, 0) - 1);
            if (cnt.get(ch) < 0) {
                return false;
            }
        }
        return true;

    }
}
```



## [面试题 01.05. 一次编辑](https://leetcode.cn/problems/one-away-lcci/)

+ 分类讨论

```python
class Solution:
    def oneEditAway(self, first: str, second: str) -> bool:
        if len(first) > len(second):
            first, second = second, first
        if len(second) - len(first) > 1:
            return False 
        i, j = 0, 0
        diff = 0
        while i < len(first):
            if first[i] != second[j]:
                if diff > 0:
                    return False  
                diff += 1
                if len(second) == len(first):
                    i += 1
                    j += 1
                else:
                    j += 1
            else:
                i += 1
                j += 1
        return True
```



## [面试题 01.06. 字符串压缩](https://leetcode.cn/problems/compress-string-lcci/)

+ StringBuilder 的使用

**Java**

```java
class Solution {
    public String compressString(String S) {
        StringBuilder compressed_s = new StringBuilder();
        int i = 0;
        while (i < S.length()) {
            int j = i;
            int cnt = 0;
            while (j < S.length()) {
                if (S.charAt(i) == S.charAt(j)) {
                    cnt += 1;
                    j += 1;
                }else {
                    break;
                }
            }
            compressed_s.append(S.charAt(i));
            compressed_s.append(String.valueOf(cnt));
            i = j;
        } 
        if (compressed_s.length() >= S.length()) {
            return S;
        }
        return compressed_s.toString();
    }
}
```



## [面试题 02.03. 删除中间节点](https://leetcode.cn/problems/delete-middle-node-lcci/)

+ 将后一个结点的值替换到node
+ 删除后一个结点

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}
```



## [面试题 02.05. 链表求和](https://leetcode.cn/problems/sum-lists-lcci/)

**Java**

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode pre = new ListNode(-1);
        ListNode node = pre;
        while (l1 != null || l2 != null) {
            int x = 0, y = 0;
            if (l1 != null) {
                x = l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                y = l2.val;
                l2 = l2.next;
            }
            int sum = x + y + carry;
            carry = sum / 10;
            node.next = new ListNode(sum % 10);
            node = node.next;
        }
        if (carry != 0) {
            node.next = new ListNode(carry);
        }
        return pre.next;
    }
}
```





## [面试题 02.06. 回文链表](https://leetcode.cn/problems/palindrome-linked-list-lcci/)

+ 找到链表中点后翻转后半部分链表
+ 比较是否相同



```java
    logo
    学习
    题库
    讨论
    竞赛
    求职

商店
1
回文链表
提交记录
26 / 26 个通过测试用例
状态：通过
执行用时: 1 ms
内存消耗: 43.8 MB
提交时间：3 个月前
执行用时分布图表
执行消耗内存分布图表
邀请好友来挑战 回文链表
提交的代码： 3 个月前

语言： java

添加备注

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        // 找中点
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode l1 = head;
        ListNode l2 = reverse(slow.next);
        while (l2 != null) {
            if (l1.val != l2.val) return false;
            l1 = l1.next;
            l2 = l2.next;
        }
        return true;
    }

    ListNode reverse(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        ListNode next = null;
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







2.08















## [面试题 04.06. 后继者](https://leetcode.cn/problems/successor-lcci/)

+ 中序遍历找下一个



**Java**

+ 递归

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
    TreeNode pre_node;
    TreeNode ans;

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        pre_node = null;
        ans = null;
        dfs(root, p);
        return ans;
    }

    private void dfs(TreeNode node, TreeNode p) {
        if (node == null || ans != null) {
            return;
        }
        dfs(node.left, p);
        if (ans != null) return;
        if (pre_node == p) {
            ans = node;
            return;
        }else {
            pre_node = node;
        }
        dfs(node.right, p);
    }
}
```

+ 迭代 

```java
class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        TreeNode pre = null;
        Deque<TreeNode> stack = new LinkedList<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.add(root);
                root = root.left;
            }
            TreeNode top = stack.removeLast();
            if (pre == p) {
                return top;
            }
            pre = top;
            root = top.right;
        }
        return null;
    }
}
```

