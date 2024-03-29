# 剑指Offer(第二版)

## [剑指Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

**思路一**

+ 使用hashset

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        HashSet set = new HashSet();

        for (int num : nums) {
            boolean b = set.add(num);
            if (b == false){
                return num;
            }
        }
        return -1;
    }
}
```

**思路二**

+ 所有数字都在 0～n-1 的范围内， 可用原数组作为hashset

+ 检查nums[i] 是否等于i， 若是则i++

+ 若不是，nums[i]的正确下标为nums[i]， 看是否相等，相等则找到了重复数字

+ 交换 下标 i 和 nums[i]， 继续在i处检查

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        i = 0
        while i < len(nums):
            val = nums[i]
            if val == i:
                i += 1
                continue
            if nums[val] == val:
                return val 
            nums[i], nums[val] = nums[val], nums[i]
        return -1
```



## [剑指Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

**思路**

+ 从右上角元素开始查找
+ target < matrix\[i\]\[j\], j -= 1
+ target > matrix\[i\]\[j\], i += 1

```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        row = len(matrix)
        if row == 0:
            return False
        column = len(matrix[0])
        if column == 0:
            return False
        i = 0
        j = column - 1
        while j >= 0 and i <= row - 1:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            else:
                i +=1
        return False
```



## [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

请实现一个函数，把字符串 s 中的每个空格替换成"%20"。



+ 注意stringbuilder的使用
+ 可以先指定一个长度，避免扩容



```java
class Solution {
    public String replaceSpace(String s) {
        StringBuilder res = new StringBuilder(s.length() * 3);
        for (int i = 0; i < s.length(); i++){
            if (s.charAt(i) == ' '){
                res.append("%20");
            }else{
                res.append(s.charAt(i));
            }
        }
        return res.toString();
    }
}
```



```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        ans = []
        for ch in s:
            if ch == ' ':
                ans.append("%20")
            else:
                ans.append(ch)
        return ''.join(ans)
```



## [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        res = []
        while head is not None:
            res.append(head.val)
            head = head.next
        return res[::-1]
```





## [剑指Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

**思路**

+ 前序遍历的首元素 为 树的根节点 node 的值

+ 在中序遍历中搜索根节点 node 的索引 ，可将 中序遍历 划分为 [ 左子树 | 根节点 | 右子树 ] 

+ 根据中序遍历中的左 / 右子树的节点数量，可将 前序遍历 划分为 [ 根节点 | 左子树 | 右子树 ] 

+ 使用pre_left, pre_right, in_left, in_right 记录范围，并使用哈希表记录中序遍历中元素，可快速查找，避免切片操作

  

```java
class Solution {
    HashMap<Integer, Integer> map;
    int[] preorder;
    int[] inorder;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        this.inorder = inorder;
        // map 记录 in_order 中的元素下标
        this.map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return this.recur(0, preorder.length - 1, 0 ,inorder.length - 1);
    }

    TreeNode recur(int pre_left, int pre_right, int in_left, int in_right) {
        if (pre_left > pre_right) {
            return null;
        }
        // 取出root
        int head = preorder[pre_left];
        // 找到root 在in_order中的下标
        int in_mid = map.get(head);
        // 左子树长度为 in_mid - in_left 
        int pre_mid = pre_left + in_mid - in_left;
        TreeNode root = new TreeNode(head);
        root.left = this.recur(pre_left + 1, pre_mid, in_left, in_mid - 1);
        root.right = this.recur(pre_mid + 1, pre_right, in_mid + 1, in_right);
        return root;
    }
}
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        self.map = {}
        n = len(inorder)
        for i in range(n):
            self.map[inorder[i]] = i
        self.preorder = preorder
        self.inorder = inorder
        root = self.helper(0, n - 1, 0, n - 1)
        return root 

        
    def helper(self, pre_l, pre_r, in_l, in_r):
        if pre_l > pre_r:
            return None
        root_val = self.preorder[pre_l]
        root = TreeNode(root_val)
        # root 在inorder的idx
        in_mid = self.map[root_val]
        # 左右子树结点个数
        l_cnt = in_mid - in_l
        r_cnt = in_r - in_mid
        root.left = self.helper(pre_l + 1, pre_l + l_cnt, in_l, in_mid - 1)
        root.right = self.helper(pre_l + l_cnt + 1, pre_r, in_mid + 1, in_r)
        return root 
```





## [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

**思路**

+ stack1 接受新的值
+ stack2 辅助pop操作

```python
class CQueue:

    def __init__(self):
        # stack1 接受新的值
        # stack2 辅助pop操作
        self.stack1 = []
        self.stack2 = []

    def appendTail(self, value: int) -> None:
        self.stack1.append(value)

    def deleteHead(self) -> int:
        # stack2中有元素，直接pop
        if len(self.stack2) != 0:
            return self.stack2.pop()
        if len(self.stack1) == 0:
            return -1
        # 将1中的元素换入2中，再pop
        while len(self.stack1) != 0:
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
```



## [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

+ 原地动态规划

```python
class Solution:
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        max_int = 1000000007
        a, b = 0, 1
        for i in range(n - 1):
            a, b = b, a + b 
            if b >= max_int:
                b %= max_int
        return b
```

**Java**

```java
class Solution {
    public int fib(int n) {
        if (n == 0) {
            return 0;
        }
        int x1 = 0;
        int x2 = 1;
        int max_int = 1000000007;
        while (n > 1) {
            n -= 1;
            int temp = x2;
            x2 = (x1 + x2) % max_int;
            x1 = temp;
        }
        return x2;
    }
}
```





## [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

```python
class Solution:
    def numWays(self, n: int) -> int:
        if n == 0:
            return 1 
        if n <= 2:
            return n 
        a, b = 1, 2
        max_int = 1000000007
        for i in range(n - 2):
            a, b = b, a + b
            if b > max_int:
                b %= max_int
        return b 
```





## [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

+ 二分查找，有重复 数字
+ numbers[i] < numbers[j]， 则此段有序，i位置即为最小值，直接返回
+ numbers[mid] < numbers[j]， 右段有序，可能 范围 i - mid
+ numbers[mid] > numbers[j], 右段无序，可能范围mid+1 - j
+ numbers[mid] == numbers[j]， 无法判断， j -= 1, mid还在，不会错过答案

```python
class Solution:
    def minArray(self, numbers: [int]) -> int:
        i, j = 0, len(numbers)-1
        while i < j:
            if numbers[i] < numbers[j]:
                return numbers[i]
            mid = i + (j - i) // 2
            if numbers[mid] < numbers[j]:
                j = mid
            elif numbers[mid] > numbers[j]:
                i = mid + 1
            else:
                j -= 1
        return numbers[i]
```



**Java**

```java
class Solution {
    public int minArray(int[] numbers) {
        int i = 0, j = numbers.length - 1;
        while (i < j) {
            if (numbers[i] < numbers[j]) {
                return numbers[i];
            }
            int mid = (i + j) / 2;
            if (numbers[mid] < numbers[j]) {
                j = mid;
            }else if (numbers[mid] > numbers[j]) {
                i = mid + 1;
            }else {
                j -= 1;
            }
        }
        return numbers[i];
    }
}
```



## [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

**思路**

+ 搜索过的位置标记为“+”， 即可解决重复搜索
+ 也可以定义传统的searched set()

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        self.word = word 
        self.ans = False
        self.m, self.n  = len(board), len(board[0])
        for i in range(self.m):
            for j in range(self.n):
                self.DFS(board, i, j, 0)
        return self.ans

        
    def DFS(self, board, i, j, idx):
        if self.ans:
            return
        if idx == len(self.word):
            self.ans = True
            return
        if i < 0 or i >= self.m or j < 0 or j >= self.n:
            return 
        if board[i][j] != self.word[idx]:
            return 
        temp = board[i][j]
        board[i][j] = '+'
        self.DFS(board, i+1, j, idx+1)
        self.DFS(board, i-1, j, idx+1)
        self.DFS(board, i, j+1, idx+1)
        self.DFS(board, i, j-1, idx+1)
        board[i][j] = temp 
        return 
```



```java
class Solution {
    boolean ans;

    public boolean exist(char[][] board, String word) {
        ans = false;
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dfs(board, word, i, j, 0, m, n);
            } 
        }
        return ans;
    }

    private void dfs(char[][] board, String word, int i, int j, int idx, int m, int n) {
        if (ans) return;
        if (idx == word.length()) {
            ans = true;
            return;
        }
        if (i < 0 || j < 0 || i > m - 1 || j > n - 1 || board[i][j] != word.charAt(idx)) {
            return;
        }
        char temp = word.charAt(idx);
        board[i][j] = '#';
        dfs(board, word, i - 1, j, idx + 1, m, n);
        dfs(board, word, i + 1, j, idx + 1, m, n);
        dfs(board, word, i, j - 1, idx + 1, m, n);
        dfs(board, word, i, j + 1, idx + 1, m, n);
        board[i][j] = temp;
    }
}
```



## [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

+ DFS
+ 如到达已搜索过的点，可以直接return，因为其他路径会从该店出发

**python**

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        self.visited = [[0] * n for _ in range(m)]
        self.ans = 0 
        self.DFS(0, 0, m, n, k)
        return self.ans 

    def DFS(self, i, j, m, n, k):
        if i < 0 or i >= m or j < 0 or j >= n:
            return 
        if self.visited[i][j] == 1:
            return 
        if self.bitsum(i) + self.bitsum(j) > k:
            return 
        self.visited[i][j] = 1
        self.ans += 1
        self.DFS(i + 1, j, m, n ,k)
        self.DFS(i - 1, j, m, n ,k)
        self.DFS(i, j + 1, m, n ,k)
        self.DFS(i, j - 1, m, n ,k)
        return 

    def bitsum(self, x):
        ans = 0
        while x > 0:
            ans += x % 10
            x = x // 10
        return ans 

```



**Java**

```java
class Solution {
    int ans;
    int[][] visited;
    
    public int movingCount(int m, int n, int k) {
        ans = 0;
        visited = new int[m][n];
        dfs(0, 0, m, n, k);
        return ans;
    }

    private void dfs(int i, int j, int m, int n, int k) {
        if (i < 0 || i >= m || j < 0 || j >= n || visited[i][j] == 1) {
            return;
        } 
        if (bitsum(i) + bitsum(j) > k) {
            return;
        }
        visited[i][j] = 1;
        ans += 1;
        dfs(i - 1, j, m, n, k);
        dfs(i + 1, j, m, n, k);
        dfs(i, j - 1, m, n, k);
        dfs(i, j + 1, m, n, k);
    }

    private int bitsum(int x) {
        int sum = 0;
        while (x > 0) {
            sum += x % 10;
            x = x / 10;
        }
        return sum;
    }
}
```





## [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

**思路1**

+ 和1 与操作， 可以得到最后一位的情况
+ 右移
+ 注意java中要用无符号右移>>>

```c++
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int ans = 0;
        while (n != 0) {
            ans += (n & 1);
            n = n >>> 1;
        }
        return ans;
    }
}
```



**思路2**

+ n & n - 1 会使n最右边的1变成0
+ 记录多少次操作会使n变为0

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int ans = 0;
        while (n != 0) {
            ans += 1;
            n = n & (n - 1);
        }
        return ans;
    }
}
```





## [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

**思路**

+ $x^n = x^{n / 2}x^{n / 2}$
+ n奇偶数
+ 递归

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        sign = 0
        if n < 0:
            sign = 1
            n = -1 * n 
        res = self.helper(x, n)
        if sign == 1:
            return 1 / res 
        return res 
    def helper(self, x, n):
        if n == 1:
            return x 
        temp = self.helper(x, n // 2)
        if n & 1 == 0:
            return temp * temp
        else:
            return x * temp * temp
```

+ 不递归

![image-20211218000111115](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211218000111115.png)

+ Python

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        ans = 1
        if n < 0:
            x = 1 / x
            n = -n
        while n != 0:   
            if (n & 1) == 1:
                ans *= x
            n = n >> 1
            x *= x
        return ans 
```



+ Java

```java
class Solution {
    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }
        long N = n;
        double ans = 1;
        if (N < 0) {
            N = -N;
            x = 1.0 / x;
        }
        while (N != 0) {
            if ((N & 1) == 1) {
                ans *= x;
            }
            N >>= 1;
            x *= x;
        }
        return ans;
    }
}
```





## [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

**思路**

+ 建立dummy head
+ 找到需要删除节点的前一个节点

```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        pre_head = ListNode(-1)
        pre_head.next = head 
        node = pre_head
        while node is not None:
            if node.next.val == val:
                node.next = node.next.next
                return pre_head.next
            node = node.next
```



```java
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        ListNode pre_head = new ListNode(-1);
        pre_head.next = head;
        ListNode pre_head_copy = pre_head;

        while (pre_head.next != null) {
            if (pre_head.next.val == val) {
                pre_head.next = pre_head.next.next;
                break;
            }
            pre_head = pre_head.next;
        }

        return pre_head_copy.next;

    }
}
```





## [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

**思路**

+ 双指针，交换靠前的偶数和靠后的奇数

```java
class Solution {
    public int[] exchange(int[] nums) {
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            // 找到从i开始第一个偶数
            while (i < j && (nums[i] & 1) == 1) {
                i++;
            }
            // 找到从j开始向前第一个奇数
            while (j > i && (nums[j] & 1) == 0) {
                j--;
            }
            // i  j 则交换
            int temp = nums[j];
            nums[j] = nums[i];
            nums[i] = temp;
            i++;
            j--;
        }
        return nums;
    }
}
```



## [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

**思路**

+ 让一个节点先走k步
+ 两个节点一起走
+ 快指针到none，则慢指针为倒数第k个节点

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        head1, head2 = head, head
        while k > 0:
            head2 = head2.next
            k -= 1
        while head2 is not None:
            head1 = head1.next
            head2 = head2.next
        return head1
```



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
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode slow = head;
        ListNode fast = head;
        while (k > 0) {
            fast = fast.next;
            k -= 1;
        }
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
```



## [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

+ pre, cur
+ cur 指向pre
+ cur.next 提前保存

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
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
```



## [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

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
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode prehead = new ListNode(-1);
        ListNode ans = prehead;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                prehead.next = l1;
                l1 = l1.next;
            }else {
                prehead.next = l2;
                l2 = l2.next;
            }
            prehead = prehead.next;
        }

        if (l1 == null) {
            prehead.next = l2;
        }else {
            prehead.next = l1;
        }
        return ans.next;
    }
}
```



## [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

**思路**

+ 遍历A的每个节点判断，是否包含B
+ 注意判断条件

```python
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if A is None or B is None:
            return False 
        if A.val == B.val:
            if self.twosame(A, B) is True:
                return True 
        # 深度优先遍历A, 对A的每一个节点进行判断
        return self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)


    def twosame(self, node1, node2):
        # node2是None, 无需考虑node1
        if node2 is None:
            return True
        # node1到底了，node2还没，肯定不是子结构
        elif node1 is None:
            return False
        # 节点值判断
        if node1.val != node2.val:
            return False
        # 向下递归
        return self.twosame(node1.left, node2.left) and self.twosame(node1.right, node2.right)
```



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
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) {
            return false;
        }
        if (healper(A, B)) {
            return true;
        }
        return isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }

    boolean healper(TreeNode A, TreeNode B) {
        if (B == null) return true;
        if (A == null || A.val != B.val) return false;
        return healper(A.left, B.left) && healper(A.right, B.right);
    }
}
```





## [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

**递归**

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        leftnode = root.left 
        root.left = self.mirrorTree(root.right)
        root.right = self.mirrorTree(leftnode)
        return root
```

```java
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if (root == null) {return root;}
        TreeNode left = root.left;
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(left);
        return root;
    }
}
```





**迭代**

+ 利用类似广度优先遍历，遍历每一个节点，并交换左右节点

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
            # 交换pop出来的节点的左右
            node.left, node.right = node.right, node.left
        return root
```

```java
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if (root == null) {return root;}
        LinkedList<TreeNode> stack = new LinkedList<>();
        stack.add(root);
        while (stack.size() != 0) {
            TreeNode node = stack.removeFirst();
            if (node.left != null) {stack.add(node.left);}
            if (node.right != null) {stack.add(node.right);}
            TreeNode left = node.left;
            node.left = node.right;
            node.right = left;
        }
        return root;
    }
}
```



## [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

+ 忽略root
+ 检查左右子树是否对称

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root is None:
            return True
        else:
            return self.twosym(root.left, root.right)


    def twosym(self, node1, node2):
        if node1 is None and node2 is None:
            return True
        if node1 is None or node2 is None:
            return False
        if node1.val != node2.val:
            return False
        return self.twosym(node1.left, node2.right) and self.twosym(node1.right, node2.left)
```

**Java**

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
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return twosym(root.left, root.right);

    }

    boolean twosym(TreeNode n1, TreeNode n2) {
        if (n1 == null && n2 == null) {
            return true;
        }else if (n1 == null || n2 == null) {
            return false;
        }else if (n1.val != n2.val) {
            return false;
        }
        return twosym(n1.left, n2.right) && twosym(n1.right, n2.left);
    }
}
```



## [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new LinkedList<>();
        int left = 0, top = 0;
        int right = matrix[0].length - 1, bottom = matrix.length - 1;

        while (left <= right && top <= bottom) {
            // top, left -> top, right
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
            }
            if (top == bottom) break;
            // top + 1, right -> bottom, right
            for (int i = top + 1; i <= bottom; i++) {
                res.add(matrix[i][right]);
            }
            if (left == right) break;
            // bottom, right - 1 -> bottom, left
            for (int i = right - 1; i >= left; i--) {
                res.add(matrix[bottom][i]);
            }
            // bottom - 1, left -> top + 1, left
            for (int i = bottom - 1; i >= top + 1; i--) {
                res.add(matrix[i][left]);
            }
            left++;
            top++;
            right--;
            bottom--;
        }
        return res;
    }
}
```



## [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

+ 栈的特点，一个元素入栈时的栈内元素和出栈时一致
+ 定义一个辅助min栈，记录每个元素入栈时的栈内最小值

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        cur_min = x 
        if (len(self.min_stack) != 0):
            cur_min = min(x, self.min_stack[-1])
        self.min_stack.append(cur_min)

    def pop(self) -> None:
        self.min_stack.pop()
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def min(self) -> int:
        return self.min_stack[-1]
```



```java
class MinStack {
    Deque<Integer> stack;
    Deque<Integer> min_stack;

    /** initialize your data structure here. */
    public MinStack() {
        stack = new LinkedList<>();
        min_stack = new LinkedList<>();
    }

    public void push(int x) {
        stack.add(x);
        if (min_stack.size() == 0) {
            min_stack.add(x);
        }
        else {
            int min = Math.min(x, min_stack.getLast());
            min_stack.addLast(min);
        }
    }

    public void pop() {
        stack.removeLast();
        min_stack.removeLast();
    }
    
    public int top() {
        return stack.getLast();
    }

    public int min() {
        return min_stack.getLast();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
```





## [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

**思路**

+ 模拟一个栈
+ 每加入一个数字后，判断能否pop

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        idx = 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[idx]:
                stack.pop()
                idx += 1
        return len(stack) == 0
```

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> stack = new LinkedList<>();
        int idx = 0;
        for (int num : pushed) {
            stack.add(num);
            while (!stack.isEmpty() && stack.getLast() == popped[idx]) {
                stack.removeLast();
                idx += 1;
            }
        }
        return stack.isEmpty();
    }
}
```





## [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

+ 广度优先遍历
+ 不需要考虑层数

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        queue = collections.deque()
        queue.append(root)
        res = []
        while queue:
            top = queue.popleft()
            # None 可以进栈，但不输出
            if top is not None:
                res.append(top.val)
                queue.append(top.left)
                queue.append(top.right)
        return res
```



Java

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
    public int[] levelOrder(TreeNode root) {
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        ArrayList<Integer> ans = new ArrayList<>();
        while (!queue.isEmpty()) {
            TreeNode top = queue.removeFirst();
            if (top != null) {
                ans.add(top.val);
                queue.add(top.left);
                queue.add(top.right);
            }
        }
        int[] res = new int[ans.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = ans.get(i);
        }
        return res;
    }
}
```



## [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

+ 广度优先遍历
+ 需要考虑层数，每一层需要单独一个循环



```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        queue = collections.deque()
        queue.append(root)
        res = []
        while len(queue) > 0:
            # 记录当前层
            cur_layer = []
            # 此时queue中为所有当前层的节点，共循环len(queue)次
            for _ in range(len(queue)):
                top = queue.popleft()
                cur_layer.append(top.val)
                if top.left:
                    queue.append(top.left)
                if top.right:
                    queue.append(top.right)
            res.append(cur_layer)
        return res
```



```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        LinkedList<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> ans = new ArrayList<>();
        if (root != null) {
            queue.add(root);
        }
        while (!queue.isEmpty()) {
            List<Integer> layer = new ArrayList<>();
            int s = queue.size();
            for (int i = 0; i < s; i++) {
                TreeNode top = queue.removeFirst();
                layer.add(top.val);
                if (top.left != null) queue.add(top.left);
                if (top.right != null) queue.add(top.right);
            }
            ans.add(layer);
        }
        return ans;
    }
}
```



## [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if(root != null) queue.add(root);
        while(!queue.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            for(int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                if(res.size() % 2 == 0) tmp.addLast(node.val); // 偶数层 -> 队列头部
                else tmp.addFirst(node.val); // 奇数层 -> 队列尾部
                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            res.add(tmp);
        }
        return res;
    }
}
```



## [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

+ 后续遍历最后一个元素是head
+ 分成左右子树，递归判断



**Python**

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        return self.recur(postorder, 0, len(postorder) - 1)

    def recur(self, postorder, i, j):
        if i >= j:
            return True 
        head_val = postorder[j]
        right = i
        while postorder[right] < head_val:
            right += 1
        for k in range(right, j):
            if postorder[k] < head_val:
                return False 
        return self.recur(postorder, i, right - 1) and self.recur(postorder, right, j - 1)

```





**Java**

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return recur(postorder, 0, postorder.length - 1);
    }

    boolean recur(int[] postorder, int i, int j) {
        if (i >= j) return true;
        int head = postorder[j];
        int mid = i;
        while (postorder[mid] < head) {
            mid += 1;
        }
        // 左子树 i - mid - 1
        for (int k = mid; k < j; k++) {
            if (postorder[k] < head) {
                return false;
            }
        }
        return recur(postorder, i, mid - 1) && recur(postorder, mid, j - 1);
    }
}
```



## [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

**Java**

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
    LinkedList<List<Integer>> ans = new LinkedList<>();
    LinkedList<Integer> path = new LinkedList<>();
    
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        dfs(root, target);
        return ans;
    }
    
    void dfs(TreeNode node, int target) {
        if (node == null) return;
        path.addLast(node.val);
        if (target == node.val && node.left == null && node.right == null) {
            ans.addLast(new LinkedList(path));
        }
        dfs(node.left, target - node.val);
        dfs(node.right, target - node.val);
        path.removeLast();
    } 
}
```



**Python**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        self.ans = []
        self.path = []
        self.recur(root, target)
        return self.ans

    def recur(self, node, target):
        if node is None:
            return 
        self.path.append(node.val)
        target -= node.val
        if target == 0 and node.left is None and node.right is None:
            self.ans.append([x for x in self.path])
        self.recur(node.left, target)
        self.recur(node.right, target)
        self.path.pop()
```





## [剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

+ 利用哈希表的查询特点，考虑构建 **原链表节点** 和 **新链表对应节点** 的键值对映射关系

**Python**

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head is None:
            return head 
        hmap = dict()
        pre_head = Node(-1, None, None)
        pre_head_copy = pre_head 
        head_copy = head 
        while head is not None:
            node = Node(head.val, None, None)
            hmap[head] = node 
            pre_head.next = node
            head = head.next
            pre_head = pre_head.next

        cur = head_copy 
        while cur is not None:
            # cur.random 可能是null
            if cur.random is not None:
                hmap[cur].random = hmap[cur.random]
            cur = cur.next 
        return pre_head_copy.next


```



**Java**

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;

        Map<Node, Node> map = new HashMap<>();
        Node prehead = new Node(-1);
        Node cur = prehead;
        Node chead = head;
        while (chead != null) {
            Node new_node = new Node(chead.val);
            map.put(chead, new_node);
            cur.next = new_node;
            cur = cur.next;
            chead = chead.next;
        }
        chead = head;
        while (chead != null) {
            map.get(chead).random = map.get(chead.random);
            chead = chead.next;
        }
        return prehead.next;
    }
}
```





## [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

**Java**

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    Node pre = new Node(-1);
    Node pre_copy = pre;

    public Node treeToDoublyList(Node root) {
        if (root == null) return null;
        DFS(root);
        Node head = pre_copy.right;
        head.left = pre;
        pre.right = head;
        return head;
    }

    void DFS(Node node) {
        if (node == null) return;
        DFS(node.left);
        pre.right = node;
        node.left = pre;
        pre = pre.right;
        DFS(node.right);
    }
}
```



```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if root is None:
            return None
        self.pre = Node(-1)
        dummy_node = self.pre
        self.dfs(root)
        head = dummy_node.right
        tail = self.pre
        head.left = tail
        tail.right = head 
        return head 


    def dfs(self, root):
        if root is None:
            return 
        self.dfs(root.left)
        cur = root
        self.pre.right = cur
        cur.left = self.pre 
        self.pre = cur
        self.dfs(root.right)
        return
```







## [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

+ 层序遍历序列化

**Python**

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return "*"
        res = []
        queue = collections.deque()
        queue.append(root)
        while len(queue) > 0:
            top = queue.popleft()
            if top is None:
                res.append("*")
            else:
                res.append(str(top.val))
                queue.append(top.left)
                queue.append(top.right)
        return "_".join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """
        if data == "*":
            return None

        data = data.split("_")
        root = self.getnode(data[0])
        queue = collections.deque()
        queue.append(root)
        i = 1
        while len(queue) > 0:
            top = queue.popleft()
            if top is None:
                continue
            top.left = self.getnode(data[i])
            i += 1
            top.right = self.getnode(data[i])
            i += 1
            queue.append(top.left)
            queue.append(top.right)
        return root

    def getnode(self, s):
        if s == "*":
            return None
        else:
            return TreeNode(int(s))

```



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

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "*";
        StringBuilder code = new StringBuilder();
        Deque<TreeNode> deque = new LinkedList<>();
        deque.add(root);
        while (!deque.isEmpty()) {
            TreeNode top = deque.removeFirst();
            if (top == null) {
                code.append("*");
            }else {
                code.append(top.val);
                deque.add(top.left);
                deque.add(top.right);
            }
            code.append("_");
        }
        code.deleteCharAt(code.length() - 1);
        //System.out.println(code.toString());
        return code.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.equals("*")) return null;
        String[] seq = data.split("_");
        // System.out.println(Arrays.toString(seq));
        TreeNode root = getNode(seq[0]);
        Deque<TreeNode> deque = new LinkedList<>();
        deque.add(root);
        int i = 1;
        while (!deque.isEmpty()) {
            TreeNode top = deque.removeFirst();
            top.left = getNode(seq[i]);
            top.right = getNode(seq[i + 1]);
            i += 2;
            if (top.left != null) deque.add(top.left);
            if (top.right != null) deque.add(top.right);
        }
        return root;
    }

    TreeNode getNode(String val) {
        if (val.equals("*")) {
            return null;
        }else {
            return new TreeNode(Integer.parseInt(val));
        }
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));
```



## [剑指 Offer 38. 字符串的排列(缺](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        self.res = []
        # 记录是字符剩余可用个数，每次只能取一个
        self.cnt = collections.Counter(s)
        # 共用的track
        self.track = []
        self.n = len(s)
        # 需要遍历的字符
        self.keys = self.cnt.keys()

        self.backtrack_dfs()
        return self.res 

    def backtrack_dfs(self):
        if len(self.track) == self.n:
            self.res.append(''.join(self.track))
            return
        # 遍历self.cnt.keys()
        for num in self.keys:
            if self.cnt[num] != 0:
                self.track.append(num)
                self.cnt[num] -= 1
                self.backtrack_dfs()
                self.track.pop()
                self.cnt[num] += 1
        return 
```



**Java**

```java
class Solution {
    Map<Character, Integer> cnt;
    ArrayList<String> ans;
    StringBuilder track;
    int n;

    public String[] permutation(String s) {
        cnt = new HashMap<>();
        ans = new ArrayList<>();
        track = new StringBuilder();
        n = s.length();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            cnt.put(ch, 1 + cnt.getOrDefault(ch, 0));
        }
        dfs();
        return ans.toArray(new String[ans.size()]);


    }

    private void dfs() {
        if (track.length() == n) {
            ans.add(track.toString());
            return;
        }
        for (Character ch : cnt.keySet()) {
            int ch_cnt = cnt.get(ch);
            if (ch_cnt != 0) {
                track.append(ch);
                cnt.put(ch, ch_cnt - 1);
                dfs();
                track.deleteCharAt(track.length() - 1);
                cnt.put(ch, ch_cnt);
            }
        }
    }
}
```



## [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

+ 摩尔投票法

![image-20211221230229136](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211221230229136.png)

**Java**

```java
class Solution {
    public int majorityElement(int[] nums) {
        int vote = 0;
        int maj = -1;
        for (int x : nums) {
            if (vote == 0) {
                maj = x;
            }
            if (x == maj) {
                vote += 1;
            }else {
                vote -= 1;
            }
        }
        return maj;
    }
}
```



**Python**

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        vote = 0
        maj = -1
        for x in nums:
            if vote == 0:
                maj = x
            if x == maj:
                vote += 1
            else:
                vote -= 1
        return maj
```





## [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

**方法1**

+ 堆排序
+ 用一个大根堆实时维护数组的前 k小值。首先将前k个数插入大根堆中，随后从第 k+1个数开始遍历
+ 如果当前遍历到的数比大根堆的堆顶的数要小，就把堆顶的数弹出，再插入当前遍历到的数
+ 复杂度O(nlogk)



```python
import heapq as Heap
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k == 0:
            return  []
        # python中的heap是小根堆
        heap = [-x for x in arr[:k]]
        Heap.heapify(heap)

        for i in range(k, len(arr)):
            num = arr[i]
            top = -heap[0]
            if num < top:
                Heap.heappop(heap)
                Heap.heappush(heap, -num)
            
        res = [-x for x in heap]
        return res 
```

**Java**

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        Queue<Integer> pq = new PriorityQueue<>();
        for (int i = 0; i < k; i++) {
            pq.offer(-arr[i]);
        }
        for (int i = k; i < arr.length; i++) {
            int num = arr[i];
            if (num < -pq.peek()) {
                pq.poll();
                pq.offer(-num);
            }
        }
        int[] ans = new int[k];
        int i = 0;
        for (int x : pq) {
            ans[i] = -x;
            i++;
        }
        return ans;
    }
}
```



**方法2 快排变形**

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        self.res = []
        if len(arr) == k:
            return arr 
        
        self.quicksearch(arr, 0, len(arr) - 1, k)
        return self.res 
    
    def quicksearch(self, lst, left, right, k):
        if right <= left:
            return 
        
        i = left
        j = right
        mid = lst[left]
        while i < j:
            while i < j and lst[j] >= mid:
                j-= 1
            lst[i], lst[j] = lst[j], lst[i]
            
            while i < j and lst[i] <= mid:
                i += 1
            lst[i], lst[j] = lst[j], lst[i]
        lst[i] = mid
        cnt = i - left
        if cnt == k - 1:
            self.res.extend(lst[left:i+1])
        elif cnt == k:
            self.res.extend(lst[left:i])
        elif cnt < k - 1:
            self.res.extend(lst[left:i+1])
            self.quicksearch(lst, i+1, right, k - cnt - 1)
        else:
            self.quicksearch(lst, left, i-1, k)
        return
```

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        qsort(arr, 0, arr.length - 1, k);
        int[] ans = new int[k];
        for (int i = 0; i < k; i++) {
            ans[i] = arr[i];
        }
        return ans;
    }


    void qsort(int[] arr, int l, int r, int k) {
        if (r < l) return;
        int i = l;
        int j = r;
        int tmp = arr[l];
        while (i < j) {
            while (i < j && arr[j] >= tmp) {
                j -= 1;
            }
            arr[i] = arr[j];
            while (i < j && arr[i] <= tmp) {
                i += 1;
            }
            arr[j] = arr[i];
        }
        arr[i] = tmp;
        if (i - l + 1 == k || i - l + 1 == k + 1) {
            return;
        }else if (i - l + 1 < k) {
            qsort(arr, i + 1, r, k - i + l - 1); 
        }else {
            qsort(arr, l, i - 1, k); 
        }
    }
}
```



## [剑指 Offer 41. 数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

+ 两个堆

```python
import heapq

class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.small = [] # 大根堆
        self.large = [] # 小根堆
        self.small_size = 0
        self.large_size = 0

    def addNum(self, num: int) -> None:
        if self.small_size == self.large_size:
            # 应该放在small
            # 先放进large, 再pop出一个放入small
            heapq.heappush(self.large, num)
            heapq.heappush(self.small, -heapq.heappop(self.large))
            self.small_size += 1
        else:
            # 应该放在large
            heapq.heappush(self.small, -num)
            heapq.heappush(self.large, -heapq.heappop(self.small))
            self.large_size += 1

    def findMedian(self) -> float:
        if self.small_size == self.large_size:
            return (-self.small[0] + self.large[0]) / 2
        else:
            return -self.small[0]
```



## [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

+ 用 dp[i]代表以第 i 个数结尾的「连续子数组的最大和」

```java
class Solution {
    public int maxSubArray(int[] nums) {   
        int[] dp = Arrays.copyOf(nums, nums.length);
        int max_sum = nums[0];
        for (int i = 1; i < dp.length; i++) {
            if (dp[i - 1] > 0) {
                dp[i] += dp[i - 1];
            }
            max_sum = Math.max(max_sum, dp[i]);
        }
        return max_sum;
    }
}
```



## [剑指 Offer 43. 1～n 整数中 1 出现的次数(没做)](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)



## [剑指 Offer 44. 数字序列中某一位的数字(没做)](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)





## [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

```java
class Solution {
    public String minNumber(int[] nums) {
        // 比较器
        Comparator<String> comp = new Comparator<>() {
            @Override
            public int compare(String o1, String o2) {
                String a = o1 + o2;
                String b = o2 + o1;
                if (a.compareTo(b) <= 0) return -1;
                return 1;
            }
        };
        String[] arr = new String[nums.length];
        for(int i = 0;i < nums.length;i++){
            arr[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(arr, comp);
        return String.join("", arr);
    }
}
```



+ python3中自己写快速排序

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        L = len(nums)
        nums = [str(num) for num in nums]
        #快排
        self.quick_sort(nums, 0, len(nums)-1)
        return ''.join(nums)

    def quick_sort(self, lst, L, R):
        if L >= R:
            return
        i = L
        j = R 
        mid = lst[i]
        while i < j:
            while i < j and self.larger(lst[j], mid):
                j -= 1
            lst[i] = lst[j]
            while  i < j and self.larger(mid, lst[i]):
                i += 1
            lst[j] = lst[i]

        lst[i] = mid
        self.quick_sort(lst, L, i - 1)
        self.quick_sort(lst, i + 1, R)
        
    def larger(self, num1, num2):
        if int(num1 + num2) >= int(num2 + num1):
            return True
        else:
            return False
```



+ python3中使用functools.cmp_to_key

```python
from functools import cmp_to_key
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        # 比较函数
        def comp(s1, s2):
            if s1 + s2 <= s2 + s1:
                return -1
            else:
                return 1

        nums = [str(num) for num in nums]
        # 注意用法 sorted(arr, key=cmp_to_key(cmp))
        nums.sort(key=cmp_to_key(comp))
        return ''.join(nums)
```



## [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

+ 简单的dp

```python
class Solution:
    def translateNum(self, num: int) -> int:
        nums = str(num)
        if len(nums) == 1:
            return 1
        dp = [1] * len(nums)
        if self.helper(nums[0:2]):
            dp[1] = 2
        
        for i in range(2, len(nums)):
            dp[i] = dp[i-1]
            if self.helper(nums[i-1:i+1]):
                dp[i] += dp[i-2]
        
        return dp[-1]
        
    def helper(self, s):
        if s >= '10' and s <= '25':
            return True
        return False
```

**Java**

```java
class Solution {
    public int translateNum(int num) {
        String snum = String.valueOf(num);
        int[] dp = new int[snum.length() + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < dp.length; i++) {
            dp[i] += dp[i - 1];
            if (valid(snum.substring(i - 2, i))) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[dp.length - 1];
    }

    boolean valid(String s) {
        if (s.compareTo("10") >= 0 && s.compareTo("25") <= 0) {
            return true;
        }
        return false;
    }
}
```



## [剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

```java
class Solution {
    int[][] dp;
    int row;
    int column;

    public int maxValue(int[][] grid) {
        this.dp = grid;
        this.row = this.dp.length;
        this.column = this.dp[0].length;
        for (int i = 0; i < this.row; i++) {
            for (int j = 0; j < this.column; j++) {
                this.dp[i][j] += Math.max(this.get(i - 1, j), this.get(i, j - 1));
            }
        }
        return this.dp[row - 1][column - 1];

    }
    int get(int i, int j) {
        if (i < 0 || j < 0) return 0;
        return this.dp[i][j];
    }
}
```



```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                dp[i][j] = max(self.get(dp, i - 1, j), self.get(dp, i, j - 1)) + grid[i][j]
        
        return dp[m - 1][n - 1]
    def get(self, grid, i, j):
        if i < 0 or j < 0:
            return 0
        return grid[i][j]
```





## [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int i = 0, j = 0;
        int n = s.length();
        Set<Character> hset = new HashSet<>();
        int ans = 0;
        while (i < n && j < n) {
            while (j < n && !hset.contains(s.charAt(j))) {
                hset.add(s.charAt(j));
                j += 1;
            }
            ans = Math.max(j - i, ans);
            hset.remove(s.charAt(i));
            i += 1;
        }
        return ans;
    }
}
```



```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        hashset = set()
        i, j = 0, 0
        n = len(s)
        length = 0
        while j < n:
            while j < n and s[j] not in hashset:
                hashset.add(s[j])
                j += 1
            length = max(length, j - i)
            while j < n and s[j] in hashset:      
                hashset.remove(s[i])
                i += 1
        return length
```



## [剑指 Offer 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

**思路动态规划**

![image-20210410120459022](https://gitee.com/sxy22/note_images/raw/master/image-20210410120459022.png)

+ p2指，x2后大于当前位置前一个丑数，的最小的一个丑数

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1] * n 
        p2, p3, p5 = 0, 0, 0

        for i in range(1, n):
            n2 = dp[p2] * 2
            n3 = dp[p3] * 3
            n5 = dp[p5] * 5
            dp[i] = min(n2, n3, n5)
            
            if n2 == dp[i]:
                p2 += 1
            if n3 == dp[i]:
                p3 += 1
            if n5 == dp[i]:
                p5 += 1
        
        return dp[-1]
```

```java
class Solution {
    public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        dp[0] = 1;
        int p2 = 0, p3 = 0, p5 = 0;
        for (int i = 1; i < n; i++) {
            int cand2 = dp[p2] * 2;
            int cand3 = dp[p3] * 3;
            int cand5 = dp[p5] * 5;
            dp[i] = Math.min(Math.min(cand2, cand3), cand5);
            if (dp[i] == cand2) p2 += 1;
            if (dp[i] == cand3) p3 += 1;
            if (dp[i] == cand5) p5 += 1;
        }
        return dp[n - 1];

    }
}
```





## [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

```java
class Solution {
    public char firstUniqChar(String s) {
        Map<Character, Integer> hashmap = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (hashmap.get(s.charAt(i)) == null) hashmap.put(s.charAt(i), 0);
            hashmap.put(s.charAt(i), 1 + hashmap.get(s.charAt(i)));
        }

        for (int i = 0; i < s.length(); i++) {
            if (hashmap.get(s.charAt(i)) == 1) return s.charAt(i);
        }
        return ' ';
    }
}
```





## [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

+ 使用归并排序
+ 考虑合并两个有序数组时，同时它们之间的逆序对个数，见代码
+ 总的来说就是分治的思想，分成两边，左边的逆序对+右边的逆序对+左右交互的逆序对
+ 排序可以降低计算左右交互的逆序对的时间复杂度，本来需要逐个比较，排序后只需要遍历一遍

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        L = len(nums)
        temp = [0] * L 
        res = self.merge_sort(nums, temp, 0, L - 1)
        return res
        
    def merge_sort(self, nums, temp, l, r):
        # 结束, 返回0，即没有逆序对
        if l >= r:
            return 0
        mid = (l + r) // 2
        # 分别计算两边的逆序对个数，并使两边成为有序数组
        left_count = self.merge_sort(nums, temp, l, mid)
        right_count = self.merge_sort(nums, temp, mid + 1, r)
        # 两段已经有序,合并阶段
        # 使用temp暂存 l -- r
        i = l 
        j = mid + 1
        temp[l: r + 1] = nums[l: r + 1]
        count = 0
        for k in range(l, r + 1):
            # 左边已经遍历完, 直接把右边加入
            if i == mid + 1:
                nums[k] = temp[j]
                j += 1 
            # 右边遍历完, 直接把左边加入
            elif j == r + 1:
                nums[k] = temp[i]
                i += 1
            # 不是逆序对
            elif temp[i] <= temp[j]:
                nums[k] = temp[i]
                i += 1
            # 发现是逆序对, i - j
            # 此时i之后的都与j构成逆序对
            # 共 mid - i + 1 组
            else:
                nums[k] = temp[j]
                j += 1
                count += mid - i + 1
        # 合并完成，l -- r 有序
        # 三者相加
        return left_count + right_count + count
```



```java
class Solution {
    int[] nums;
    int[] temp;
    public int reversePairs(int[] nums) {
        this.nums = nums;
        this.temp = new int[nums.length];
        return merge_sort(0, nums.length - 1);
    }

    int merge_sort(int l, int r) {
        if (l >= r) {
            return 0;
        }
        int m = (l + r) / 2;
        int l_count = merge_sort(l, m);
        int r_count = merge_sort(m + 1, r);
        // 两段已经有序,合并阶段
        // 使用temp暂存 l -- r
        for (int i = l; i <= r; i++) {
            temp[i] = nums[i];
        }
        int i = l;
        int j = m + 1;
        int cross_count = 0;
        for (int k = l; k <= r; k++) {
            if (i == m + 1) {
                nums[k] = temp[j];
                j += 1;
            }else if (j == r + 1) {
                nums[k] = temp[i];
                i += 1;
            }else if (temp[i] <= temp[j]) {
                nums[k] = temp[i];
                i += 1; 
            }else {
                nums[k] = temp[j];
                j += 1;
                cross_count += m - i + 1;
            }
        }
        return l_count + r_count + cross_count;   
    }
}
```





## [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

+ 见链表160

```java
class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode node1 = headA;
        ListNode node2 = headB;
        while (node1 != node2) {
            node1 = (node1 == null ? headB : node1.next);
            node2 = (node2 == null ? headA : node2.next);
        }
        return node1;
    }
}
```



## [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

```java
class Solution {
    public int search(int[] nums, int target) {
        int end = bi_right(nums, target);
        int start = bi_right(nums, target - 1);
        Arrays.binarySearch()
        return end - start;
    }

    private int bi_right(int[] nums, int target) {
        int lo = 0;
        int hi = nums.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] <= target) {
                lo = mid + 1;
            }else {
                hi = mid;
            }
        }
        return lo;
    }
}
```



```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        right = self.bi_right(nums, target)
        left = self.bi_right(nums, target - 0.001)# 小技巧，减一个小数字
        return right - left

    def bi_right(self, nums, x):
        # 找到第一个比x大的数的下标
        # 范围  0 - len(nums)
        # 保证插入有序
        lo = 0
        hi = len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] <= x:
                lo = mid + 1
            else:
                hi = mid 
        return lo
```



+ 用bisect包

```python
import bisect
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        right = bisect.bisect(nums, target)
        left = bisect.bisect(nums, target - 0.001)
        return right - left
```







## [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

+ 二分查找第一个下标小于该位置数值的下标

```java
class Solution {
    public int missingNumber(int[] nums) {
        int i = 0, j = nums.length - 1;
        if (j == nums[j]) return j + 1;
        while (i < j) {
            int mid = (i + j) / 2;
            if (mid == nums[mid]) {
                i = mid + 1;
            }else {
                j = mid;
            }
        }
        return i;
    }
}
```







## [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

+ 二叉搜索 右中左 则是从大到小
+ 做 k 次

```python
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        # 右中左 第k个停止
        stack = []
        res = -1
        while k > 0:
            while root:
                stack.append(root)
                root = root.right
            top = stack.pop()
            res = top.val
            k -= 1
            root = top.left
        return res
```



**Java DFS**

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
    int k;
    int ans = -1;
    public int kthLargest(TreeNode root, int k) {
        this.k = k;
        DFS(root);
        return ans;
    }

    void DFS(TreeNode node) {
        if (node == null || k == 0) return;
        DFS(node.right);
        if (k == 0) return;
        ans = node.val;
        k--;
        DFS(node.left);
    }  
}
```



**Java 迭代**

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

    public int kthLargest(TreeNode root, int k) {
        LinkedList<TreeNode> stack = new LinkedList<>();
        int ans = -1;
        while (k > 0) {
            while (root != null) {
                stack.addLast(root);
                root = root.right;
            }
            TreeNode top = stack.removeLast();
            ans = top.val;
            k -= 1;
            root = top.left;
        }
        return ans;
    }
}
```





## [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        else:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

**Java**

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
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int ldepth = maxDepth(root.left);
        int rdepth = maxDepth(root.right);
        return 1 + Math.max(ldepth, rdepth);
    }
}
```



## [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

+ 求二叉树深度相同的思路
+ 若一个节点满足条件，则返回它的深度，不满足则直接返回-1， 此时整棵树都不满足，一路返回-1到root

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if self.helper(root) == -1:
            return False
        else:
            return True
 
    def helper(self, root):
        if root is None:
            return 0
        leftdepth = self.helper(root.left)
        if leftdepth == -1:
            return -1
        rightdepth = self.helper(root.right)
        if rightdepth == -1:
            return -1
        if abs(leftdepth - rightdepth) <= 1:
            return 1 + max(leftdepth, rightdepth)
        else:
            return -1
```



**Java**

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
    public boolean isBalanced(TreeNode root) {
        int dep = help(root);
        if (dep == -1) return false;
        return true;
    }

    int help(TreeNode node) {
        if (node == null) return 0;
        int ldepth = help(node.left);
        if (ldepth == -1) return -1;
        int rdepth = help(node.right);
        if (rdepth == -1) return -1;
        if (Math.abs(ldepth - rdepth) > 1) return -1;
        return 1 + Math.max(ldepth, rdepth);
    }
}
```



## [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

![image-20210416184051356](https://gitee.com/sxy22/note_images/raw/master/image-20210416184051356.png)



+ 一个数字和0 异或 不变
+ 举例： 检查一个数num的第三位 是否是0， 拿100 & num。 结果为0， 说明第三位为0， 结果为100， 说明第三位为1

```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        all_xor = 0
        # 一个数字和0 异或 不变
        for num in nums:
            all_xor = all_xor ^ num
        div = 1
        while all_xor & div == 0:
            div <<= 1

        a = 0
        b = 0
        for num in nums:
            if (num & div) != 0:
                a = a ^ num 
            else:
                b = b ^ num 
        return [a, b]
```



**Java**

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        int all_xor = 0;
        int diff = 1;
        int g0 = 0, g1 = 0;
        for (int num : nums) {
            all_xor = all_xor ^ num;  
        }
        while ((all_xor & diff) == 0) {
            diff = diff << 1;
        }
        for (int num : nums) {
            if ((num & diff) == 0) {
                g0 = g0 ^ num;
            }else {
                g1 = g1 ^ num;
            }
        }
        return new int[]{g0, g1};
    }
}
```





## [剑指 Offer 56 - II. 数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

+ 对于出现三次的数字，各 位 出现的次数都是 3 的倍数
+ 统计所有数字的各二进制位中1的出现次数，并对3求余

```java
class Solution {
    public int singleNumber(int[] nums) {
        int ans = 0;
        int[] counts = new int[32];
        for (int num : nums) {
            for (int i = 31; i >= 0; i--) {
                counts[i] += (num & 1);
                num >>= 1;
            }
        }
        for (int i = 0; i < 32; i++) {
            ans <<= 1;
            ans = ans | (counts[i] % 3);
        }
        return ans;
    }
}
```



![image-20211228011303714](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211228011303714.png)

![image-20211228011316439](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211228011316439.png)

![image-20211228011324727](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211228011324727.png)

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







## [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i = 0 
        j = len(nums) - 1
        while i < j:
            if nums[i] + nums[j] == target:
                return [nums[i], nums[j]]
            elif nums[i] + nums[j] > target:
                j -= 1
            else:
                i += 1
```



**Java**

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int i = 0, j = nums.length - 1;
        while (i < j) {
            int sum = nums[i] + nums[j];
            if (sum == target) {
                return new int[]{nums[i], nums[j]};
            }else if (sum < target) {
                i++;
            }else {
                j--;
            }
        }
        return new int[2];
    }
}
```



## [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

+ 滑动窗口的思路

```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        res = []
        # 从1 - 2开始
        i = 1
        j = 2
        end = (target + 1) // 2 
        # 计算i到j的sum，与target比较
        while i < j and j <= end:
            s = (i + j) * (j - i + 1) / 2
            if s < target:
                j += 1
            elif s > target:
                i += 1
            else:    
                res.append(list(range(i, j+1)))
                i += 1
                j += 1
            
        return res
```



**Java**

```java
class Solution {
    public int[][] findContinuousSequence(int target) {
        int i = 1, j = 2;
        ArrayList<int[]> ans = new ArrayList<>();
        while (i < j) {
            int sum = (i + j) * (j - i + 1) / 2;
            if (sum == target) {
                int[] range = new int[j - i + 1];
                for (int k = i; k <= j; k++) {
                    range[k - i] = k;
                }
                ans.add(range);
                i += 1;
                j += 1;
            }else if (sum < target) {
                j += 1;
            }else {
                i += 1;
            }
        }
        return ans.toArray(new int[0][]);

    }
}
```







## [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s += ' '
        word = []
        cur = ''
        for i in range(len(s)):
            if s[i] == ' ' :
                if len(cur) > 0:
                    word.append(cur)
                    cur = ''
            else:
                cur += s[i]
        return ' '.join(reversed(word))
```



## [剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        return s.substring(n, s.length()) + s.substring(0, n);
    }
}
```



## [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

+ 单调队列

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 0:
            return []
        queue = collections.deque()
        res = []
        for i in range(k):
            self.add(queue, nums[i])
        
        i =  0
        j = k - 1
        while j < len(nums):
            res.append(queue[0])
            # 删除 i 
            if queue[0] == nums[i]:
                queue.popleft()
            # 加入 j + 1
            if j + 1 < len(nums):
                self.add(queue, nums[j+1])
            i += 1
            j += 1
        return res 
    
    def add(self, queue, val):
        while len(queue) > 0 and queue[-1] < val:
            queue.pop()
        queue.append(val)
```



```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0) return new int[0];
        int[] ans = new int[nums.length - k + 1];
        LinkedList<Integer> queue = new LinkedList<>();
        for (int i = 0; i < k; i++) {
            qadd(queue, nums[i]);
        }

        int i = 0, j = k - 1;
        while (j < nums.length) {
            ans[i] = queue.peekFirst();
            // 删除 i 处
            if (queue.peekFirst() == nums[i]) {
                queue.removeFirst();
            }
            if (j + 1 < nums.length) {
                qadd(queue, nums[j + 1]);
            }
            i += 1;
            j += 1;
        }
        return ans;
    }

    void qadd(LinkedList<Integer> queue, int x) {
        while (!queue.isEmpty() && x > queue.peekLast()) {
            queue.removeLast();
        }
        queue.addLast(x);
    }
}
```



## [剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

```java
class MaxQueue {
    Deque<Integer> queue;
    Deque<Integer> max_queue;
    public MaxQueue() {
        queue = new LinkedList<>();
        max_queue = new LinkedList<>();
    }

    public int max_value() {
        if (queue.isEmpty()) {
            return -1;
        }
        return max_queue.getFirst();
    }

    public void push_back(int value) {
        queue.addLast(value);
        while (!max_queue.isEmpty() && max_queue.getLast() < value) {
            max_queue.removeLast();
        }
        max_queue.addLast(value);
    }

    public int pop_front() {
        if (queue.isEmpty()) {
            return -1;
        }
        int top = queue.removeFirst();
        if (top == max_queue.getFirst()) {
            max_queue.removeFirst();
        }
        return top;
    }
}
```





## [剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

![image-20211226220535085](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211226220535085.png)

```java
class Solution {
    public double[] dicesProbability(int n) {
        double[] pre = new double[6];
        Arrays.fill(pre, 1.0/6);
        int[] dose = {1, 2, 3, 4, 5, 6};
        for (int k = 2; k <= n; k++) {
            double[] cur = new double[5 * k + 1];
            for (int idx = 0; idx < cur.length; idx++) {
                int x = k + idx;
                for (int i : dose) {
                    if (x - i < k - 1) break;
                    if (x - i <= 6 * (k - 1)) {
                        cur[idx] += pre[x - i - k + 1] / 6; 
                    }
                }
            }
            pre = cur;
        }
        return pre;
     }
}
```



```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        pre = [1/6] * 6
        for k in range(2, n + 1):
            cur = [0] * (5 * n + 1)
            for i in range(len(cur)):
                x = k + i 
                for j in range(1, 7):
                    if x - j < k - 1:
                        break
                    if x - j <= 6 * (k - 1):
                        cur[i] += pre[x - j - k + 1] / 6
            pre = cur 
        return pre 
```



## [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

```java
class Solution {
    public boolean isStraight(int[] nums) {
        int max = -1, min = 14;
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (num == 0) continue;
            if (set.contains(num)) return false;
            set.add(num);
            max = Math.max(max, num);
            min = Math.min(min, num);
        }
        return (max - min) < 5;
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





## [剑指 Offer 63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices.length == 0) return 0;
        int min = prices[0];
        int max = 0;
        for (int i = 1; i < prices.length; i++) {
            max = Math.max(max, prices[i] - min);
            min = Math.min(min, prices[i]);
        }
        return max;

    }
}
```



## [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

Q ： 若数字 a 和 b 中有负数，则变成了减法，如何处理？
A ： 在计算机系统中，数值一律用 补码 来表示和存储。补码的优势： 加法、减法可以统一处理（CPU只有加法器）。因此，以上方法 同时适用于正数和负数的加法 

```java
class Solution {
    public int add(int a, int b) {
        while (b != 0) {
            int temp = a;
            a = a ^ b;
            b = (temp & b) << 1;
        }
        return a;
    }
}
```



## [剑指 Offer 66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

```java
class Solution {
    public int[] constructArr(int[] a) {
        int n = a.length;
        if (n == 0) return new int[0];
        int[] left = new int[n];
        int[] right = new int[n];
        int[] ans = new int[n];
        left[0] = 1;
        right[0] = 1;
        for (int i = 1; i < n; i++) {
            left[i] = left[i - 1] * a[i - 1];
            right[i] = right[i - 1] * a[n - i];
        }
        for (int i = 0; i < n; i++) {
            ans[i] = left[i] * right[n - i - 1];
        }
        return ans;
    }
}
```



## [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

+ 从上向下找
+ 当 p,q 都在 root 的 右子树 中，则遍历至 root.right
+ 否则， 都在 root的 左子树 中，则遍历至 root.left
+ 否则，说明找到了 最近公共祖先 ，跳出。

+ 迭代
+ 自顶向下的搜索

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        maxi = max(p.val, q.val)
        mini = min(p.val, q.val)
        while root:
            # 两个节点均在右子树
            if root.val < mini:
                root = root.right
            # 两个节点均在左子树
            elif root.val > maxi:
                root = root.left
            # 找到
            else:
                return root 
        return 
```

+ 递归

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        maxi = max(p.val, q.val)
        mini = min(p.val, q.val)
        # 两个节点均在右子树
        if root.val < mini:
            return self.lowestCommonAncestor(root.right, p, q)
        # 两个节点均在左子树
        elif root.val > maxi:
            return self.lowestCommonAncestor(root.left, p, q)
        # 找到
        else:
            return root 
```





## [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

**思路**

+ 空节点返回None

+ 当 root 等于 p,q，则直接返回 root
+ left, right 接受左右子节点的情况
  + left，right都为空，说明左右子树中 都没有pq，返回None
  + left，right都不为空，此时root为答案，返回root，会 一路 返回上去
  + left，right一个不为空，则返回不为空的那个

```python
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if root is None:
            return root
        if root == p or root == q:
            return root 
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left is not None and right is not None:
            return root 
        if left is None:
            return right
        else:
            return left
```

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
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        }else if (left == null) {
            return right;
        }else {
            return left;
        }
    }
}
```



`

