# 链表

## [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

![image-20210413111057040](https://gitee.com/sxy22/note_images/raw/master/image-20210413111057040.png)

+ 若有相同节点，一定会在走完后同时到达
+ 若没有相同节点，则node1 node2都走到None

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node1 = headA
        node2 = headB 
        while node1 != node2:
            if node1 is None:
                node1 = headB
            else:
                node1 = node1.next
            if node2 is None:
                node2 = headA
            else:
                node2 = node2.next
        return node1
```



## [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

**思路**

+ pre记录前一个节点
+ cur记录当前节点，是cur指向pre

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None 
        cur = head 
        while cur is not None:
            # 记录cur.next
            next_cur = cur.next
            cur.next = pre
            pre = cur 
            cur = next_cur
        # 最后cur -> None, pre -> 最后的node， 返回pre
        return pre 
```



## [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

**思路1**

+ 反转指定区间，再插入

```python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        pre_head = ListNode(-1)
        pre_head_copy = pre_head
        pre_head.next = head 
        l = n - m
        # 使pre_head 指向反转区间的前一个节点
        while m - 1 > 0:
            pre_head = pre_head.next
            m -= 1
        # rev_head指向反转的第一个节点，rev_end 指向反转最后一个节点
        rev_head = pre_head.next
        rev_end = rev_head
        while l > 0:
            rev_end = rev_end.next
            l -= 1
        # 反转rev_head, pre为rev_end.next, 即反转区间后的节点
        # 反转结果接在pre_head 后面
        pre = rev_end.next
        rev_end.next = None
        pre_head.next = self.revList(rev_head, pre)
        return pre_head_copy.next
         
    def revList(self, head, pre):
        cur = head 
        while cur is not None:
            next_cur = cur.next
            cur.next = pre
            pre = cur 
            cur = next_cur
        return pre 
```



**思路2 一次遍历**

![image-20210318115052245](https://gitee.com/sxy22/note_images/raw/master/image-20210318115052245.png)

+ cur指向当前节点，其下一个节点需要被摘除并插入到pre之后

```python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        pre_head = ListNode(-1)
        pre = pre_head
        pre_head.next = head 
        
        # 使pre指向反转区间的前一个节点
        for _ in range(m - 1):
            pre = pre.next

        cur = pre.next
        # 需操作 n - m 次
        for _ in range(n - m):
            # 摘除下一个节点
            next_node = cur.next
            cur.next = cur.next.next
            # 插入到pre后
            next_node.next = pre.next
            pre.next = next_node
        return pre_head.next
```



## [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

**思路**

+ 快慢指针
+ 快指针一次走两步，慢指针一次走一步，若有环，则快指针一定会在某一节点追上慢指针



```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if head is None or head.next is None:
            return False
        slow = head
        fast = head.next

        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next.next
        
        return True
```



