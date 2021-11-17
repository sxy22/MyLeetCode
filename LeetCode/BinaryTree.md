# 二叉树

##  先序遍历 迭代&递归

```python
#递归
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        self.res = []
        self.preorder(root)
        return self.res

    def preorder(self, t):
        if t is None:
            return
        self.res.append(t.val)
        self.preorder(t.left)
        self.preorder(t.right)
        
#迭代
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []

        res = []
        stack = [root]

        while stack:
            root = stack.pop()
            while root:
                res.append(root.val)
                if root.right:
                    stack.append(root.right)
                root = root.left    
        return res
```



## 中序遍历 迭代&递归

```python
# 递归
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        self.res = []
        self.mid(root)
        return self.res 

    def mid(self, node):
        if node is None:
            return
        self.mid(node.left)
        self.res.append(node.val)
        self.mid(node.right)
# 迭代
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        #迭代
        res = []
        stack = []
        while root or len(stack) != 0:
            while root:
                stack.append(root)
                root = root.left
            top = stack.pop()
            res.append(top.val)
            root = top.right
        return res 
```



## [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        queue = collections.deque()
        queue.append(root)
        ans = []
        while queue:
            cur_layer = []
            for _ in range(len(queue)):
                node = queue.popleft()
                cur_layer.append(node.val)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            ans.append(cur_layer)
        return ans 
```



## [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

+ diameter = 左子树深度加右子树深度，最长路不一定经过根节点，要遍历每一个节点
+ 深度优先

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.res = 0
        d = self.depth(root)
        return self.res 
    
    def depth(self, Node):
        if Node is None:
            return 0
        else:
            L = self.depth(Node.left)
            R = self.depth(Node.right)
            dia = L + R
            self.res = max(self.res, dia)
            return 1 + max(L, R)
```



## [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

见剑指offer07



## [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```java
class Solution {
    HashMap<Integer, Integer> map;
    int[] postorder;
    int[] inorder;
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        this.postorder = postorder;
        this.inorder = inorder;
        this.map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return this.recur(0, inorder.length - 1, 0, postorder.length - 1);

    }

    TreeNode recur(int in_left, int in_right, int post_left, int post_right) {
        if (post_left > post_right) {
            return null;
        }
        int head = postorder[post_right];
        int in_mid = map.get(head);
        // 注意mid的计算
        int post_mid = post_left + in_mid - in_left - 1;
        TreeNode root = new TreeNode(head);
        root.left = this.recur(in_left, in_mid - 1, post_left, post_mid);
        root.right = this.recur(in_mid + 1, in_right, post_mid + 1, post_right - 1);
        return root;
    }
}
```



## [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

**思路**

+ 空节点返回None

+ 当 root 等于 p,q，则直接返回 root
+ left, right 接受左右子节点的情况
  + left，right都为空，说明左右子树中 都没有pq，返回None
  + left，right都不为空，此时root为答案，返回root，会 一路 返回上去
  + left，right一个不为空，则返回不为空的那个

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
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
        if right is None: 
            return left
```

