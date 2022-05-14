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

