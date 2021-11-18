# 双指针 & 滑动窗口

## [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

**思路**

+ 使用两个指针表示字符串中的某个子串（或窗口）的左右边界，其中左指针代表着「枚举子串的起始位置」

+ 在每一步的操作中，我们会将左指针向右移动一格，表示 我们开始枚举下一个字符作为起始位置，然后我们可以不断地向右移动右指针，但需要保证这两个指针对应的子串中没有重复的字符。在移动结束后，这个子串就对应着 以左指针开始的，不包含重复字符的最长子串。记录下这个子串的长度



```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        L = len(s)
        hashset = set()
        i, j = 0, 0
        res = 0
        while i < L and j < L:
            while j < L and s[j] not in hashset:
                hashset.add(s[j])
                j += 1
            res = max(res, j - i)
            hashset.remove(s[i])
            i += 1
        return res
```



## [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)

**思路**

+ 排列，说明每个字符出现的次数相等，用hashmap记录
+ diff记录目前还有几个字符不同



```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        l1 = len(s1)
        l2 = len(s2)
        cnt1 = collections.defaultdict(int)
        cnt2 = collections.defaultdict(int)
        for s in s1:
            cnt1[s] += 1

        diff = len(cnt1)
        for i in range(l1):
            s = s2[i]
            if cnt1[s] == cnt2[s]:
                diff += 1
            if cnt1[s] - cnt2[s] == 1:
                diff -= 1
            cnt2[s] += 1
        if diff == 0:
            return True

        i = 0
        j = l1 - 1
        while j + 1 < l2:
            # 考虑加入j+1
            s = s2[j+1]
            if cnt1[s] == cnt2[s]:
                diff += 1
            if cnt1[s] - cnt2[s] == 1:
                diff -= 1
            cnt2[s] += 1
            # 考虑去除i
            s = s2[i]
            if cnt1[s] == cnt2[s]:
                diff += 1
            if cnt1[s] - cnt2[s] == -1:
                diff -= 1
            cnt2[s] -= 1

            if diff == 0:
                break
            i += 1
            j += 1

        if diff == 0:
            return True
        return False
```



## [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)

**思路**

+ left，right记录当前的位置， cnt记录每个字符出现的次数
+ max_count 记录当前子串中，出现最多的个数
+ 每次循环加入right位置的字符，使其出现次数+1
+ 如果right - left + 1 - max_count > k，则不符合要求，因为最多只能变k个字符，此时需要将left位置的字符次数-1， left +1
  + 此时并不需要更新max_count
  + 首先，虽然max_count不一定正确，但cnt中记录的次数是正确的
  + 当前的max_count不符合要求，一定需要更大的max_count才能符合更新ans的要求
  + 
+ 如果right - left + 1 - max_count > k, 符合要求，更新ans
+ 每一次都要right + 1，不需要保持right不变，使left +1 ，这样不会得到更大的ans



```python
class Solution:
    def characterReplacement(self, s, k):
        length = len(s)
        if length <= k:
            return length

        count = [0] * 26
        max_count = 0
        left, right = 0, 0
        ans = 0

        while right < length:
            w = s[right]
            count[ord(w) - ord('A')] += 1
            max_count = max(max_count, count[ord(w) - ord('A')])
            if right - left + 1 - max_count > k:
                count[ord(s[left]) - ord('A')] -= 1
                left += 1
            else:
                ans = max(ans, right - left + 1)
            right += 1

        return ans
```



## [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

+ 滑动窗口
+ 用distance记录当前字符串范围是否覆盖了t

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        L = len(t)
        cnt_t = collections.Counter(t)
        cnt_s = collections.Counter()
        distance = 0
        i, j = 0, 0
        left = 0
        right = len(s)
        if_exist = False
        while j < len(s) or distance == L:
            if distance < L:
                # 加入j
                w = s[j]
                if cnt_s[w] < cnt_t[w]:
                    distance += 1
                cnt_s[w] += 1
                j += 1
            else:
                # 记录
                if_exist = True
                if j - i < right - left:
                    left = i
                    right = j
                # 删除i
                # 此时已经是全覆盖的情况，cnt_s 中都是>= cnt_t
                w = s[i]
                if cnt_s[w] == cnt_t[w]: 
                    distance -= 1
                cnt_s[w] -= 1
                i += 1
        if if_exist is False:
            return ''
        return s[left: right]

```

