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

