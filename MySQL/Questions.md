# LeetCode 数据库

## [175. 组合两个表](https://leetcode-cn.com/problems/combine-two-tables/)

<img src="https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211213221543084.png" alt="image-20211213221543084" style="zoom:80%;" />

编写一个 SQL 查询，满足条件：无论 person 是否有地址信息，都需要基于上述两表提供 person 的以下信息：

FirstName, LastName, City, State

```mysql
# Write your MySQL query statement below
select P.FirstName, P.LastName, A.City, A.State
from Person as P 
left join Address as A 
on P.PersonId = A.PersonId;
```



## [176. 第二高的薪水](https://leetcode-cn.com/problems/second-highest-salary/)

![image-20211213221901837](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211213221901837.png)

```mysql
select ifnull((
    select distinct Salary
    from Employee 
    order by Salary DESC
    limit 1, 1
), null) as SecondHighestSalary;
```

```mysql
select max(Salary) as SecondHighestSalary 
from Employee
where Salary < (select max(Salary) from Employee);
```



## [178. 分数排名(window)](https://leetcode-cn.com/problems/rank-scores/)

![image-20211215213404018](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211215213404018.png)

![image-20211215213414097](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211215213414097.png)

+ 常规，统计大于等于改行的distinct个数，就是排名

```mysql
select 
    s1.Score,
    count(distinct s2.Score) as `Rank`
from Scores as s1 
join Scores as s2 
on s1.Score <= s2.Score
group by s1.Id
order by  `Rank`;
```

+ window function DENSE_RANK

```mysql
# Write your MySQL query statement below
select 
    Score,
    DENSE_RANK() over(order by Score DESC) as `RANK`
from Scores
order by `RANK`;
```



## [180. 连续出现的数字](https://leetcode-cn.com/problems/consecutive-numbers/)

![image-20211215214202832](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211215214202832.png)

```mysql
# Write your MySQL query statement below
select distinct L1.Num as ConsecutiveNums
from `Logs` as L1
join `Logs` as L2
on abs(L1.Id - L2.Id) <=1
and L1.Num = L2.Num
group by L1.Id
having count(L2.Id) = 3;
```



## [181. 超过经理收入的员工](https://leetcode-cn.com/problems/employees-earning-more-than-their-managers/)

![image-20211215214434481](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211215214434481.png)

```mysql
select
    e1.Name as Employee
from Employee as e1 
join Employee as e2
on e1.ManagerId = e2.Id 
where e1.Salary > e2.Salary;
```

