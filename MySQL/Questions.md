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

