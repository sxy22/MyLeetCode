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



## [182. 查找重复的电子邮箱](https://leetcode-cn.com/problems/duplicate-emails/)

![image-20211216203108580](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216203108580.png)

```mysql
# Write your MySQL query statement below
select 
    Email 
from Person
group by Email
having count(Id) > 1;
```



## [183. 从不订购的客户](https://leetcode-cn.com/problems/customers-who-never-order/)

![image-20211216203244080](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216203244080.png)



+ left join

```mysql
# Write your MySQL query statement below
select
    C.Name as Customers
from Customers as C
left join Orders as O 
on C.Id = O.CustomerId
where O.Id is Null;
```

+ 子查询

```mysql
select `Name` as Customers
from Customers
where Id not in (select CustomerId from Orders)
```



## [184. 部门工资最高的员工](https://leetcode-cn.com/problems/department-highest-salary/)

![image-20211216204818757](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216204818757.png)

![image-20211216204826370](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216204826370.png)

```mysql
select 
    d.Name as "Department", 
    e.Name as "Employee", 
    e.Salary
from Employee as e 
join (
    select DepartmentId, max(salary) as ms
    from Employee
    group by DepartmentId
) as T
on e.DepartmentId = T.DepartmentId and e.Salary = T.ms
join Department as d 
on e.DepartmentId = d.Id
```



## [185. 部门工资前三高的所有员工](https://leetcode-cn.com/problems/department-top-three-salaries/)

![image-20211216204846709](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216204846709.png)

![image-20211216204856309](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216204856309.png)

```mysql
# Write your MySQL query statement below
select 
    D.Name as Department ,
    T.Name as Employee,
    T.Salary
from (
    select 
        *,
        DENSE_RANK() over(partition by DepartmentId order by Salary DESC) as rk
    from Employee
) as T 
join Department as D 
on T.DepartmentId = D.Id 
where T.rk <= 3;
```



## [196. 删除重复的电子邮箱(blank)](https://leetcode-cn.com/problems/delete-duplicate-emails/)



## [197. 上升的温度](https://leetcode-cn.com/problems/rising-temperature/)

![image-20211216205050037](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216205050037.png)

![image-20211216205116693](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216205116693.png)

```mysql
select w1.id
from Weather as w1
join Weather as w2
on DATEDIFF(w1.RecordDate, w2.RecordDate) = 1
where w1.Temperature > w2.Temperature;
```



## [511. 游戏玩法分析 I](https://leetcode-cn.com/problems/game-play-analysis-i/)

![image-20211217210740533](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211217210740533.png)

![image-20211217210749485](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211217210749485.png)

```mysql
# Write your MySQL query statement below
select 
    player_id,
    min(event_date) as first_login
from Activity 
group by player_id;
```



## [512. 游戏玩法分析 II](https://leetcode-cn.com/problems/game-play-analysis-ii/)

![image-20211217210919062](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211217210919062.png)

```mysql
select A.player_id, A.device_id
from Activity as A
join (
    select player_id, min(event_date) as fd
    from Activity
    group by player_id
) as T
on A.player_id = T.player_id
and A.event_date = T.fd
```



+ window function

```mysql
# Write your MySQL query statement below
select
    T.player_id, 
    T.device_id
from (
    select 
        player_id,
        device_id,
        RANK() over(partition by player_id order by event_date) as rk
    from Activity
) as T 
where T.rk = 1;
```



## [534. 游戏玩法分析 III](https://leetcode-cn.com/problems/game-play-analysis-iii/)

+ window function

```mysql
# Write your MySQL query statement below
select 
    player_id, 
    event_date,
    SUM(games_played) over(partition by player_id order by event_date) as 'games_played_so_far'
from Activity;
```



## [550. 游戏玩法分析 IV](https://leetcode-cn.com/problems/game-play-analysis-iv/)

![image-20211217211549996](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211217211549996.png)

```mysql
# Write your MySQL query statement below
select 
    ROUND(count(A.player_id) / (select count(distinct player_id) from Activity) ,2) as fraction
from Activity as A 
join (
    select player_id,
    MIN(event_date) as first_date
    from Activity
    group by player_id
) as T
on A.player_id = T.player_id
and DATEDIFF(A.event_date, T.first_date) = 1;
```



## [570. 至少有5名直接下属的经理](https://leetcode-cn.com/problems/managers-with-at-least-5-direct-reports/)

![image-20211217212014696](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211217212014696.png)

```mysql
# Write your MySQL query statement below
select
    E1.Name
from Employee as E1
join Employee as E2
on E1.Id = E2. ManagerId
group by E1.Id
having count(E2.Id) >=5;
```



## [574. 当选者](https://leetcode-cn.com/problems/winning-candidate/)

![image-20211218201758825](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211218201758825.png)

```mysql
# Write your MySQL query statement below
select
    C.Name
from Vote as V
join Candidate as C 
on V.CandidateId = C.id
group by V.CandidateId
order by count(V.id) DESC
limit 1;
```



## [577. 员工奖金](https://leetcode-cn.com/problems/employee-bonus/)

![image-20211218204253679](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211218204253679.png)

```mysql
select `name`, bonus
from Employee as E
left join Bonus as B 
on E.empId = B.empId
where IFnull(B.bonus, 0) < 1000;

# case when
select E.name, B.bonus
from Employee as E
left join Bonus as B
on E.empId = B.empId
where (case
       when B.bonus is Null then 0
       when B.bonus < 1000 then 0
       else 1
       end) = 0;
```



## [578. 查询回答率最高的问题](https://leetcode-cn.com/problems/get-highest-answer-rate-question/)

![image-20211218205116473](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211218205116473.png)

```mysql
select 
    question_id as survey_log
from SurveyLog 
group by question_id
order by sum(if(action = 'answer', 1, 0)) / sum(if(action = 'show', 1, 0)) desc, question_id
limit 1;
```



## [579. 查询员工的累计薪水](https://leetcode-cn.com/problems/find-cumulative-salary-of-an-employee/)

![image-20211219194650303](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211219194650303.png)

```mysql
# Write your MySQL query statement below
select 
    E1.Id, 
    E1.Month, 
    SUM(E2.Salary) as Salary
from Employee as E1
join Employee as E2
on E1.Id = E2.Id
and E1.Month - E2.Month between 0 and 2
left join (select Id, max(Month) as MM from Employee group by Id) as T
on E1.Id = T.Id and E1.Month = T.MM
where T.Id is null
group by E1.Id, E1.Month
order by E1.Id, E1.Month DESC;
```



## [580. 统计各专业学生人数](https://leetcode-cn.com/problems/count-student-number-in-departments/)

![image-20211219194925811](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211219194925811.png)

![image-20211219194939735](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211219194939735.png)

![image-20211219194953272](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211219194953272.png)

```mysql
# Write your MySQL query statement below
select
    de.dept_name,
    count(st.student_id) as student_number
from department as de
left join student as st 
on de.dept_id = st.dept_id
group by de.dept_id
order by student_number DESC, de.dept_name;
```



## [584. 寻找用户推荐人](https://leetcode-cn.com/problems/find-customer-referee/)

![image-20211219195153698](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211219195153698.png)

```mysql
# Write your MySQL query statement below
select
    name 
from customer
where IFNULL(referee_id, -1) != 2;

select `name`
from customer
where ( case
        when referee_id = 2 then 1
        else 0
        end) = 0;
```



## [585. 2016年的投资](https://leetcode-cn.com/problems/investments-in-2016/)

![image-20211220211658807](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211220211658807.png)

```mysql
select round(sum(TIV_2016),2) as TIV_2016
from insurance
where PID not in (
            select PID from insurance
            group by TIV_2015
            having count(*) = 1)
and PID in (
            select PID from insurance
            group by LAT, LON
            having count(*) = 1
);
```



+ window func

```mysql
select 
    ROUND(sum(T.TIV_2016), 2) as TIV_2016
from (
    select
    *,
    count(PID) over(partition by TIV_2015) as cnt1,
    count(PID) over(partition by LAT, LON) as cnt2
    from insurance
) as T
where T.cnt1 >1 and T.cnt2 = 1;
```



## [586. 订单最多的客户](https://leetcode-cn.com/problems/customer-placing-the-largest-number-of-orders/)

![image-20211220212835065](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211220212835065.png)

```mysql
# Write your MySQL query statement below
select
    customer_number
from orders 
group by customer_number
order by count(order_number) DESC
limit 1;
```



## [596. 超过5名学生的课](https://leetcode-cn.com/problems/classes-more-than-5-students/)

![image-20211220213043680](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211220213043680.png)

```mysql
# Write your MySQL query statement below
select
    class 
from courses
group by class
having count(student) >= 5;
```



## [597. 好友申请 I：总体通过率](https://leetcode-cn.com/problems/friend-requests-i-overall-acceptance-rate/)

```mysql
select round( ifnull((select count(distinct requester_id, accepter_id) from request_accepted) / 
              (select count(distinct sender_id, send_to_id) from friend_request),0)
    ,2) as accept_rate
```



## [602. 好友申请 II ：谁有最多的好友](https://leetcode-cn.com/problems/friend-requests-ii-who-has-the-most-friends/)

![image-20211221203111893](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211221203111893.png)

```mysql
# Write your MySQL query statement below
select id, count(*) as num
from 
    (select  requester_id as id from request_accepted
    UNION ALL
    select  accepter_id as id from request_accepted) as T
group by id 
order by count(*) DESC
limit 1;
```



## [603. 连续空余座位](https://leetcode-cn.com/problems/consecutive-available-seats/)

![image-20211221203718619](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211221203718619.png)

```java
# Write your MySQL query statement below

select
    distinct s1.seat_id as seat_id
from cinema as s1 
join cinema as s2
on abs(s1.seat_id - s2.seat_id) = 1
where s1.free = 1 and s2.free = 1
order by seat_id;
```





## [608. 树节点](https://leetcode-cn.com/problems/tree-node/)

![image-20211221205013748](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211221205013748.png)

```mysql
# Write your MySQL query statement below

select 
    t1.id,
    (case
    when t1.p_id is null then 'Root'
    when count(t2.id) = 0 then "Leaf"
    else "Inner"
    end) as `Type`
from tree as t1
left join tree as t2
on t1.id = t2.p_id
group by t1.id;
```



## [614. 二级关注者](https://leetcode-cn.com/problems/second-degree-follower/)

![image-20211222204053595](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211222204053595.png)

```mysql
# Write your MySQL query statement below
select
    f1.follower,
    count(distinct f2.follower) as num
from follow as f1
join follow as f2
on f1.follower = f2.followee
group by f1.follower
order by f1.follower;
```



## [619. 只出现一次的最大数字](https://leetcode-cn.com/problems/biggest-single-number/)

![image-20211222204845763](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211222204845763.png)

```mysql
# Write your MySQL query statement below
select (
    select
        num
    from MyNumbers 
    group by num 
    having count(num) = 1
    order by num DESC 
    limit 1
) as num;
```



## [620. 有趣的电影](https://leetcode-cn.com/problems/not-boring-movies/)

![image-20211222205337321](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211222205337321.png)

```mysql
# Write your MySQL query statement below
select
    *
from cinema
where description != "boring"
and id % 2 = 1
order by rating DESC;
```



## [627. 变更性别](https://leetcode-cn.com/problems/swap-salary/)

![image-20211223204710871](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211223204710871.png)

```mysql
# Write your MySQL query statement below

update Salary
set sex = IF(sex = "f", "m", "f");
```



## [1045. 买下所有产品的客户](https://leetcode-cn.com/problems/customers-who-bought-all-products/)

![image-20211223204957955](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211223204957955.png)

```mysql
select 
    customer_id
from Customer 
group by customer_id
having count(distinct product_key) = (select count(*) from Product);
```



## [1050. 合作过至少三次的演员和导演](https://leetcode-cn.com/problems/actors-and-directors-who-cooperated-at-least-three-times/)

![image-20211223205135215](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211223205135215.png)

```mysql
# Write your MySQL query statement below
select 
    actor_id, director_id
from ActorDirector 
group by actor_id, director_id
having count(`timestamp`) >= 3;
```



## [1077. 项目员工 III](https://leetcode-cn.com/problems/project-employees-iii/)

![image-20211224210153146](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211224210153146.png)

```mysql
# Write your MySQL query statement below
select
    T.project_id, 
    T.employee_id
from (
    select
        P.project_id, 
        P.employee_id, 
        DENSE_RANK() over(partition by P.project_id order by E.experience_years DESC) as 'rk'
    from Project as P 
    join Employee as E 
    on P.employee_id = E.employee_id
) as T 
where T.rk = 1;
```



## [1082. 销售分析 I(group window) ](https://leetcode-cn.com/problems/sales-analysis-i/)

![image-20211224210951272](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211224210951272.png)

```mysql
# Write your MySQL query statement below
select
    T.seller_id
from (
    select
        seller_id,
        DENSE_RANK() over(order by SUM(price) DESC) as rk
    from Sales
    group by seller_id    
) as T 
where T.rk = 1;
```



## [1083. 销售分析 II](https://leetcode-cn.com/problems/sales-analysis-ii/)

![image-20211224211301178](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211224211301178.png)

```mysql
# Write your MySQL query statement below

select 
    s.buyer_id
from Sales as s
join Product as p 
where s.product_id = p.product_id
group by s.buyer_id
having SUM(IF(p.product_name = "S8", 1, 0)) > 0
and SUM(IF(p.product_name = "iPhone", 1, 0)) = 0;
```



## [1084. 销售分析III](https://leetcode-cn.com/problems/sales-analysis-iii/)

![image-20211224211535638](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211224211535638.png)

```mysql
# Write your MySQL query statement below
select 
    p.product_id, 
    p.product_name
from Product as p
join Sales as s 
on p.product_id = s.product_id
group by p.product_id
having sum(s.sale_date not between '2019-01-01' and '2019-03-31') = 0;
```

