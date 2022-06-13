# 牛客 shell



## SHELL1 统计文件的行数            

写一个 bash脚本以输出一个文本文件 nowcoder.txt中的行数

```shell
#!/bin/bash

# 先cat 再传给wc
line=$(cat ./nowcoder.txt | wc -l)
echo $line

# 重定向
```



# 命令记录

## wc

利用wc指令我们可以计算文件的Byte数、字数、或是列数。

若不指定文件名称、或是所给予的文件名为"-"，则wc指令会从标准输入设备读取数据。



```sql
wc [-clw][--help][--version][文件...]


```

- -c或--bytes或--chars   只显示Bytes数。
- -l或--lines   显示行数。
- -w或--words   只显示字数。
- --help   在线帮助。
- --version   显示版本信息。