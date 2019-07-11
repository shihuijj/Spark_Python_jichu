# 实验目的
练习pyspark的使用

# 实验原理
 
 

#### python Spark 
- pyspark是由spark项目提供的python命令行，可以接受python代码并转换成spark分布式i任务。

# 实验平台

- 操作系统： Ubuntu14.04
- Hadoop版本：2.7.4
- JDK版本：1.7
- Spark版本 2.0.2
- IDE：Eclipse


# 实验步骤

#### 一、启动HDFS和Yarn

```
# service ssh restart
# /usr/local/hadoop/sbin/start-dfs.sh
# /usr/local/hadoop/sbin/start-yarn.sh
```
在终端输入jps查看启动情况
```
# jps
```
####修改pyspark的python本地环境
````
#echo 'export PYSPARK_DRIVER_PYTHON_OPTS="--pylab" '>> /usr/local/spark/conf/spark-env.sh
````
#### 打开pyspark shell

```
# cd /usr/local/spark
# bin/pyspark
```

####创建新的RDD:

````
>>> textFile = sc.textFile("file:///usr/local/spark/README.md")
````

####执行action

````
>>> textFile.count()  # 计数，返回RDD中items的个数，这里就是README.md的总行 数#
99
>>> textFile.first()  # RDD中的第一个item，这里就是文件README.md的第一行
u'# Apache Spark'

````

#### 使用filter
````
>>> linesWithSpark = textFile.filter(lambda line: "Spark" in line)
````

#### transform 和 action 结合
````
>>> textFile.filter(lambda line: "Spark" in line).count()  # 有多好行含有“Spark”这一字符串
19
````
#### 使用lamda 和 reduce

````
>>> textFile.map(lambda line: len(line.split())).reduce(lambda a, b: a if (a>b) else b)
22

````
####利用RDD的动作和转换能够完成复杂的计算（1)
 - map函数将len(line.split())这一语句在所有line上执行，返回每个line所含有的单词个数，也就是将line都map到一个整数值，然后创建一个新的RDD。然后调用reduce，找到最大值。map和reduce函数里的参数是python中的匿名函数（lambda），事实上，我们这里也可以传递python中更顶层的函数。比如，我们先定义一个比较大小的函数，这样我们的代码会更容易理解：:

````
>>> def max(a, b):
. . .     if a > b:
. . .         return a
. . .     else:
. . .         return b
. . .
>>> textFile.map(lambda line: len(line.split())).reduce(max)

````

####利用RDD的动作和转换能够完成复杂的计算（2)
 - flatMap, map和reduceByKey三个转换，计算文件README.md中每个单词出现的个数，并返回一个新的RDD，每个item的格式为(string, int)，即单词和对应的出现次数
````
>>> wordCounts = textFile.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
````

####利用RDD的动作和转换能够完成复杂的计算（3)
 - flatMap(func)：与map相似，但是每个输入的item能够被map到0个或者更多的输出items上，也就是说func的返回值应当是一个Seq，而不是一个单独的item，上述语句中，匿名函数返回的就是一句话中所含的每个单词。

 - ReduceByKey(func)：可以作用于使用“键-值”(K, V)形式存储的数据集上并返回一组新的数据集(K, V)，其中，每个键的值为聚合使用func操作的结果，这里相当于python中字典的含义。上述语句中，相当于当某个单词出现一次时，就在这个单词的出现次数上加1，每个单词就是一个Key，reduceByKey中的匿名函数计算单词的出现次数。
 - collect这一动作收集上述语句的计算结果

````
>>> wordCounts.collect()
[(u'when', 1), (u'R,', 1), (u'including', 3), (u'computation', 1), ...]

````

####缓存Caching
 - Spark也支持将数据集存入集群范围的内存缓存中。这对于需要进行重复访问的数据非常有用，比如我们需要在一个小的数据集中执行查询操作，或者需要执行一个迭代算法（例如PageRank）。下面，利用之前命令中得到的linesWithSpark数据集，演示缓存这一操作过程

````
>>> linesWithSpark.cache()
PythonRDD[26] at RDD at PythonRDD.scala:48
>>> linesWithSpark.count()
19
>>> linesWithSpark.count()
19

````