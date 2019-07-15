# 实验目的
练习pyspark kmeans的使用

# 实验原理
 
 

#### 使用pyspark.ml.clustering模块对商场顾客聚类 

#数据准备：
数据下载：
数据为kaggle上的关于商场客户的数据，地址：https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python

数据集很小，四个特征值：性别，年龄，收入能力，消费能力，这里我们用收入能力和消费能力两项对客户进行聚类处理

# 实验平台

- 操作系统： Ubuntu14.04
- Hadoop版本： 
- JDK版本：1.7
- Spark版本 2.0.2
- IDE：Eclipse


# 实验步骤

#### 一、启动HDFS和Yarn

```
# service ssh restart
# /usr/local/hadoop/sbin/start-dfs.sh
# /usr/local/hadoop/sbin/start-yarn.sh
````
#### 在终端输入jps查看启动情况
```
# jps
```
#### 一、修改pyspark的python本地环境
````
#echo 'export PYSPARK_DRIVER_PYTHON_OPTS="--pylab" '>> /usr/local/spark/conf/spark-env.sh
````
#### 打开pyspark shell

```
# cd /usr/local/spark
# bin/pyspark
```

#### 创建新的RDD:

````
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
ss = SparkSession(sc)
# 导入数据
df = ss.read.csv('file:///root/Desktop/myFile/Mall_Customers.csv', header=True, inferSchema=True)
# 更换列名
df = df.withColumnRenamed('Annual Income (k$)', 'Income').withColumnRenamed('Spending Score (1-100)', 'Spend')
# 看下数据
df.show(3)

+----------+------+---+------+-----+
|CustomerID|Gender|Age|Income|Spend|
+----------+------+---+------+-----+
|         1|  Male| 19|    15|   39|
|         2|  Male| 21|    15|   81|
|         3|Female| 20|    16|    6|
+----------+------+---+------+-----+
only showing top 3 rows

# 查看是否有缺失值
df.toPandas().isnull().sum()

CustomerID    0
Gender        0
Age           0
Income        0
Spend         0
dtype: int64
# 没有缺失值，都是0

#选取特征项，将特征项合并成向量
from pyspark.ml.feature import VectorAssembler
vecAss = VectorAssembler(inputCols = df.columns[3:], outputCol = 'features')
df_km = vecAss.transform(df).select('CustomerID', 'features')

df_km.show(3)
#看一下
+----------+-----------+
|CustomerID|   features|
+----------+-----------+
|         1|[15.0,39.0]|
|         2|[15.0,81.0]|
|         3| [16.0,6.0]|
+----------+-----------+
only showing top 3 rows
## 可视化数据看一下
# spark不方便提取列数据转换成pandas dataframe
pd_df = df.toPandas()

x = pd_df.Income
y = pd_df.Spend
plt.scatter(x, y)
plt.show()

````

#### 结果如下
![](https://kfcoding-static.oss-cn-hangzhou.aliyuncs.com/gitcourse-bigdata/20181116192247263_20190429115734.034.jpg) 


看上去是5类；是不是我们接下来看。。
KMeans k均值聚类
````
class pyspark.ml.clustering.KMeans(self, featuresCol="features", predictionCol="prediction", k=2, initMode="k-means||", initSteps=2, tol=1e-4, maxIter=20, seed=None)
````
 
参数
````
initMode: 初始化算法，可以使随机的“random"，也可以是”k-means||"
initSteps: k-means||初始化的步数，需>0
fit(datast,params=None)方法

````
model方法

````
cluster: 每个训练数据点预测的聚类中心数据框
clusterSize: 每个簇的大小（簇内数据点的个数）
k: 模型训练的簇个数
predictions: 由模型transform方法产生的数据框

````

````
plt.close()
from pyspark.ml.clustering import KMeans 
# k取 2-19 时 获取对应 cost 来确定 k 的最优取值
cost = list(range(2,20))
for k in range(2, 20):
    kmeans = KMeans(k=k, seed=1)
    km_model = kmeans.fit(df_km) 
    # computeCost:计算输入点与其对应的聚类中心之间的平方距离之和。
    cost[k-2] = km_model.computeCost(df_km) 
	

# 可视化
import matplotlib.pyplot as plt 
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(range(2,20), cost)
ax.set_xlabel('k')
ax.set_ylabel('cost') 

````

结果如下：
![](https://kfcoding-static.oss-cn-hangzhou.aliyuncs.com/gitcourse-bigdata/2018111619281716_20190429115824.024.jpg)

#### 可以见到在k=5时，出现了拐角，我们取k=5
````
# k=5 创建模型
kmeans = KMeans(k=5, seed=1)
km_model = kmeans.fit(df_km)
centers = km_model.clusterCenters()
# 集簇中心点
centers
[array([55.2962963 , 49.51851852]),
 array([25.72727273, 79.36363636]),
 array([86.53846154, 82.12820513]),
 array([88.2       , 17.11428571]),
 array([26.30434783, 20.91304348])]

# 获取聚类预测结果
transformed = km_model.transform(df_km).select('CustomerID', 'prediction')

# 合并表格
df_pred = df.join(transformed, 'CustomerID')

# 转化pandas dataframe 然后可视化
pd_df = df_pred.toPandas()
x = pd_df.Income
y = pd_df.Spend
plt.scatter(x, y,c=pd_df.prediction,cmap=plt.cm.rainbow)　
plt.show()　

````


结果如下：
![](https://kfcoding-static.oss-cn-hangzhou.aliyuncs.com/gitcourse-bigdata/20181116193252223_20190429115902.002.jpg)
