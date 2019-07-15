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
#### 上传数据到hdfs
````
#hdfs dfs -put /root/Desktop/myFile/sample_libsvm_data.txt /tmp/
````
#### 修改pyspark的python本地环境
````
#echo 'export PYSPARK_PYTHON=/data/anaconda3/bin/python' >> /usr/local/spark/conf/spark-env.sh
#echo 'export PYSPARK_DRIVER_PYTHON_OPTS="--pylab" '>> /usr/local/spark/conf/spark-env.sh
````
#### 打开pyspark shell

```
# cd /usr/local/spark
# bin/pyspark
```

#### 创建新的RDD:

````
>>> 
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession

spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()


# Load the data stored in LIBSVM format as a DataFrame.
data = spark.read.format("libsvm").load("/tmp/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
# summary only
print(treeModel)
````

#### 执行action
````
`+----------+------------+--------------------+
|prediction|indexedLabel|            features|
+----------+------------+--------------------+
|       1.0|         1.0|(692,[95,96,97,12...|
|       1.0|         1.0|(692,[100,101,102...|
|       1.0|         1.0|(692,[121,122,123...|
|       1.0|         1.0|(692,[123,124,125...|
|       1.0|         1.0|(692,[124,125,126...|
+----------+------------+--------------------+
only showing top 5 rows

Test Error = 0.0285714 
DecisionTreeClassificationModel (uid=DecisionTreeClassifier_49f5b151055db55ad5a5) of depth 1 with 3 nodes

````
