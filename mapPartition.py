import pyspark
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import StringType, FloatType, StructField, StructType

import time
import bert_get_entity

conf = SparkConf()
conf.set("spark.driver.memory", "2g")
conf.set("spark.executor.memory", "8g")
# conf.set("spark.executor.instances", "2")
# conf.set("spark.sql.shuffle.paritions", "10")

# conf.set("spark.default.parallelism", "500")
conf.set("spark.executor.cores", "6")
# conf.set("spark.num.executor", "10")
# conf.set("spark.archives", "http://192.168.137.89/pyspark_venv.tar.gz#environment")
# conf.set("spark.archives", "/KgSpark/bert-base-chinese.zip#/KgSpark/bert-base-chinese")

spark = SparkSession \
    .builder \
    .master("spark://192.168.137.200:7077") \
    .appName("mapPartition-test") \
    .config(conf=conf) \
    .getOrCreate()

# 傳遞 bert_model.py 到工作節點

sc = spark.sparkContext


qa_dataset_path = "/KgSpark/qa_tseting_dataset.csv"

df = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .load(qa_dataset_path).limit(100)
# df = df.repartition(2)
print(df['question'])

df_test = spark.createDataFrame(data = df.rdd , schema = df.schema)
df_test = df_test.repartition(4)

# 測試原始數據
# data = [('James','Smith','M',3000),
#   ('Anna','Rose','F',4100),
#   ('Robert','Williams','M',6200),
#   ('James1','Smith','M',3000),
#   ('Anna1','Rose','F',4100),
#   ('Robert1','Williams','M',6200),
#   ('James2','Smith','M',3000),
#   ('Anna2','Rose','F',4100),
#   ('Robert2','Williams','M',6200),
# ]

# columns = ["firstname","lastname","gender","salary"]
# df_test = spark.createDataFrame(data=data, schema = columns)
# df_test = df_test.repartition(4)
# df_test.show()

print("Number of Partitions: " + str(df_test.rdd.getNumPartitions()))
print("Action: First element: " + str(df_test.rdd.first()))
print("Count: " + str(df_test.rdd.count()))

def reformat(partitionData):
    for row in partitionData:
        print(row)
        yield [row.question, row.answer, "test"]
        # yield [row.firstname+","+row.lastname,row.salary*10/100]

df2 = df_test.rdd.mapPartitions(reformat).toDF(["question", "answer", "prediction"])
df2.show(50)
