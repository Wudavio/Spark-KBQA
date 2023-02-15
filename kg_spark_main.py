import pyspark
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import StringType, FloatType, StructField, StructType
from pyspark.sql.functions import broadcast
from pyspark import TaskContext
import time
import bert_get_entity
import sim_main
import pandas as pd

def spark_config():
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.default.parallelism", "1000")
    conf.set("spark.executor.cores", "3") # 每個executor核心數量
    conf.set("spark.executor.instances", "4") # executor數量
    conf.set("spark.task.cpus", "3") # 每個任務可以使用的核心數量
    conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    return conf

spark = SparkSession \
    .builder \
    .master("spark://192.168.137.201:7077") \
    .appName("kg-spark-test") \
    .config(conf=spark_config()) \
    .getOrCreate()


sc = spark.sparkContext
sc.addPyFile("/KgSpark/qa_tseting_dataset.csv")
sc.addPyFile("/KgSpark/bertcrf.py")
sc.addPyFile("/KgSpark/ner_main.py")
sc.addPyFile("/KgSpark/sim_main.py")
sc.addPyFile("/KgSpark/bert_get_entity.py")

def text_match(attribute_list, answer_list, sentence):

    assert len(attribute_list) == len(answer_list)

    idx = -1
    for i, attribute in enumerate(attribute_list):
        if str(attribute) in str(sentence):
            idx = i
            break

    if idx != -1:
        return attribute_list[idx], answer_list[idx]

    return "",""

def load_qa_dataset():
    qa_dataset_path = "/KgSpark/qa_tseting_throughput_dataset.csv"
    qa_dataset = spark \
        .read \
        .format("csv") \
        .option("header", "true") \
        .load(qa_dataset_path)
        # .limit(300)
    return qa_dataset

qa_dataset = load_qa_dataset()
qa_dataset_df = spark.createDataFrame(data = qa_dataset.rdd , schema = qa_dataset.schema)
qa_dataset_df = qa_dataset_df.repartition(10)
# qa_dataset_df.show()

ner_model = bert_get_entity.get_ner_model()
ner_model_bcast = sc.broadcast(ner_model)

sim_mdoel = sim_main.get_sim_model()
sim_mdoel_bcast = sc.broadcast(sim_mdoel)

print("Number of Partitions: "+str(qa_dataset_df.rdd.getNumPartitions()))
# print("Action: First element: "+str(qa_dataset_df.rdd.first()))
print("Count: "+str(qa_dataset_df.rdd.count()))

def kbqaFlow(partitionIndex, partitionData):

    print(f"partitionId: {partitionIndex}")
    # 在分區中先將模型與DB載入
    model = ner_model_bcast.value
    sim_model = sim_mdoel_bcast.value
    kgdb = pd.read_csv('/KgSpark/knowledge_db_1G_split.csv')
    partition_start = time.time()
    for row in partitionData:
        # get entity
        entity = bert_get_entity.get_entity(model, row.question)
        # get kgdb attribute
        triple_list = []
        # pick = kgdb.loc[kgdb['entity'].str.startswith(entity)]
        pick = kgdb.loc[kgdb['entity_split'] == entity]
        triple_list = pick.values.tolist()

        if not entity or entity == "":
            continue

        # 第一次判斷(從Kgdb搜尋相關實體的屬性與答案)
        if len(triple_list) == 0:
            answer = "找不到實體: '{}' attribute".format(entity)
            continue

        # 第二次判斷 (attribute從question逐字掃描)
        triple_list = list(zip(*triple_list))
        attribute_list = triple_list[2]
        answer_list = triple_list[3]
        attribute, answer = text_match(attribute_list, answer_list, row.question)

        # 第三次判斷 (語義相似度比對)
        if attribute and answer:
            answer = answer
        else:
            #去除實體
            # raw_text = raw_text.replace(entity, '')
            attribute_idx = sim_main.semantic_match(sim_model, row.question, attribute_list, answer_list, 128).item()

            if attribute_idx == -1:
                answer = ''
            else:
                attribute = attribute_list[attribute_idx]
                answer = answer_list[attribute_idx]
        # print(f'{row.question} , {row.answer}, {entity}, {answer}')
        yield [row.question, row.answer, entity, answer]

    partition_end = time.time()
    print(f"執行時間: {(partition_end - partition_start)} 秒")


start = time.time()
# qa_dataset_df.rdd.mapPartitions(kbqaFlow).toDF(["question", "real_answer", "enitiy", "prediction_answer"]).collect()
qa_dataset_df.rdd.mapPartitionsWithIndex(kbqaFlow).collect()

end = time.time()
print("執行時間：%f 秒" % (end - start))

sc.stop()