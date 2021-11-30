from pyspark.sql import SparkSession
import os

DATA_PATH = "/ada/data/reddit/parquet"

os.environ["JAVA_HOME"] = "/ada/projects/commembed/jdk-13.0.2"
os.environ["SPARK_HOME"] = "/ada/projects/commembed/spark-3.0.0-preview2-bin-without-hadoop"

spark = SparkSession.builder.getOrCreate()

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

already_loaded_tables = {}

def spark_context():

    print("Spark WebUI: %s" % spark.sparkContext.uiWebUrl)

    return spark

def load(table_name):

    if table_name in already_loaded_tables:
        return already_loaded_tables[table_name]

    print("(Freshly loading table %s)" % table_name)
    df = spark.read.load("/ada/data/reddit/parquet/%s.parquet" % table_name)
    df.cache()
    df.createOrReplaceTempView(table_name)
    already_loaded_tables[table_name] = df
    return df

