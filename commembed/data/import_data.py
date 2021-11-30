import os
import sys

from urls import get_urls

from multiprocessing.dummy import Pool
import random
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from tqdm import tqdm

from . import DATA_PATH

os.environ["JAVA_HOME"] = "/ada/projects/commembed/jdk-13.0.2"
os.environ["SPARK_HOME"] = "/ada/projects/commembed/spark-3.0.0-preview2-bin-without-hadoop"

spark = SparkSession.builder.getOrCreate()

#
# The data importer converts from compressed json files into parquet tables for efficient use with
# Spark.
#

result_filename = "%s.parquet" % sys.argv[1]

object_types = {
    "comments.parquet": ["comments"],
    "all_objects.parquet": ["submissions", "comments"],
    "submissions.parquet": ["submissions"],
    "comments_2019.parquet": ["comments"],
    "submissions_2019.parquet": ["submissions"]
}[result_filename]

# Note: do not include id, parent_id, these are automatically added later
comments_cols = [
        col("score").cast(IntegerType()),
        col("author_flair_css_class").cast(StringType()),
        col("author_flair_text").cast(StringType()),
        col("link_id").cast(StringType()),
        col("author").cast(StringType()),
        col("subreddit").cast(StringType()),
        regexp_replace(regexp_replace(regexp_replace(regexp_replace(col("body"), "&gt;", ">"), "&lt;", "<"), "&amp;", "&"),"\\r\\n","\n").cast(StringType()).alias("body"),
        col("edited").cast(IntegerType()),
        col("gilded").cast(StringType()),
        col("controversiality").cast(IntegerType()),
        col("created_utc").cast(IntegerType()),
        col("distinguished").cast(StringType())
    ]
submissions_cols = [
        col("author").cast(StringType()),
        col("author_flair_css_class").cast(StringType()),
        col("author_flair_text").cast(StringType()),
        col("created_utc").cast(IntegerType()),
        col("distinguished").cast(StringType()),
        col("domain").cast(StringType()),
        col("edited").cast(IntegerType()),
        col("gilded").cast(StringType()),
        col("is_self").cast(BooleanType()),
        col("over_18").cast(BooleanType()),
        col("score").cast(IntegerType()),
        regexp_replace(regexp_replace(regexp_replace(regexp_replace(col("selftext"), "&gt;", ">"), "&lt;", "<"), "&amp;", "&"),"\\r\\n","\n").cast(StringType()).alias("selftext"),
        col("title").cast(StringType()),
        col("url").cast(StringType()),
        col("subreddit").cast(StringType())
    ]
columns = {
    "all_objects.parquet": [
        col("created_utc").cast(IntegerType()),
        col("subreddit").cast(StringType()),
        col("author").cast(StringType()),
        col("score").cast(IntegerType())
    ],
    "comments.parquet": comments_cols,
    "comments_2019.parquet": comments_cols,
    "submissions.parquet": submissions_cols,
    "submissions_2019.parquet": submissions_cols
}[result_filename]

only_study_period = {
    "all_objects.parquet": False,
    "comments.parquet": True,
    "submissions.parquet": True,
    "comments_2019.parquet": "2019_only",
    "submissions_2019.parquet": "2019_only"
}[result_filename]



# by manually specifying the schema, we can avoid the initial pass over all the data to
# infer it (can be an incomplete schema w/ only data we need)
data_schemas = {
    "comments": "id STRING, parent_id STRING, score INTEGER, author_flair_css_class STRING, author_flair_text STRING, link_id STRING, author STRING, subreddit STRING, body STRING, edited INTEGER, gilded STRING, controversiality INTEGER, created_utc STRING, distinguished STRING",
    "submissions": "author STRING, author_flair_css_class STRING, author_flair_text STRING, created_utc STRING, distinguished STRING, domain STRING, edited INTEGER, gilded STRING, id STRING, is_self BOOLEAN, over_18 BOOLEAN, score INTEGER, selftext STRING, title STRING, url STRING, subreddit STRING"
}

for object_type in object_types:
    print("Loading %s ->" % object_type)

    columns_copy = list(columns)
    id_type = "t1_" if object_type == "comments" else "t3_"
    columns_copy.insert(0, (lit(None) if object_type == "submissions" else col("parent_id")).cast(StringType()).alias("parent_id"))
    columns_copy.insert(0, concat(lit(id_type), col("id")).cast(StringType()).alias("id"))

    filenames = list(get_urls(object_type, only_study_period=(only_study_period==True)))
    if only_study_period == "2019_only":
        filenames = [f for f in filenames if f[0][3:].startswith("2019")]
    
    print("\tCreating DF")
    df = spark.read.format("json").option("encoding", "UTF-8").schema(data_schemas[object_type]) \
        .load([("/ada/data/reddit/%s/%s" % (object_type, filename[0])) for filename in filenames])

    if result_filename == "comments.parquet":
        print("\tJoin and filter")
        # Filter to only active users.
        # Important to use broadcast join to avoid shuffle.
        df = df.join(broadcast(users), users.user == df.author, 'left')
        df = df.filter((df.is_bot == False) & (df.is_active == True))

    print("\tSelect")
    df = df.select(*columns_copy)

    print("\tWriting")
    df.write.mode("append").save(DATA_PATH + "/" + result_filename)
    print("\tDone")

print("All files imported.")
