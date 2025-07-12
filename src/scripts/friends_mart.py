import findspark
findspark.init()
findspark.find()
import os

os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['JAVA_HOME']='/usr'
os.environ['SPARK_HOME'] ='/usr/lib/spark'
os.environ['PYTHONPATH'] ='/usr/local/lib/python3.8'

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from datetime import datetime, timedelta

from pyspark.sql.functions import count, row_number, desc
from pyspark.sql.window import Window
import sys

from pyspark.sql.functions import explode, col, countDistinct
from pyspark.sql.functions import lit

spark = SparkSession.builder \
    .appName("Friends Mart") \
    .master("local") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "2g") \
    .config("spark.driver.cores", "1") \
    .config("spark.executor.instances", "5") \
    .getOrCreate()

# Данные
events = spark.read.parquet("/user/master/data/geo/events")
geo = spark.read.option("header", True).option("sep", ";").csv("/user/marbiana/data/geo.csv")

events = events.withColumn("lat_rad", F.radians(F.col("lat"))) \
               .withColumn("lon_rad", F.radians(F.col("lon")))
geo = geo.withColumn("city_lat_rad", F.radians(F.col("lat").cast("double"))) \
         .withColumn("city_lon_rad", F.radians(F.col("lng").cast("double")))

events = events.withColumn("date", F.to_date("event.datetime"))

# Зона
joined = events.crossJoin(geo)
distance = 2 * 6371 * F.asin(F.sqrt(
    F.pow(F.sin((F.col("lat_rad") - F.col("city_lat_rad")) / 2), 2) +
    F.cos(F.col("lat_rad")) * F.cos(F.col("city_lat_rad")) *
    F.pow(F.sin((F.col("lon_rad") - F.col("city_lon_rad")) / 2), 2)
))
joined = joined.withColumn("distance", distance)
w = Window.partitionBy("event.message_id").orderBy("distance")
zone_events = joined.withColumn("rn", F.row_number().over(w)).filter("rn = 1").withColumnRenamed("city", "zone_id")

# Подписки
subs = zone_events.filter(F.col("event_type") == "subscription").select(
    F.col("event.subscription_user").alias("user_id"),
    F.col("event.subscription_channel").alias("channel_id"),
    "zone_id", "lat_rad", "lon_rad"
)

# Пары
pairs = subs.alias("a").join(
    subs.alias("b"),
    (F.col("a.channel_id") == F.col("b.channel_id")) &
    (F.col("a.zone_id") == F.col("b.zone_id")) &
    (F.col("a.user_id") < F.col("b.user_id"))
)

pairs = pairs.withColumn(
    "distance",
    2 * 6371 * F.asin(F.sqrt(
        F.pow(F.sin((F.col("a.lat_rad") - F.col("b.lat_rad")) / 2), 2) +
        F.cos(F.col("a.lat_rad")) * F.cos(F.col("b.lat_rad")) *
        F.pow(F.sin((F.col("a.lon_rad") - F.col("b.lon_rad")) / 2), 2)
    ))
).filter("distance <= 1.0")

# Исключаем переписку
messages = zone_events.filter(F.col("event_type") == "message").select(
    F.col("event.message_from").alias("from_user"),
    F.col("event.message_to").alias("to_user")
).distinct()

pairs_clean = pairs.join(
    messages,
    ((F.col("a.user_id") == F.col("from_user")) & (F.col("b.user_id") == F.col("to_user"))) |
    ((F.col("a.user_id") == F.col("to_user")) & (F.col("b.user_id") == F.col("from_user"))),
    "left_anti"
)

friends = pairs_clean.select(
    F.least(F.col("a.user_id"), F.col("b.user_id")).alias("user_left"),
    F.greatest(F.col("a.user_id"), F.col("b.user_id")).alias("user_right"),
    "a.zone_id",
    F.current_timestamp().alias("processed_dttm")
)

friends.coalesce(1).write.mode("overwrite").parquet("/user/marbiana/data/friends_recommendations")

spark.stop()
