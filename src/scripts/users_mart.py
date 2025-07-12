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
    .appName("Users Mart") \
    .master("local") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "2g") \
    .config("spark.driver.cores", "1") \
    .config("spark.executor.instances", "5") \
    .getOrCreate()


# Загрузка данных
events = spark.read.parquet("/user/master/data/geo/events")
geo = spark.read.option("header", True).option("sep", ";").csv("/user/marbiana/data/geo.csv")

# Перевод координат в радианы
events = events.withColumn("lat_rad", F.radians(F.col("lat"))) \
               .withColumn("lon_rad", F.radians(F.col("lon")))
geo = geo.withColumn("city_lat_rad", F.radians(F.col("lat").cast("double"))) \
         .withColumn("city_lon_rad", F.radians(F.col("lng").cast("double")))

# Картезиан, гаверсинус
joined = events.crossJoin(geo)
distance = 2 * 6371 * F.asin(F.sqrt(
    F.pow(F.sin((F.col("lat_rad") - F.col("city_lat_rad")) / 2), 2) +
    F.cos(F.col("lat_rad")) * F.cos(F.col("city_lat_rad")) *
    F.pow(F.sin((F.col("lon_rad") - F.col("city_lon_rad")) / 2), 2)
))
joined = joined.withColumn("distance", distance)

w = Window.partitionBy("event.message_id").orderBy("distance")
events_with_city = joined.withColumn("rn", F.row_number().over(w)).filter("rn = 1")

# Актуальный город
latest = events_with_city.groupBy(F.col("event.message_from")).agg(
    F.max("event.datetime").alias("max_time")
)
act_city = latest.join(
    events_with_city,
    (latest["message_from"] == events_with_city["event.message_from"]) &
    (latest["max_time"] == events_with_city["event.datetime"])
).select(
    F.col("event.message_from").alias("user_id"),
    F.col("city").alias("act_city")
)

# Домашний город
visits = events_with_city.select(
    F.col("event.message_from").alias("user_id"),
    F.col("city"),
    F.to_date("event.datetime").alias("date")
).distinct()

visits_per_city = visits.groupBy("user_id", "city").agg(F.countDistinct("date").alias("days"))
home_city = visits_per_city.filter("days >= 27").withColumn(
    "rank", F.row_number().over(Window.partitionBy("user_id").orderBy(F.desc("days")))
).filter("rank = 1").select("user_id", F.col("city").alias("home_city"))

# Travel count + array
travel = visits.orderBy("date").groupBy("user_id").agg(
    F.count("*").alias("travel_count"),
    F.collect_list("city").alias("travel_array")
)

# Local time
local = latest.join(
    events_with_city,
    (latest["message_from"] == events_with_city["event.message_from"]) &
    (latest["max_time"] == events_with_city["event.datetime"])
).select(
    F.col("event.message_from").alias("user_id"),
    F.from_utc_timestamp("event.datetime", "Australia/Sydney").alias("local_time")
)

# Финал
users_mart = act_city.join(home_city, "user_id", "left") \
                     .join(travel, "user_id", "left") \
                     .join(local, "user_id", "left")

users_mart.coalesce(1).write.mode("overwrite").parquet("/user/marbiana/data/users_final")

spark.stop()