import findspark
findspark.init()
findspark.find()

import os

os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['JAVA_HOME'] = '/usr'
os.environ['SPARK_HOME'] = '/usr/lib/spark'
os.environ['PYTHONPATH'] = '/usr/local/lib/python3.8'

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# SparkSession
spark = SparkSession.builder \
    .appName("Zones Mart") \
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

events = events.withColumn("id", F.monotonically_increasing_id())

events = events.withColumn("lat_rad", F.radians(F.col("lat"))) \
               .withColumn("lon_rad", F.radians(F.col("lon")))

geo = geo.withColumn("city_lat_rad", F.radians(F.col("lat").cast("double"))) \
         .withColumn("city_lon_rad", F.radians(F.col("lng").cast("double")))

# Дата, неделя, месяц ---
events = events.withColumn("date", F.to_date("event.datetime")) \
               .withColumn("week", F.weekofyear("date")) \
               .withColumn("month", F.month("date"))

# Гаверсинус
joined = events.crossJoin(geo)

distance = 2 * 6371 * F.asin(F.sqrt(
    F.pow(F.sin((F.col("lat_rad") - F.col("city_lat_rad")) / 2), 2) +
    F.cos(F.col("lat_rad")) * F.cos(F.col("city_lat_rad")) *
    F.pow(F.sin((F.col("lon_rad") - F.col("city_lon_rad")) / 2), 2)
))

joined = joined.withColumn("distance", distance)

w = Window.partitionBy("id").orderBy("distance")

zone_events = joined.withColumn("rn", F.row_number().over(w)) \
                    .filter("rn = 1") \
                    .withColumnRenamed("city", "zone_id")

# Первое сообщение = регистрация
reg = zone_events.filter(F.col("event_type") == "message") \
                 .groupBy("event.message_from") \
                 .agg(F.min("date").alias("reg_date"))

users = reg.join(
    zone_events,
    (reg["event.message_from"] == zone_events["event.message_from"]) &
    (reg["reg_date"] == zone_events["date"])
).select(
    F.col("event.message_from").alias("user_id"),
    "zone_id",
    "week",
    "month"
)

# Группировки
week = zone_events.groupBy("zone_id", "week", "event_type") \
                  .agg(F.count("*").alias("cnt"))

month = zone_events.groupBy("zone_id", "month", "event_type") \
                   .agg(F.count("*").alias("cnt"))

week_pivot = week.groupBy("zone_id", "week") \
    .pivot("event_type").sum("cnt") \
    .withColumnRenamed("message", "week_message") \
    .withColumnRenamed("reaction", "week_reaction") \
    .withColumnRenamed("subscription", "week_subscription")

month_pivot = month.groupBy("zone_id", "month") \
    .pivot("event_type").sum("cnt") \
    .withColumnRenamed("message", "month_message") \
    .withColumnRenamed("reaction", "month_reaction") \
    .withColumnRenamed("subscription", "month_subscription")

week_user = users.groupBy("zone_id", "week") \
                 .agg(F.count("*").alias("week_user"))

month_user = users.groupBy("zone_id", "month") \
                  .agg(F.count("*").alias("month_user"))

week_mart = week_pivot.join(week_user, ["zone_id", "week"], "left") \
    .select("zone_id", "week", "week_message", "week_reaction", "week_subscription", "week_user")

month_mart = month_pivot.join(month_user, ["zone_id", "month"], "left") \
    .select("zone_id", "month", "month_message", "month_reaction", "month_subscription", "month_user")

# Сохраняем результат
week_mart.coalesce(1).write.mode("overwrite").parquet("/user/marbiana/data/week_zone")
month_mart.coalesce(1).write.mode("overwrite").parquet("/user/marbiana/data/month_zone")

spark.stop()