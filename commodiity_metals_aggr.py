from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, floor, concat
from pyspark.sql.window import Window

# # Initialize Spark Session
# spark = SparkSession.builder \
#     .appName("Commodity Price Analysis") \
#     .enableHiveSupport() \
#     .getOrCreate()

spark = SparkSession\
    .builder\
    .appName("DataExploration")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://go01-demo/")\
    .enableHiveSupport() \
    .getOrCreate()


    

# Read data from Hive table in S3
commodities_df = spark.table("demo_bi.worldbank")



# List of commodity columns
commodity_columns = ["potash", "aluminum", "iron_ore", "copper", "lead", "tin", "nickel", "zinc", "gold", "platinum", "silver"]

columns_to_select = ["periode_year", "periode_month"] + commodity_columns
df = commodities_df.select(columns_to_select)

# Create a quarter column
df = df.withColumn("quarter", concat(col("periode_year").cast("string"), 
                                     (floor((col("periode_month") - 1) / 3) + 1).cast("string")))

# Create window specifications
year_window = Window.partitionBy("periode_year")
quarter_window = Window.partitionBy("quarter")

# Create new columns for yearly and quarterly averages and compute them
for commodity in commodity_columns:
    # Create new column names for yearly and quarterly averages
    year_avg_col_name = f"{commodity}_yr_avg"
    quarter_avg_col_name = f"{commodity}_qtr_avg"
    
    # Compute yearly average
    df = df.withColumn(
        year_avg_col_name,
        avg(col(commodity)).over(year_window)
    )
    
    # Compute quarterly average
    df = df.withColumn(
        quarter_avg_col_name,
        avg(col(commodity)).over(quarter_window)
    )

# Save the transformed data as a Hive table
df.write \
    .mode("overwrite") \
    .format("hive") \
    .saveAsTable("demo_bi.commodity_metals_aggr")


# Let us read this newly created table and show some records and the schema

# Read the transformed table
df = spark.table("demo_bi.commodity_metals_aggr")

# Read the transformed table
df = spark.table("demo_bi.commodity_metals_aggr")

print("======Showing the Commodity Metals Transformation for 2024 descending===== \n\n")

# Filter for years 2020 to 2024 and show top 10 rows
df.filter((df.periode_year >= 2020) & (df.periode_year <= 2024)) \
  .orderBy(df.periode_year.desc(), df.periode_month.desc()) \
  .show(10)

print("======Showing the Commodity Metals Transformation for 2024 descending======")
# Show the schema of the table
df.printSchema()

# Stop the Spark session
spark.stop()    