from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, avg, floor, when
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession\
    .builder\
    .appName("DataExploration")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://go01-demo/")\
    .enableHiveSupport() \
    .getOrCreate()

# Read data from the Worldbank table
df = spark.table("demo_bi.Worldbank")

# List of commodity columns (based on the provided column names)
commodity_columns = [
    "crude_petro", "crude_brent", "crude_dubai", "crude_wti", "coal_aus", "coal_safrica",
    "ngas_us", "ngas_eur", "ngas_jp", "inatgas", "cocoa", "coffee_arabic", "coffee_robus",
    "tea_avg", "tea_colombo", "tea_kolkata", "tea_mombasa", "coconut_oil", "grnut", "fish_meal",
    "grnut_oil", "palm_oil", "plmkrnl_oil", "soybeans", "soybean_oil", "soybean_meal", "rapeseed_oil",
    "sunflower_oil", "barley", "maize", "sorghum", "rice_05", "rice_25", "rice_a1", "rice_05_vnm",
    "wheat_us_srw", "wheat_us_hrw", "banana_eu", "banana_us", "orange", "beef", "chicken", "lamb",
    "shrimp_mex", "sugar_eu", "sugar_us", "sugar_wld", "tobac_us", "logs_cmr", "logs_mys", "sawnwd_cmr",
    "sawnwd_mys", "plywood", "cotton_a_indx", "rubber_tsr20", "rubber1_mysg", "phosrock", "dap", "tsp",
    "urea_ee_bulk", "potash", "aluminum", "iron_ore", "copper", "lead", "tin", "nickel", "zinc",
    "gold", "platinum", "silver"
]

# Create a function to generate quarter number
def get_quarter(month):
    return when((month >= 1) & (month <= 3), 1) \
           .when((month >= 4) & (month <= 6), 2) \
           .when((month >= 7) & (month <= 9), 3) \
           .otherwise(4)

# Add quarter number column
df = df.withColumn("qtr_nbr", get_quarter(col("periode_month")))

# Prepare the stack expression
stack_expr = ", ".join(f"'{com}', {com}" for com in commodity_columns)

# Melt the dataframe to create commodity and price columns
df_melted = df.select(
    "periode_year", 
    "periode_month", 
    "qtr_nbr", 
    expr(f"stack({len(commodity_columns)}, {stack_expr}) as (commodity, price)")
)

# Create window specifications
year_window = Window.partitionBy("periode_year", "commodity")
quarter_window = Window.partitionBy("periode_year", "qtr_nbr", "commodity")

# Calculate yearly and quarterly averages
df_final = df_melted.withColumn("price_qty_avg", avg("price").over(quarter_window)) \
                    .withColumn("price_yr_avg", avg("price").over(year_window))

# Select and order the final columns
df_final = df_final.select("periode_year", "periode_month", "qtr_nbr", "commodity", "price", "price_qty_avg", "price_yr_avg") \
                   .orderBy("periode_year", "periode_month", "commodity")

# Save as a new Hive table
df_final.write \
    .mode("overwrite") \
    .format("hive") \
    .saveAsTable("demo_bi.commodity_prices")

print("======Showing the Commodity  Transformation for 2024 - 2020 descending===== \n\n")

# Filter for years 2020 to 2024 and show top 10 rows
df_final.filter((df_final.periode_year >= 2020) & (df_final.periode_year <= 2024)) \
  .orderBy(df.periode_year.desc(), df.periode_month.desc()) \
  .show(20)

print("======Showing the Commodity  Transformation Schema======")
# Show the schema of the table
df_final.printSchema()



# Stop the Spark session
spark.stop()