from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("PySpark Test").getOrCreate()

# Create a simple DataFrame
data = [("Alice", 1), ("Bob", 2), ("Cathy", 3)]
df = spark.createDataFrame(data, ["name", "age"])

# Show the DataFrame
df.show()

# Stop the SparkSession
spark.stop()