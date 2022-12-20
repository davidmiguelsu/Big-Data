from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, IntegerType
from pyspark.sql.functions import * 
import pyspark.sql.functions as pysparkfunc
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.sql.column import Column
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import math

# Create Spark Session and Name it
spark = (
    SparkSession
    .builder
    .appName('Big_Data_Project')
    .getOrCreate()
)

# Read csv input file
df = spark.read.option("header", True).csv("input/1992.csv")

print((df.count(), len(df.columns)))

#Drop collumns that we can't use
df = df.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",  "CarrierDelay", "WeatherDelay", "NASDelay",
             "SecurityDelay", "LateAircraftDelay")

print((df.count(), len(df.columns)))

#Drop columns that have high correlation with other features or aren't relevant to the problem we are solving or that have more than 50% null values
df = df.drop("Cancelled", "UniqueCarrier", "CRSDepTime", "CRSElapsedTime","TailNum", "TaxiOut", "CancellationCode", "FlightNum")

#Change datatype of columns - https://www.geeksforgeeks.org/how-to-change-column-type-in-pyspark-dataframe/
df = df.withColumn("Year",df["Year"].cast(IntegerType())) \
    .withColumn("Month",df["Month"].cast(IntegerType())) \
    .withColumn("DayofMonth",df["DayofMonth"].cast(IntegerType())) \
    .withColumn("ArrDelay",df["ArrDelay"].cast(IntegerType())) \
    .withColumn("DepDelay",df["DepDelay"].cast(IntegerType())) \
    .withColumn("Distance",df["Distance"].cast(IntegerType())) 

# Convert the 2 last variables from HHMM to minutes
# https://sparkbyexamples.com/pyspark/pyspark-timestamp-difference-seconds-minutes-hours/
df = df.withColumn("DepTime", when(pysparkfunc.length("DepTime") == 3, 
                        df["DepTime"].substr(1, 1).cast(IntegerType()) * 60 +
                        df["DepTime"].substr(2, 2).cast(IntegerType()))
                   .when(pysparkfunc.length("DepTime") == 4, 
                        df["DepTime"].substr(1, 2).cast(IntegerType()) * 60 +
                        df["DepTime"].substr(3, 2).cast(IntegerType()))) 
df = df.withColumn("CRSArrTime", when(pysparkfunc.length("CRSArrTime") == 3,
                        df["CRSArrTime"].substr(1, 1).cast(IntegerType()) * 60 +
                        df["CRSArrTime"].substr(2, 2).cast(IntegerType()))
                    .when(pysparkfunc.length("CRSArrTime") == 4,
                        df["CRSArrTime"].substr(1, 2).cast(IntegerType()) * 60 +
                        df["CRSArrTime"].substr(3, 2).cast(IntegerType())))

# drop rows with any null or "NA" value
# https://stackoverflow.com/questions/54843227/drop-rows-containing-specific-value-in-pyspark-dataframe
df = df.na.drop()
expr = ' and '.join('(%s != "NA")' % col_name for col_name in df.columns)
df.filter(expr)


print((df.count(), len(df.columns)))
df.printSchema()

# Converting categorical variables to numerical
# Create a StringIndexer instance
indexer = StringIndexer(inputCol="Origin", outputCol="Origin_ind")
indexer1 = StringIndexer(inputCol="Dest", outputCol="Dest_ind")
indexer2 = StringIndexer(inputCol="DayOfWeek", outputCol="DayOfWeek_ind")

encoder = OneHotEncoder(inputCols=["Origin_ind"], outputCols=["Origin_enc"])
encoder1 = OneHotEncoder(inputCols=["Dest_ind"], outputCols=["Dest_enc"])
encoder2 = OneHotEncoder(inputCols=["DayOfWeek_ind"], outputCols=["DayOfWeek_enc"])

# # create a list of the columns to standardize
# columns_to_standardize = ["Year", "Month", "DayofMonth", "DepTime", "CRSArrTime", "ArrDelay", "DepDelay", "Distance"]
# assembler = VectorAssembler(inputCols=columns_to_standardize, outputCol="features")

# # create a StandardScaler object
# scaler = StandardScaler(inputCol="features", outputCol="std_col")

#pipeline = Pipeline(stages=[indexer, indexer1, indexer2, encoder,encoder1, encoder2, assembler, scaler])
pipeline = Pipeline(stages=[indexer, indexer1, indexer2, encoder,encoder1, encoder2])
transformed_df = pipeline.fit(df).transform(df)
df.drop("Origin", "Dest", "DayOfWeek", "Origin_ind", "Dest_ind", "DayOfWeek_ind")

print((df.count(), len(df.columns)))
df.printSchema()

print((transformed_df.count(), len(transformed_df.columns)))
transformed_df.printSchema()





