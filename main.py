from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, IntegerType
from pyspark.sql.functions import * 
import pyspark.sql.functions as pysparkfunc
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
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
df = spark.read.option("header", True).csv("assignment_g30/input/1992.csv")

#Drop collumns that we can't use
df = df.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",  "CarrierDelay", "WeatherDelay", "NASDelay",
             "SecurityDelay", "LateAircraftDelay")

# Get percentage of missing values in each column - 
# https://stackoverflow.com/questions/59969378/how-do-i-calculate-the-percentage-of-none-or-nan-values-in-pyspark

#amount_missing_df = df.select([(count(when(col(c).contains("NA"), c))/count(lit(1))).alias(c) for c in df.columns])
#amount_missing_df.show()

#Drop columns with over 50% null values
df = df.drop("TailNum", "TaxiOut", "CancellationCode")

#Get the dataType of each column in the dataset
#df.printSchema()

#Export dataframe into a csv
#df.write.csv('mycsv.csv')
#Change datatype of columns - https://www.geeksforgeeks.org/how-to-change-column-type-in-pyspark-dataframe/
df = df.withColumn("Year",df["Year"].cast(IntegerType())) \
    .withColumn("Month",df["Month"].cast(IntegerType())) \
    .withColumn("DayofMonth",df["DayofMonth"].cast(IntegerType())) \
    .withColumn("DayofWeek",df["DayofWeek"].cast(IntegerType()))\
    .withColumn("FlightNum",df["FlightNum"].cast(IntegerType())) \
    .withColumn("CRSElapsedTime",df["CRSElapsedTime"].cast(IntegerType())) \
    .withColumn("ArrDelay",df["ArrDelay"].cast(IntegerType())) \
    .withColumn("DepDelay",df["DepDelay"].cast(IntegerType())) \
    .withColumn("Cancelled",df["Cancelled"].cast(IntegerType())) \
    .withColumn("Distance",df["Distance"].cast(IntegerType())) 



# Convert the 3 last variables from the HHMM to minutes
#https://sparkbyexamples.com/pyspark/pyspark-timestamp-difference-seconds-minutes-hours/
df = df.withColumn("DepTime", when(pysparkfunc.length("DepTime") == 3, 
                        df["DepTime"].substr(1, 1).cast(IntegerType()) * 60 +
                        df["DepTime"].substr(2, 2).cast(IntegerType()))
                   .when(pysparkfunc.length("DepTime") == 4, 
                        df["DepTime"].substr(1, 2).cast(IntegerType()) * 60 +
                        df["DepTime"].substr(3, 2).cast(IntegerType()))) 
df = df.withColumn("CRSDepTime", when(pysparkfunc.length("CRSDepTime") == 3,
                        df["CRSDepTime"].substr(1, 1).cast(IntegerType()) * 60 +
                        df["CRSDepTime"].substr(2, 2).cast(IntegerType()))
                    .when(pysparkfunc.length("CRSDepTime") == 4,
                        df["CRSDepTime"].substr(1, 2).cast(IntegerType()) * 60 +
                        df["CRSDepTime"].substr(3, 2).cast(IntegerType()))) 
df = df.withColumn("CRSArrTime", when(pysparkfunc.length("CRSArrTime") == 3,
                        df["CRSArrTime"].substr(1, 1).cast(IntegerType()) * 60 +
                        df["CRSArrTime"].substr(2, 2).cast(IntegerType()))
                    .when(pysparkfunc.length("CRSArrTime") == 4,
                        df["CRSArrTime"].substr(1, 2).cast(IntegerType()) * 60 +
                        df["CRSArrTime"].substr(3, 2).cast(IntegerType())))

df.printSchema()

# drop rows with any null value
df = df.na.drop()
print((df.count(), len(df.columns)))


