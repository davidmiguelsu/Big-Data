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
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.transformer import Transformer
from pyspark.ml.util import DefaultParams



#######
from pyspark.ml.feature import ChiSqSelector
num_features = 10

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

selected_columns = [col for col in df.columns if col not in ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",  "CarrierDelay", "WeatherDelay", "NASDelay",
             "SecurityDelay", "LateAircraftDelay"]]

print((df.count(), len(df.columns)))

selected_columns = [col for col in selected_columns if col not in ["Cancelled", "UniqueCarrier", "CRSDepTime", "CRSElapsedTime","TailNum", "TaxiOut", "CancellationCode", "FlightNum"]]

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

# Use VectorAssembler to create a single vector column from the selected columns
selected_columns = [col for col in selected_columns if col not in ["Origin", "Dest", "DayOfWeek", "Origin_ind", "Dest_ind", "DayOfWeek_ind"]]

# Define a custom transformer that calls the drop_na function on a DataFrame
class DropNA(Transformer, HasInputCol, HasOutputCol):
    # Define the input and output columns
    inputCol = Param(Params._dummy(), "inputCol", "input column")
    outputCol = Param(Params._dummy(), "outputCol", "output column")

    # Set the input and output columns
    def setInputCol(self, value):
        self._set(inputCol=value)
    def setOutputCol(self, value):
        self._set(outputCol=value)

    # Define the transform method
    def transform(self, dataset):
        # Select the input column
        df = dataset.select(col(self.getInputCol()))

        # Drop rows with NA values
        df = df.na.drop()

        return df

assembler = VectorAssembler(inputCols=selected_columns, outputCol="features")

selector = ChiSqSelector(numTopFeatures=num_features, featuresCol="features", labelCol="ArrDelay", outputCol="selected_features")

pipeline = Pipeline(stages=[indexer, indexer1, indexer2, encoder,encoder1, encoder2, assembler, drop_na,  selector])
transformed_df = pipeline.fit(df).transform(df)
# Drop the unnecessary columns
transformed_df = transformed_df.select("selected_features")
#df.drop("Origin", "Dest", "DayOfWeek", "Origin_ind", "Dest_ind", "DayOfWeek_ind")

print((df.count(), len(df.columns)))
df.printSchema()

print((transformed_df.count(), len(transformed_df.columns)))
transformed_df.printSchema()

num_features = 10





