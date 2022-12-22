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
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

# Create Spark Session and Name it
spark = (
    SparkSession
    .builder
    .appName('Big_Data_Project')
    .getOrCreate()
)


# Read the data
df = spark.read.option("header", True).csv("input/1992.csv")
print((df.count(), len(df.columns)))


#Drop collumns that we can't use
df = df.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",  "CarrierDelay", "WeatherDelay", "NASDelay",
             "SecurityDelay", "LateAircraftDelay")

print((df.count(), len(df.columns)))

#THIS DEPENDS ON THE DATASET, WE SHOULD NOT HAVE IT HARDCODED
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

indexer = StringIndexer(inputCol="Origin", outputCol="Origin_ind")
indexer1 = StringIndexer(inputCol="Dest", outputCol="Dest_ind")
indexer2 = StringIndexer(inputCol="DayOfWeek", outputCol="DayOfWeek_ind")

#colunas = df.columns
#colunas = [col for col in colunas if col not in ["Origin", "Dest", "DayOfWeek"]]
columns = [c for c in df.columns if c not in ["Origin", "Dest", "DayOfWeek"]]


# Remove rows with missing values
print("Removing rows with missing values ...")
df = df.na.drop()
expr = ' and '.join('(%s != "NA")' % col_name for col_name in df.columns)
df.filter(expr)
expr = ' and '.join('(%s != "")' % col_name for col_name in df.columns)
df.filter(expr)

# Create a VectorAssembler to combine the features into a single vector column
assembler = VectorAssembler(inputCols=columns, outputCol="features")


selector = UnivariateFeatureSelector(featuresCol="features", outputCol="selectedFeatures",
                                     labelCol="ArrDelay", selectionMode="numTopFeatures")


selector.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(6)

pipeline = Pipeline(stages=[indexer, indexer1, indexer2, assembler, selector])
transformed_df = pipeline.fit(df).transform(df)
transformed_df = transformed_df.select(['selectedFeatures', "ArrDelay"])

print("UnivariateFeatureSelector output with top %d features selected using f_classif"
      % selector.getSelectionThreshold())

transformed_df.show()

transformed_df = transformed_df.drop("Origin", "Dest", "DayOfWeek")
transformed_df.printSchema()

transformed_df.printSchema()

# Create training and test sets for the models
training, test = transformed_df.randomSplit([0.70, 0.30], seed=19283)
print("Rows in training set: " + str(training.count()))
print("Rows in test set: " + str(test.count()))


####################################################################################################################################################################################################

print("Creating the Linear Regression model ...")
lr = LinearRegression(featuresCol="selectedFeatures", labelCol="ArrDelay")
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.0, 0.4, 0.8]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.fitIntercept, [True]) \
    .build()

# Select the best model with 3-folds cross validation
cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay"),
                    numFolds=3)

# Fit the model and get the output
model = cv.fit(training)
output = model.transform(test).select("prediction", "ArrDelay")

# Transform output into RDD and compute metrics
output_rdd = output.rdd.map(
    lambda x: (float(x[0]), float(x[1]))
)
metrics = RegressionMetrics(output_rdd)
print("______________________________________________________________________________________________")
print("LINEAR REGRESSION METRICS" + "\n" +
        "Explained Variance: " + str(metrics.explainedVariance) + "\n" +
        "Mean absolute error: " + str(metrics.meanAbsoluteError) + "\n" +
        "Mean squared error (MSE): " + str(metrics.meanSquaredError) + "\n" +
        "R Squared: " + str(metrics.r2))
print("______________________________________________________________________________________________")


####################################################################################################################################################################################################
