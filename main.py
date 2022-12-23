from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import * 
import pyspark.sql.functions as pysparkfunc
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.stat import Correlation
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os

def remove_null(df):
    # https://stackoverflow.com/questions/54843227/drop-rows-containing-specific-value-in-pyspark-dataframe
    df = df.na.drop()
    expr = ' and '.join('(%s != "NA")' % col_name for col_name in df.columns)
    df.filter(expr)
    expr = ' and '.join('(%s != "")' % col_name for col_name in df.columns)
    df.filter(expr)
    return df

def data_analysis(df, dataset_name):
        df = df.drop("UniqueCarrier", "Origin", "Dest", "Year", "Cancelled")
        df.printSchema()
        #https://stackoverflow.com/questions/52214404/how-to-get-the-correlation-matrix-of-a-pyspark-data-frame
        # convert to vector column first
        vector_col = "corr_features"
        assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
        df_vector = assembler.transform(df).select(vector_col)

        # get correlation matrixx
        matrix = Correlation.corr(df_vector, vector_col)
        matrix = Correlation.corr(df_vector, 'corr_features').collect()[0][0] 
        corr_matrix = matrix.toArray().tolist() 
        corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = df.columns, index= df.columns) 
        corr_matrix_df.style.background_gradient(cmap='coolwarm').set_precision(2)


        plt.figure(figsize=(19,12))  
        sns.heatmap(corr_matrix_df, 
                    xticklabels=corr_matrix_df.columns.values,
                    yticklabels=corr_matrix_df.columns.values,  cmap="Greens", annot=True, vmin=-0.5, vmax=1)

        heatmap_name = dataset_name + ".png"
        plt.savefig(heatmap_name)


def data_cleaning(df):
    #Drop collumns that we can't use
    df = df.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",  "CarrierDelay", "WeatherDelay", "NASDelay",
             "SecurityDelay", "LateAircraftDelay")
    
    #Drop columns that have high correlation with other features or aren't relevant to the problem we are solving or that have more than 50% null values
    df = df.drop("Cancelled", "UniqueCarrier", "CRSDepTime", "CRSElapsedTime","TailNum", "TaxiOut", "CancellationCode", "FlightNum")

    df = remove_null(df)
    
    return df

def data_preparation(df):
    #Change datatype of columns - https://www.geeksforgeeks.org/how-to-change-column-type-in-pyspark-dataframe/
    df = df.withColumn("Year",df["Year"].cast(IntegerType())) \
        .withColumn("Month",df["Month"].cast(IntegerType())) \
        .withColumn("DayofMonth",df["DayofMonth"].cast(IntegerType())) \
        .withColumn("ArrDelay",df["ArrDelay"].cast(IntegerType())) \
        .withColumn("DepDelay",df["DepDelay"].cast(IntegerType())) \
        .withColumn("Distance",df["Distance"].cast(IntegerType())) \
        .withColumn("DepTime",df["DepTime"].cast(IntegerType())) \
        .withColumn("CRSArrTime",df["CRSArrTime"].cast(IntegerType())) 

    # Convert the 2 last variables from HHMM to minutes
    # https://sparkbyexamples.com/pyspark/pyspark-timestamp-difference-seconds-minutes-hours/
    df = df.withColumn("DepTime", when(pysparkfunc.length("DepTime") == 3, 
                            df["DepTime"].substr(1, 1) * 60 +
                            df["DepTime"].substr(2, 2) ) 
                    .when(pysparkfunc.length("DepTime") == 4, 
                            df["DepTime"].substr(1, 2) * 60 +
                            df["DepTime"].substr(3, 2)) )
    df = df.withColumn("CRSArrTime", when(pysparkfunc.length("CRSArrTime") == 3,
                            df["CRSArrTime"].substr(1, 1) * 60 +
                            df["CRSArrTime"].substr(2, 2))
                        .when(pysparkfunc.length("CRSArrTime") == 4,
                            df["CRSArrTime"].substr(1, 2) * 60 +
                            df["CRSArrTime"].substr(3, 2)))                
    return df


def data_transformation(df):
    stages = []
    stages.append(StringIndexer(inputCol="Origin", outputCol="Origin_ind"))
    stages.append(StringIndexer(inputCol="Dest", outputCol="Dest_ind"))
    stages.append(StringIndexer(inputCol="DayOfWeek", outputCol="DayOfWeek_ind"))
    df = remove_null(df)

    columns = [c for c in df.columns if c not in ["Origin", "Dest", "DayOfWeek"]]

    stages.append(VectorAssembler(inputCols=columns, outputCol="features"))

    stages.append(StandardScaler(inputCol="features",
                        outputCol="scaledFeatures",
                        withStd=True,
                        withMean=False))

    selector = UnivariateFeatureSelector(featuresCol="scaledFeatures", outputCol="selectedFeatures",
                                     labelCol="ArrDelay", selectionMode="numTopFeatures")
    selector.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(5)
    stages.append(selector)

    pipeline = Pipeline(stages=stages)
    df = pipeline.fit(df).transform(df)
    df = df.select(['selectedFeatures', "ArrDelay"])

    print("Final DATASET!")
    df.show()
    df.printSchema()
    return df


def linear_regression(training, test):
    lr = LinearRegression(featuresCol="selectedFeatures", labelCol="ArrDelay")
    paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.2, 0.4, 0.6, 0.8, 1.0]).build()

    cv = CrossValidator(estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay"),
                        numFolds=5)

    
    model = cv.fit(training)

    return model.transform(test).select("prediction", "ArrDelay")


def decision_tree(training, test):
    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="selectedFeatures", labelCol="ArrDelay")

    paramGrid = ParamGridBuilder().build()

    cv = CrossValidator(estimator=dt,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay"),
                        numFolds=5)

    
    model = cv.fit(training)
    
    return model.transform(test).select("prediction", "ArrDelay")
    

def data_evaluation(model_output):
    output_rdd = model_output.rdd.map(
        lambda x: (float(x[0]), float(x[1]))
    )
    metrics = RegressionMetrics(output_rdd)
    print("|Explained Variance = " + str(metrics.explainedVariance) + "|\n"
          "|Mean Absolute Error = " + str(metrics.meanAbsoluteError) + "|\n"
          "|Mean Squared Error = " + str(metrics.meanSquaredError) + "|\n"
          "|Root Mean Squared Error = "  + str(metrics.rootMeanSquaredError) + "|\n"
          "|R2 =" + str(metrics.r2) + "|" )   


if __name__ == '__main__': 
    # Create Spark Session and Name it
    spark = (
        SparkSession
        .builder
        .appName('Big_Data_Project')
        .getOrCreate()
    )
    #Help user
    if len(sys.argv) == 2:
        path = sys.argv[1]
        if path == "-h" or path == "help":
            print("\n\nProgram needs 3 arguments as follows: \n\npython main.py path_to_dataset ml_algorithm_number \n")
            print("Algorithms must be: \n\n1- Linear Regression\n\n or \n\n2- Decision Tree\n\n")
        sys.exit(-1)

    #verify if the number of arguments
    if len(sys.argv) != 3:
        print("\n\nProgram needs 3 arguments as follows: \n\npython main.py path_to_dataset ml_algorithm_number \n\n")
        sys.exit(-1)

    app_path = sys.argv[0]
    path = sys.argv[1]
    ml_algorithm = int(sys.argv[2])

    if not os.path.exists(path):
        print("Wrong dataset path!")
        sys.exit(-1)

    #verify the number of the algorithm 
    if (ml_algorithm != 1 and ml_algorithm!= 2):
        print("\n\nAlgorithms must be: \n\n1- Linear Regression\n\n or \n\n2- Decision Tree\n\n")
        sys.exit(-1)

    # Read the data
    df = spark.read.option("header", True).csv(path)

    df = data_cleaning(df)
    df = data_preparation(df)
    df = data_transformation(df)

    # Create training and test sets for the models
    training, test= df.randomSplit([0.70, 0.30], 100)
    print("Training set size: " + str(training.count()))
    print("Testing set size: " + str(test.count()))
    
    #data_modeling
    if ml_algorithm == 1:
        print("Using linear regression!")
        model = linear_regression(training, test)

    if ml_algorithm == 2:
        print("Using decision tree!")
        model = decision_tree(training, test)
    
    data_evaluation(model)
    spark.stop()
    


    
