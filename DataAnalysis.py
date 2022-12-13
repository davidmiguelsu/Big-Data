# Get percentage of missing values in each column - 
# https://stackoverflow.com/questions/59969378/how-do-i-calculate-the-percentage-of-none-or-nan-values-in-pyspark

#amount_missing_df = df.select([(count(when(col(c).contains("NA"), c))/count(lit(1))).alias(c) for c in df.columns])
#amount_missing_df.show()

#Get the dataType of each column in the dataset
#df.printSchema()

#correlation between attributes and class variable
def get_dtype(df, colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]


for c in df.columns:
    if c == "ArrDelay" or get_dtype(df, c) != "string":
        cor = df.stat.corr(c, "ArrDelay")
        if math.isnan(cor):
            cor = "0"
        else:
            cor = str(cor)
        print("The correlation between " + c + " and the Class Variable ArrDelay is: " + cor)

#Export dataframe into a csv
#df.write.csv('halfclean.csv')


# Script for producing a Correlation Matrix
df2 = df
df2 = df2.drop("UniqueCarrier", "Origin", "Dest", "Year", "Cancelled")
df2.printSchema()
#https://stackoverflow.com/questions/52214404/how-to-get-the-correlation-matrix-of-a-pyspark-data-frame
# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df2.columns, outputCol=vector_col)
df_vector = assembler.transform(df2).select(vector_col)

# get correlation matrixx
matrix = Correlation.corr(df_vector, vector_col)

matrix = Correlation.corr(df_vector, 'corr_features').collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = df2.columns, index= df2.columns) 
corr_matrix_df.style.background_gradient(cmap='coolwarm').set_precision(2)


plt.figure(figsize=(19,12))  
sns.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,  cmap="Greens", annot=True, vmin=-0.5, vmax=1)
plt.savefig("heatmap3.png")



#TESTAR DEPOIS COM OneHotEncoder (Se formos usar Linear Regression dps pode ser uma ma escolha)
# Create a OneHotEncoder instance
#encoder = OneHotEncoder(inputCol="Origin", outputCol="Origin_index")
# Transform the dataframe
#onehot_df = encoder.transform(df)