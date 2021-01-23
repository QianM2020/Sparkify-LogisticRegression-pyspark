#Sparkify Project Workspace
# import libraries
# coding=<encoding name>
#!/usr/bin/python
# coding=utf-8
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum as Fsum
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, PCA, RegexTokenizer, StandardScaler, StopWordsRemover, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, concat, count, lit, udf, avg
from sklearn.metrics import classification_report

import datetime
import pickle
import numpy as np
import pandas as pd
# % matplotlib inline
import matplotlib.pyplot as plt

def load_data(database_filepath):
    '''
    The function creates a Spark session and loads the dataset from the json file into spark.
    '''
    df = spark.read.json(database_filepath) # database_filepath = "mini_sparkify_event_data.json"
    return df
    pass

def clean_data(df):
    '''
    The function drops rows ["userId", "sessionId"] have Missing Values or empty from the inputed dataset
                 add new cols ('Churn','chgrd','time_gap','sex') into dataset
                 flag 'cancelled user' as 1, 'others' as 0 in col 'Churn'
                 flag 'paid' usage in col 'chgrd'
                 flag 'gender' in numeric format in col 'chgrd'
                 get usage time by calculate max(ts) - min(ts) per user and saved the result in col 'time_gap'
    '''
    # drop rows ["userId", "sessionId"] have Missing Values or empty
    user_log_valid = user_log.dropna(how = "any", subset = ["userId", "sessionId"])
    user_log_valid = user_log_valid.filter(user_log_valid["userId"] != "")

    # find whether users Cancellation Confirmation and then flag'cancelled user' as 1, 'others' as 0 in 'Churn'.
    flag_Cancel_event = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
    user_log_valid = user_log_valid.withColumn("Churn", flag_Cancel_event("page"))
    windowval = Window.partitionBy("userId").orderBy(desc("ts")).rangeBetween(Window.unboundedPreceding, 0)
    user_log_valid = user_log_valid.withColumn("Churn", Fsum("Churn").over(windowval))

    # get numbers of page operations when the users' level = 'paid' and saved the numbers in column 'chgrd'.
    flag_changegrade_event = udf(lambda x: 1 if x == 'paid' else 0, IntegerType())
    user_log_valid = user_log_valid.withColumn("chgrd", flag_changegrade_event("level"))
    windowval = Window.partitionBy("userId").orderBy(desc("ts")).rangeBetween(Window.unboundedPreceding, 0)
    user_log_valid = user_log_valid.withColumn("chgrd", Fsum("chgrd").over(windowval))

    # identify users' gender, flag male as 1 and female as 0 in 'sex'
    flag_gender_event = udf(lambda x: 1 if x == "M" else 0, IntegerType())
    user_log_valid = user_log_valid.withColumn("sex", flag_gender_event("gender"))

    # create a temporary View and run SQL queries to get cleaned data df
    user_log_valid.createOrReplaceTempView("user_log_valid_table")
    # cleaned data includes user Id, genda, paid usage(chrgd), users' stay time (time_gap), and cancel or not flag (Churn)
    df = spark.sql(
        '''
        select t1.userId,sex,chgrd,time_gap,Churn
        from user_log_valid_table t1
        join
         (select userId, max(chgrd) as mchgrd,(max(ts) - min(ts))/1000 as time_gap
         from user_log_valid_table
         group by userId) t2
        on (t1.userId == t2.userId and t1.chgrd == t2.mchgrd)
        order by Churn
        '''
                      ).dropDuplicates()
    return df
    pass

def explore_data(df):
    '''
    The function displays the Sex distribution,Usage Time distribution,
    and paid usage Distribution between cancled users and noncancled users
    '''
    # display the sex distribution between cancled users and noncancled users
    print('Insights:\n 1 Sex Disparity of Cancled User \n')
    df.groupby(['Churn','sex']).count().show()

    # display the usage time distribution between cancled users and noncancled users
    print('\n 2 Usage Time Distribution \n')
    df.groupby('Churn').avg('time_gap').show()

    # display the paid usage (average of 'chgrd')distribution between cancled users and noncancled users
    #print('\n 3 Usage paid usage Distribution \n')
    #df.groupby('Churn').avg('chgrd').show()

    pass

def feature_engi(df):
    '''
    The function Combine the gender, usage time, and paid usage columns into a vector,
    and Scales the Vectors
    '''
    #Combine the gender, usage time, and paid usage columns into a vector
    assembler = VectorAssembler(inputCols=["sex","time_gap","chgrd"], outputCol="NumFeatures")
    df = assembler.transform(df)
    pca = PCA(k=2, inputCol="NumFeatures", outputCol="pca") # k is the number of dims
    model = pca.fit(df)
    df = model.transform(df)
    #Scale the Vectors
    scaler = StandardScaler(inputCol="pca", outputCol="features",withMean=True, withStd=False)
    scalerModel = scaler.fit(df)
    df = scalerModel.transform(df)
    return df
    pass

def build_model():
    '''
    The function build model with
    machine pipeline which take in the combined and scaled vector column "ScaledNumFeatures" as input and output
    classification results on the 2 labels("Churn") in the dataset.
    '''
    indexer = StringIndexer(inputCol="Churn", outputCol="label")
    lr =  LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)
    pipeline = Pipeline(stages=[indexer,lr])
    #tune model
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam,[0.0, 0.1]) \
        .addGrid(lr.maxIter,[10, 20]) \
        .build()
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(),
                              numFolds=3)
    return crossval
    pass

def getBestParam(cvModel):
    '''
    The function gets the best parameter of cvModel
    '''
    params = cvModel.getEstimatorParamMaps()
    avgMetrics = cvModel.avgMetrics

    all_params = list(zip(params, avgMetrics))
    best_param = sorted(all_params, key=lambda x: x[1], reverse=True)[0]
    return best_param

best_param = getBestParam(cvmodel)[0]
for p, v in best_param.items():
	print("{} : {}".format(p.name, v))

def evaluate_model(model, validation):
    '''
    The function reports the accuracy of the model prediction against validation dataset
    '''
    results = model.transform(validation)
    # using classification_report to get F1 score
    labels = np.unique(results.select ('prediction').collect())
    confusion_mat = classification_report(results.select ('label').collect(), results.select ('prediction').collect(),labels=labels)
    print(" Accuracy:\n")
    print((results.filter(results.label == results.prediction).count())/(results.count()))
    print(" Confusion Matrix:\n ", confusion_mat)
    return results
    pass

def save_model(model, model_filepath):
    '''
    The function export tained model as a pickle file
    '''
    s = pickle.dumps(model.avgMetrics)
    with open(model_filepath,'wb+') as f: # mode is'wb+'，represents binary writen
        f.write(s)
    pass

def main():
    '''
    The main() function combines and executes all the above modules.
    '''
    # create a Spark session
    spark = SparkSession.builder \
        .master("local") \
        .appName("sparkify") \
        .getOrCreate()

    database_filepath, model_filepath = sys.argv[1:]
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    df = load_data(database_filepath)
    df = clean_data(df)
    explore_data(df)

    print('Building model...')
    df = feature_engi(df)
    model = build_model()

    # Train Test Split
    #break data set into 90% of training data and set aside 10%. Set random seed to 42.
    rest, validation = df.randomSplit([0.9, 0.1], seed=42)


    #Train pipeline
    print('Training model...')
    cvmodel = model.fit(rest)

    print('Evaluating model...')
    evaluate_model(cvmodel ,validation)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

if __name__ == '__main__':
    main()
