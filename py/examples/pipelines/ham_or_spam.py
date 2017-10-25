from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, RegexTokenizer, StopWordsRemover, IDF
import h2o
from pysparkling import *
from pysparkling.ml import ColumnPruner

# Initiate SparkSession
spark = SparkSession.builder.appName("App name").getOrCreate()

hc = H2OContext.getOrCreate(spark)
sms_data_filename = "smsData.txt"
sms_data_filepath = "examples/smalldata/" + sms_data_filename

## This method loads the data, perform some basic filtering and create Spark's dataframe
def load(dataFile):
    sms_schema = StructType([StructField("label", StringType(), False), StructField("text", StringType(), False)])
    row_RDD = spark.sparkContext.textFile(sms_data_filepath).map(lambda x: x.split("\t")).filter(lambda r: r[0].strip())
    return spark.createDataFrame(row_RDD, sms_schema)

##
## Define the pipeline stages
##

## Tokenize the messages
tokenizer = RegexTokenizer(inputCol="text",
                           outputCol="words",
                           minTokenLength=3,
                           gaps=False,
                           pattern="[a-zA-Z]+")


## Remove ignored words
stopWordsRemover = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                                    outputCol="filtered",
                                    stopWords=["the", "a", "", "in", "on", "at", "as", "not", "for"],
                                    caseSensitive=False)


## Hash the words
hashingTF = HashingTF(inputCol=stopWordsRemover.getOutputCol(),
                      outputCol="wordToIndex",
                      numFeatures=1 << 10)


## Create inverse document frequencies model
idf = IDF(inputCol=hashingTF.getOutputCol(),
          outputCol="tf_idf",
          minDocFreq=4)

## Create GBM model
gbm = None

## Remove all helper columns
colPruner = ColumnPruner(columns=[idf.getOutputCol(), hashingTF.getOutputCol(), stopWordsRemover.getOutputCol(), tokenizer.getOutputCol()])

##  Create the pipeline by defining all the stages
pipeline = Pipeline(stages=[tokenizer, stopWordsRemover, hashingTF, idf, gbm, colPruner])

## Train the pipeline model
data = load(sms_data_filename)
model = pipeline.fit(data)

##
## Make predictions on unlabeled data
## Spam detector
##
def isSpam(smsText, model, h2oContext, hamThreshold = 0.5):
    smsTextSchema = StructType([StructField("text", StringType(), False)])
    smsTextRowRDD = spark.sparkContext.parallelize([smsText])
    smsTextDF = spark.sqlContext.createDataFrame(smsTextRowRDD, smsTextSchema)
    prediction = model.transform(smsTextDF)
    return prediction.select("spam").first().getDouble(0) > hamThreshold


print(isSpam("Michal, h2oworld party tonight in MV?", model, hc))

print(isSpam("We tried to contact you re your reply to our offer of a Video Handset? 750 anytime any networks mins? UNLIMITED TEXT?", model, hc))
