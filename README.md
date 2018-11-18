# Predicting-Baby-Weight-using-Tensorflow-on-GCP
Natality dataset,a public data set from the USA's Centers for Disease Control and Prevention (CDC) consist of 137.8 millions samples. Goal is to predict the weight of the baby so that hospital staff can make appropriate arrangement (i.e incubator /oxygen supply) just before delivery.
# Objectives
1. Explore a public dataset with Cloud Datalab.
2. Execute queries to collect sample data from the dataset that is stored in BigQuery.
3. Identify features to use in your ML model.
4. Visualize the data using the Python data analysis tool pandas.
5. Split the data into training and evaluation data files using Cloud Dataflow.
6. Launch a preprocessing pipeline using Cloud Dataflow to create training and evaluation datasets.
7. Creaeting the tensorflow model.
8. Training on Cloud ML Engine 
9. Hyperparameter tuning 
10. Deploying and predicting with model 
11. Flask API to invoke prediction from deployed model.



![ml-w-sd1](https://user-images.githubusercontent.com/38006823/48672605-d8664d80-eb59-11e8-83d8-f1d484a6e974.png)


# Getting Started
Train, evaluate, and deploy a machine learning model to predict a baby's weight and send requests to the model to make online predictions. 

## Prerequisites
Creating the Project and Bucket in the Google Cloud Platform
```
BUCKET = 'qwiklabs-gcp-c7c9baab3ef322d9'
PROJECT = 'qwiklabs-gcp-c7c9baab3ef322d9'
REGION = 'us-central1'
```

Setting environment variable in bash
```
import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8' 
```
# Implementation

Create SQL query using natality data after the year 2000
```
query = """
SELECT weight_pounds, is_male, mother_age, plurality, gestation_weeks, ABS(FARM_FINGERPRINT(CONCAT(CAST(year AS STRING), CAST(month AS STRING)))) AS hashmonth
FROM publicdata.samples.natality
WHERE year > 2000 
 """
 ```
 
Call BigQuery and examine in dataframe
```
import google.datalab.bigquery as bq
df = bq.Query(query + " LIMIT 100 ").execute().result().to_dataframe()
df.head()
```
 Create an input function reading a file using the Dataset API
 ```
 dataset = tf.data.TextLineDataset(file_list).map(decode_csv)
 ```
 
 Define feature columns using
 ```
 tf.feature_column.categorical_column_with_vocabulary_list()
 tf.feature_column.numeric_column()
 ```
 
 To predict with the TensorFlow model, we also need a serving input function. We will want all the inputs from our user. That is
  ```
  tf.estimator.export.ServingInputReceiver(features, feature_placeholder)
   ```
Create an estimator function to train and evaluate
 ```
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
 ```
 
 Monitor in tensorboard
```
from google.datalab.ml import TensorBoard
TensorBoard().start('./babyweight_trained')
```

Install apache-beam 
```
pip install apache-beam[gcp]
```

Train the model on cloud
```
%bash
OUTDIR=gs://${BUCKET}/babyweight/trained_model
JOBNAME=babyweight_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=trainer.task \
  --package-path=$(pwd)/babyweight/trainer \
  --job-dir=$OUTDIR \
  --staging-bucket=gs://$BUCKET \
  --scale-tier=STANDARD_1 \
  --runtime-version=$TFVERSION \
  -- \
  --bucket=${BUCKET} \
  --output_dir=${OUTDIR} \
  --train_examples=200000
  ```
  
  Deploye the trained model to act as a REST web service 
```
%bash
MODEL_NAME="babyweight"
MODEL_VERSION="ml_on_gcp"
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/babyweight/trained_model/export/exporter/ | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION
```
