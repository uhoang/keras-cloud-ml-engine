# Keras on Cloud ML engine: A simple sentiment analysis

A guide to set up and run a LSTM model for a simple sentiment analysis on Google's Cloud ML Engine. You can get a sample of train and test datasets from [here](https://github.com/liufuyang/kaggle-youtube-8m/tree/master/tf-learn/example-3-sentiment).

### Prepare your Google Cloud Machine Learning Engine
1. Follow [this guide](https://cloud.google.com/ml-engine/docs/quickstarts/command-line) to set up GCP account and activate the Cloud ML Engine API 
2. Install Google Cloud SDK, you can check [here](https://cloud.google.com/sdk/docs/)
3. Set up the credentials quickly via web browser, run
`gcloud auth application-default login`
4. Check your Cloud ML Engine available models:
`gcloud ml-engine model list`

You should see `List 0 items`. because we haven't created any ML Engine models yet.

### Upload the preprocessed data to a Google Cloud Storage bucket 
Create bucket and copy the model input into it. Pay attention to **service region** (here I use **us-east1**).

```
gsutil mb us-east1 gs://your-bucket-name
gsutil cp -r sentiment_set.pickle gs://your-bucket-name/sentiment_set.pickle
```

### Create Cloud ML Engine's project structure 
This is how the basic project structure looks like on your local machine:
![project structure](img/recommended-project-structure.png?raw=true)

The `setup.py` file in the project root directory allows the Cloud ML Engine to automatically package your training application, then install the package with any dependencies on the servers it spins up.

`trainer/__init__.py` is required for the Cloud ML Engine to create a package from your module.

### Run the model with python (locally)
`python trainer/sentiment_keras_hpt.py --job-dir ./tmp/sentiment --train-file sentiment_set.pickle`

### Run the model with gcloud 
To run the model locally:
```
gcloud ml-engine local train \
  --module-name trainer.sentiment_keras_hpt \
  --package-path ./trainer \
  -- \
  --train-file sentiment_set.pickle \
  --job-dir ./tmp/sentiment_keras_hpt
```

or `source gcloud.local.run.keras.sh`

To submit a job to Cloud ML Engine:
```
export BUCKET_NAME=keras-sentiment
export JOB_NAME="sentiment_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --runtime-version 1.0 \
  --module-name trainer.sentiment_keras_hpt \
  --package-path ./trainer \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --train-file gs://keras-sentiment/sentiment_set.pickle 
```

Click here to view the [job status](https://console.cloud.google.com/mlengine/jobs?project=zinc-chiller-213404).

### Hyperparameter tuning
A **hyperparameter** is a parameter that is set before the model is trained. By contrast, the **weights** and **biases** are derived via training.
Cloud ML Engine can do hyperparameter tuning, i.e. running the training multiple times to figure out good values for hyperparameters. To do this, the trainer module needs to take in the hyperparameters as arguments.

For example, the [file](https://github.com/uhoang/keras-cloud-ml-engine/blob/master/trainer/sentiment_keras_hpt.py) takes the `dropout-one` and `dropout-two` arguments corresponding to the dropout rates of the two hidden layers. The dropout rate are doubles between 0.1 and 0.5, that means to drop out 10% to 50% of the incoming parameters from the previous layer. The doubles are chosen to maximize the `accuracy` metric. `UNIT_REVERSE_LOG_SCALE` is chosen so that it checks values more densely on the bottom end of the range, since the original values were 0.25. Eight trials are run, with a maximum of four running at any given time:

```

trainingInput:
  scaleTier: CUSTOM
  # standard_gpu provides 1 GPU. Change to complex_model_m_gpu for 4 GPUs
  masterType: standard_gpu
  runtimeVersion: "1.0"
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: accuracy
    maxTrials: 8
    enableTrialEarlyStopping: True
    maxParallelTrials: 4
    params:
      - parameterName: dropout-one
        type: DOUBLE
        minValue: 0.1
        maxValue: 0.5
        scaleType: UNIT_REVERSE_LOG_SCALE
      - parameterName: dropout-two
        type: DOUBLE
        minValue: 0.1
        maxValue: 0.5
        scaleType: UNIT_REVERSE_LOG_SCALE
```

References:
  -[A great tutorial for Tensorflow and Keras with Cloud ML Engine](http://liufuyang.github.io/2017/04/02/just-another-tensorflow-beginner-guide-4.html) for beginners from Fuyang Liu
  -[Hyperparameter Tuning on Cloud ML Engine](https://github.com/clintonreece/keras-cloud-ml-engine)
  -[Overview of Hyperparameter Tuning](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview)