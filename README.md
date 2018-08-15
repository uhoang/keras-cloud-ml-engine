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