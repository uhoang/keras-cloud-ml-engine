# Keras on Cloud ML engine: A simple sentiment analysis

A beginner guide to set up and run a LSTM model for a simple sentiment analysis on Google's Cloud ML Engine. You can get a sample of train and test datasets from [here](https://github.com/liufuyang/kaggle-youtube-8m/tree/master/tf-learn/example-3-sentiment).

### Prepare your Google Cloud Machine Learning Engine
1. Follow [this guide](https://cloud.google.com/ml-engine/docs/quickstarts/command-line) to set up GCP account and activate the Cloud ML Engine API 
2. Install Google Cloud SDK, you can check [here](https://cloud.google.com/sdk/docs/)
3. Set up the credetials quickly via web browser, run
`gcloud auth application-default login`
4. Check your Cloud ML Engine available models:
`gcloud ml-engine model list`

You should see `List 0 items`. because we haven't created any ML Engine models yet.

### Upload the preprocessed data to a Google Cloud Storage bucket 
```
gsutil mb gs://your-bucket-name
gsutil cp -r sentiment_set.pickle gs://your-bucket-name/sentiment_set.pickle
```

### Create Cloud ML Engine's project structure 
The basic project structure will look something like this: 
[project structure](/img/recommended-project-structure.png)

<!-- ```
├── README.md
├── sentiment_set.pickle
├── setup.py
└── trainer
    ├── cloudml-gpu.yaml
    ├── __init__.py
    └── sentiment_keras_hpt.py
``` -->