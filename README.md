## Keras on Cloud ML engine: A simple sentiment analysis

In this post, we will try to run a LSTM model for a simple sentiment analysis on Google's Cloud ML Engine. You can get the sample of train and test data from [here](https://github.com/liufuyang/kaggle-youtube-8m/tree/master/tf-learn/example-3-sentiment).

### Prepare your Google Cloud Machine Learning Engine
Follow this guide to setup GC ML Engine, https://cloud.google.com/ml-engine/docs/quickstarts/command-line 

### Cloud ML Engine's project structure 
The basic project structure will look something like this: 

```
├── README.md
├── sentiment_set.pickle
├── setup.py
└── trainer
    ├── cloudml-gpu.yaml
    ├── __init__.py
    └── sentiment_keras_hpt.py
```