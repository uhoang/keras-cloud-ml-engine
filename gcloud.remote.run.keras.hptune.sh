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
  --config=trainer/hptuning-config.yaml \
  -- \
  --train-file gs://keras-sentiment/sentiment_set.pickle \
  --dropout-one 0.25 \ 
  --dropout-two 0.25
