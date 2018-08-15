mkdir -p ./tmp/sentiment_keras
gcloud ml-engine local train \
  --module-name trainer.sentiment_keras \
  --package-path ./trainer \
  -- \
  --train-file sentiment_set.pickle \
  --job-dir ./tmp/sentiment_keras