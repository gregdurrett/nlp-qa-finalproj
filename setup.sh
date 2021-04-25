#!/bin/bash

mkdir -p datasets
mkdir -p glove

if which -s wget; then
  WGET_CMD='wget'
  SAVE_ARG=''
  SAVE_TO_ARG='-O'
else
  WGET_CMD='curl -L' # -L to follow redirects
  SAVE_ARG='-O'
  SAVE_TO_ARG='-o'
fi

$WGET_CMD https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz $SAVE_TO_ARG datasets/squad_train.jsonl.gz
$WGET_CMD https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz $SAVE_TO_ARG datasets/newsqa_train.jsonl.gz
$WGET_CMD https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz $SAVE_TO_ARG datasets/squad_dev.jsonl.gz
$WGET_CMD https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz $SAVE_TO_ARG datasets/newsqa_dev.jsonl.gz
$WGET_CMD http://participants-area.bioasq.org/MRQA2019/ $SAVE_TO_ARG datasets/bioasq.jsonl.gz
$WGET_CMD http://cs.utexas.edu/~gdurrett/courses/fa2020/squad_dev_addOneSent_mrqa.json.gz $SAVE_TO_ARG datasets/squad_adversarial_addonesent.jsonl.gz

$WGET_CMD http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.6B.zip $SAVE_ARG
unzip glove.6B.zip
mv glove.6B.*.txt glove/
rm glove.6B.zip

echo
echo "done!"
