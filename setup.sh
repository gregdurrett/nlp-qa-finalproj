#!/bin/bash

mkdir -p datasets
mkdir -p glove

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz -O datasets/squad_train.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz -O datasets/newsqa_train.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz -O datasets/squad_dev.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz -O datasets/newsqa_dev.jsonl.gz
wget http://participants-area.bioasq.org/MRQA2019/ -O datasets/bioasq.jsonl.gz
wget http://cs.utexas.edu/~gdurrett/courses/fa2020/squad_dev_addOneSent_mrqa.json.gz -O datasets/squad_adversarial_addonesent.jsonl.gz

wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.*.txt glove/
rm glove.6B.zip

echo
echo "done!"
