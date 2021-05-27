# NLP QA Final Project

This repository contains starter code for the Natural Language Processing final project on question answering. For specific details on deliverables and deadlines, please refer to the final project spec posted on the course website.

Authors: Shrey Desai, Yasumasa Onoe, and Greg Durrett

1. [Getting Started](#getting-started)
    1. [Packages and Dependencies](#packages-and-dependencies)
    2. [File Descriptions](#file-descriptions)
2. [Datasets](#datasets)
3. [Training and Evaluation](#training-and-evaluation)
    1. [Training from Scratch](#training-from-scratch)
    2. [Using Pre-trained Models](#using-pre-trained-models)
3. [GCP Setup](#gcp-setup)

## Getting Started 

### Packages and Dependencies

To begin, clone this repository onto your local machine:

```bash
$ git clone https://github.com/gregdurrett/nlp-qa-finalproj.git
```

This project requires Python 3.6+ and the following dependencies:

- `torch==1.4.0`
- `numpy==1.18.2`
- `tqdm==4.44.1`
- `termcolor==1.1.0`

These packages can be installed via `pip`, either globally or in a virtual environment. If using a virtual environment, please use the following commands:

```bash
$ virtualenv -p python3.6 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Finally, `./setup.sh` will download the QA datasets (SQuAD, NewsQA, BioASQ) into `datasests/` and GloVe embeddings into `glove/`. By default, we use the 300-dimensional embeddings (`glove.6B.300d.txt`), but other embeddings are also provided for your convenience. If you don't have wget (e.g., macOS doesn't by default), it will attempt to use curl (which macOS does have). If neither work, you can just manually download the files from the URLs (first argument) and put them in the expected locations.

### File Descriptions

We provide descriptions of the following files:

- `main.py`: Main script for training and evaluating models, and writing predictions to an output file.
- `model.py`: Implementation of the baseline question answering model and attention modules.
- `data.py`: Handles the vocabulary, tokenization, dataset loading, and batching.
- `evaluate.py`: Calculates question answering metrics -- exact match (EM) and F1 -- given an output prediction file.
- `utils.py`: Miscellaneous utilities for training.

Each file also contains detailed docstrings and comments. Please get in touch with the course staff on Piazza or office hours if you have questions regarding the provided code.

## Datasets 

We provide the following datasets:

- SQuAD (`datasets/squad_train.jsonl.gz`, `datasets/squad_dev.jsonl.gz`)
- NewsQA (`datasets/newsqa_train.jsonl.gz`, `datasets/newsqa_dev.jsonl.gz`)
- BioASQ (`datasets/bioasq.jsonl.gz`)

Note that, unlike SQuAD and NewsQA, BioASQ does not have a designated training split. So, for instance, you could (1) use 50% of the data for fine-tuning and 50% for evaluating; or (2) train on SQuAD/NewsQA and evaluate on BioASQ to see if your QA model can generalize to out-of-domain samples.

Use our visualization script to see the types of passages, questions, and answers present in each dataset (answer spans will be colored red in stdout):

```
$ python3 visualize.py --path datasets/squad_train.jsonl.gz --samples 1

----------------------------------------------------------------------------------------------------

[METADATA]
path = 'datasets/squad_train.jsonl.gz'
question id = 38cc2597b6624bd8af1e8ba7f693096f

[CONTEXT]
architecturally , the school has a catholic character . atop the main building 's gold dome is a
golden statue of the virgin mary . immediately in front of the main building and facing it , is a
copper statue of christ with arms upraised with the legend " venite ad me omnes " . next to the main
building is the basilica of the sacred heart . immediately behind the basilica is the grotto , a
marian place of prayer and reflection . it is a replica of the grotto at lourdes , france where the
virgin mary reputedly appeared to saint bernadette soubirous in 1858 . at the end
of the main drive ( and in a direct line that connects through 3 statues and the gold dome ) , is a
simple , modern stone statue of mary .

[QUESTION]
to whom did the virgin mary allegedly appear in 1858 in lourdes france ?

[ANSWER]
saint bernadette soubirous

----------------------------------------------------------------------------------------------------
```

## Training and Evaluation

### Training from Scratch

`main.py` is the primary script for training and evaluating models on the provided datasets. Here is an example of training and testing the baseline QA model on SQuAD:

```bash
$ python3 main.py \
    --use_gpu \
    --model "baseline" \
    --model_path "squad_model.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --output_path "squad_predictions.txt" \
    --hidden_dim 256 \
    --bidirectional \
    --do_train \
    --do_test
```

Descriptions for all command line arguments are provided in `main.py`, but we will cover the ones listed above in detail:

| Flag                      | Description                                                                                                                                            |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--use_gpu`               | Use this flag if a GPU is available.                                                                                                                   |
| `--model`                 | The name of the model to invoke. Our baseline QA model is aptly named "baseline".                                                                      |
| `--model_path`            | Specifies the path where the model will be saved. It may be convenient to create a new directory for model checkpoints.                                |
| `--train_path`            | Path to the training dataset.                                                                                                                          |
| `--dev_path`              | Path to the development dataset.                                                                                                                       |
| `--output_path`           | Path where output predictions are stored. Here, predicted answer spans are indexed by question id (`qid`). Used by `evaluation.py` to calculate EM/F1. |
| `--hidden_dim`            | Controls the size of hidden vectors in the baseline model. Small/medium/large models use 128/256/512, respectively.                                    |
| `--bidirectional`         | Whether or not the LSTM/GRUs are bidirectional. We recommend using this flag for best performance.                                                     |
| `--do_train`              | Flag that enables training. Optional if you do NOT want to train.                                                                                      |
| `--do_test`               | Flag that enables testing. Optional if you do NOT want to test.                                                                                        |

Once training and testing are finished, the predictions are stored in `squad_predictions.txt`. Let's take a look at some of the answer spans our model produced:

```bash
$ head -n 10 squad_predictions.txt
{"qid": "b0626b3af0764c80b1e6f22c114982c1", "answer": "american football conference"}
{"qid": "8d96e9feff464a52a15e192b1dc9ed01", "answer": "national football league ( nfl ) for the 2015 season . the american football conference"}
{"qid": "190fdfbc068243a7a04eb3ed59808db8", "answer": "san francisco bay area at santa clara , california"}
{"qid": "e8d4a7478ed5439fa55c2660267bcaa1", "answer": "national football league"}
{"qid": "74019130542f49e184d733607e565a68", "answer": "golden anniversary \" with various gold - themed initiatives"}
{"qid": "3729174743f74ed58aa64cb7c7dbc7b3", "answer": "golden anniversary \" with various gold - themed initiatives"}
{"qid": "cc75a31d588842848d9890cafe092dec", "answer": "february 7 , 2016"}
{"qid": "7c1424bfa53a4de28c3ec91adfbfe4ab", "answer": "the american football conference"}
{"qid": "78a00c316d9e40e69711a9b5c7a932a0", "answer": "golden anniversary \" with various gold - themed initiatives"}
{"qid": "1ef03938ae3848798b701dd4dbb30bd9", "answer": "american football conference"}
```

We can benchmark our model using the evaluation script, passing in the SQuAD development dataset and predictions output file as arguments:

```bash
$ python3 evaluate.py \
    --dataset_path "datasets/squad_dev.jsonl.gz" \
    --output_path "squad_predictions.txt"
```

The single-model [state-of-the-art](https://rajpurkar.github.io/SQuAD-explorer/) on SQuAD is currently held by [ELECTRA](https://arxiv.org/abs/2003.10555), a pre-trained Transformer, which gets 88.11 EM and 91.42 F1 -- our baseline has a lot of room for improvement!

If you would like to evaluate the trained model on another dataset, simply point `args.dev_path` to another evaluation dataset. Once again, using the predictions output file, run the evaluation script to obtain EM/F1 scores. Note that, by default, `args.train_path` is still required because the vocabulary is built based using the words present in the training dataset. The SQuAD-trained model only gets 20.44/32.50 EM/F1 on NewsQA, which is substantially lower than the in-domain results:

```bash
$ python3 main.py \
    --use_gpu \
    --model "baseline" \
    --model_path "squad_model.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/newsqa_dev.jsonl.gz" \
    --output_path "newsqa_predictions.txt" \
    --hidden_dim 256 \
    --bidirectional \
    --do_test
$ python3 evaluate.py \
    --dataset_path "datasets/newsqa_dev.jsonl.gz" \
    --output_path "newsqa_predictions.txt"
{'EM': 20.44, 'F1': 32.50}
```

### Using Pre-trained Models

We provide the following pre-trained baseline model. Models are benchmarked using a NVIDIA Tesla K80 GPU on GCP:

| Model                        | Hidden Dim | Parameters | SQuAD Dev EM/F1 | Train & Test Time |     Output File    |
|------------------------------|:----------:|:----------:|:---------------:|:-----------------:|:------------------:|
| [`baseline_small_squad.pt`](https://cs.utexas.edu/~gdurrett/courses/online-course/fp/baseline_small_squad.pt)  |     128    |  16M  |   47.21/59.81   |       9m 47s      |       [NA]

Usage is largely the same as the previous examples; make sure to pass in the appropriate hidden dimension when invoking the script. For example, to use the baseline-small model for testing:

```bash
$ python3 main.py \
    --use_gpu \
    --model "baseline" \
    --model_path "baseline_small_squad.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --output_path "baseline_small_squad_preds.txt" \
    --hidden_dim 128 \
    --bidirectional \
    --do_test
```

## GCP Setup

[Google Cloud Platform (GCP)](https://cloud.google.com/) offers $300 in credits for 365 days towards most of their compute services, notably including access to GPUs. Using these resources is strictly optional. However, if you plan to pursue a final project that requires intensive model development and re-training the model and do not otherwise have access to a GPU, this resource may be very helpful. **It is critical that you turn off your GPUs when they are not being used to avoid burning through the credits quickly.**

To begin, follow the instructions on their website to begin your free trial. Note that although a credit card must be supplied, you will not be charged, and the billing account will be automatically deactivated once the credits expire.

You will need to upgrade your account to enable GPU usage and create a VM instance as the first steps. This workflow changes somewhat frequently, so you'll have to follow the latest instructions from GCP for how to do this. If you have trouble with these steps, contact the course staff for assistance.

**SSH into the instance.** Use the cloud SSH to log into the machine. Before starting development, there are a couple of checks we will do to make sure the GPU has been installed successfully. First, enter `nvidia-smi` in the terminal; you should see a Tesla K80 GPU occupying card 0. Second, enter the following commands in the Python REPL to ensure PyTorch can use the GPU:

```bash
$ python3
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.tensor(1).cuda()
tensor(1, device='cuda:0')
```

Now, you're ready to use your instance! Follow the instructions in the [Getting Started](#getting-started) section to setup the instance. Note that if you are using `virtualenv` to install dependencies, this is not installed by default, but `pip install virtualenv` should suffice. Lastly, if you run into any issues, please consult your peers or Google the error messages first before contacting the course staff.
