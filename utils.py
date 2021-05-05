"""General utilities for training.

Author:
    Shrey Desai
"""

import os
import json
import gzip
import pickle

import torch
from tqdm import tqdm
import numpy as np



def cuda(args, tensor):
    """
    Places tensor on CUDA device (by default, uses cuda:0).

    Args:
        tensor: PyTorch tensor.

    Returns:
        Tensor on CUDA device.
    """
    if args.use_gpu and torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def unpack(tensor):
    """
    Unpacks tensor into Python list.

    Args:
        tensor: PyTorch tensor.

    Returns:
        Python list with tensor contents.
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy().tolist()


def load_dataset(path):
    """
    Loads MRQA-formatted dataset from path.

    Args:
        path: Dataset path, e.g. "datasets/squad_train.jsonl.gz"

    Returns:
        Dataset metadata and samples.
    """
    with gzip.open(path, 'rb') as f:
        elems = [
            json.loads(l.rstrip())
            for l in tqdm(f, desc=f'loading \'{path}\'', leave=False)
        ]
    meta, samples = elems[0], elems[1:]
    return (meta, samples)


def load_cached_embeddings(path):
    """
    Loads embedding from pickle cache, if it exists, otherwise embeddings
    are loaded into memory and cached for future accesses.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    bare_path = os.path.splitext(path)[0]
    cached_path = f'{bare_path}.pkl'
    if os.path.exists(cached_path):
        return pickle.load(open(cached_path, 'rb'))
    embedding_map = load_embeddings(path)
    pickle.dump(embedding_map, open(cached_path, 'wb'))
    return embedding_map


def load_embeddings(path):
    """
    Loads GloVe-style embeddings into memory. This is *extremely slow* if used
    standalone -- `load_cached_embeddings` is almost always preferable.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    embedding_map = {}
    with open(path) as f:
        next(f)  # Skip header.
        for line in f:
            try:
                pieces = line.rstrip().split()
                embedding_map[pieces[0]] = [float(weight) for weight in pieces[1:]]
            except:
                pass
    return embedding_map


def search_span_endpoints(start_probs, end_probs, window=15):
    """
    Finds an optimal answer span given start and end probabilities.
    Specifically, this algorithm finds the optimal start probability p_s, then
    searches for the end probability p_e such that p_s * p_e (joint probability
    of the answer span) is maximized. Finally, the search is locally constrained
    to tokens lying `window` away from the optimal starting point.

    Args:
        start_probs: Distribution over start positions.
        end_probs: Distribution over end positions.
        window: Specifies a context sizefrom which the optimal span endpoint
            is chosen from. This hyperparameter follows directly from the
            DrQA paper (https://arxiv.org/abs/1704.00051).

    Returns:
        Optimal starting and ending indices for the answer span. Note that the
        chosen end index is *inclusive*.
    """
    max_start_index = start_probs.index(max(start_probs))
    max_end_index = -1
    max_joint_prob = 0.

    for end_index in range(len(end_probs)):
        if max_start_index <= end_index <= max_start_index + window:
            joint_prob = start_probs[max_start_index] * end_probs[end_index]
            if joint_prob > max_joint_prob:
                max_joint_prob = joint_prob
                max_end_index = end_index

    return (max_start_index, max_end_index)
def top_n_spans(start_probs, end_probs, question, passage, window, n=4):
    top_k_starts = start_probs.argsort()[-min(2,len(start_probs)):][::-1]

    span_list = []
    for idx in top_k_starts:
        end_possibilities = end_probs[idx:]
        top_ends = end_possibilities.argsort()[-min(n//2,len(end_possibilities)):][::-1]
        top_ends += idx
        for e_idx in top_ends:
            span_lst.append((idx,e_idx))
    return span_list


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])    


def find_ngrams_upto(input_list, n):
    multigrams = [find_ngrams(input_list,i) for i in range(1,min(n + 1,len(input_list)))]
    multigrams = set(multigrams)
    return multigrams

def top_n_spans(start_probs, end_probs, question, passage, window, n=4):
    top_k_starts = start_probs.argsort()[-min(2,len(start_probs)):][::-1]

    span_list = []
    for idx in top_k_starts:
        end_possibilities = end_probs[idx:]
        top_ends = end_possibilities.argsort()[-min(n//2,len(end_possibilities)):][::-1]
        top_ends += idx
        for e_idx in top_ends:
            span_lst.append((idx,e_idx))
    return span_list


def unigram_span_endpoints(start_probs,end_probs,question,passage,window=15):

    start_probs = np.array(start_probs)
    end_probs = np.array(end_probs)
    top_spans = top_n_spans(start_probs,end_probs,question,passage,window)

    #Normal
    for span in top_spans:
        start, end = span
        words = passage[start:end + 1]
    return top_spans[0][0], top_spans[0][1]

    #question length restriction
    # for span in top_spans:
    #     start, end = span
    #     delta = end - start
    #     if delta < question.length:
    #        words = passage[start:end + 1]
    #     else:
    #        words = passage[start:start+question.length]
    # return top_spans[0][0], top_spans[0][1]


def multigram_span_endpoints(start_probs, end_probs,question,passage,window=15):
    q_multigrams = find_ngrams_upto(question, 2)
    start_probs = np.array(start_probs)
    end_probs = np.array(end_probs)
    top_spans = top_n_spans(start_probs, end_probs, question, passage, window)
    
    best_overlap = set()
    best_start = top_spans[0][0]
    best_end = top_spans[0][1]

    ## Normal
    for span in top_spans:
        start, end = span
        delta = end - start
        words = passage[start:end + 1]
        multigrams = find_ngrams_upto(words, 2)
        overlap = multigrams.intersection(q_multigrams)
        if len(overlap) > len(best_overlap):
            best_overlap = overlap
            best_start = start
            best_end = end
    return best_start, best_end

    ##  Question length restriction
    # for span in top_spans:
    #     start, end = span
    #     delta = end - start
    #     if delta < question.length:
    #         words = passage[start:end + 1]
    #     else:
    #         words = passage[start:start+question.length]
    #     multigrams = find_ngrams_upto(words, 2)
    #     overlap = multigrams.intersection(q_multigrams)
    #     if len(overlap) > len(best_overlap):
    #         best_overlap = overlap
    #         best_start = start
    #         best_end = end
    # return best_start, best_end
