"""Prints QA examples.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import argparse
import random
import textwrap
import stanza

from termcolor import colored

from data import QADataset


RULE_LENGTH = 100
DOC_WIDTH = 100
TEXT_WRAPPER = textwrap.TextWrapper(width=DOC_WIDTH)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--path',
    type=str,
    default='datasets/squad_dev.jsonl.gz',
    required=False,
    help='path to display samples from',
)
parser.add_argument(
    '--samples',
    type=int,
    default=10,
    required=False,
    help='number of samples to visualize',
)
parser.add_argument(
    '--shuffle',
    action='store_true',
    help='whether to shuffle samples before displaying',
)
parser.add_argument(
    '--max_context_length',
    type=int,
    default=384,
    help='maximum context length (do not change!)',
)
parser.add_argument(
    '--max_question_length',
    type=int,
    default=64,
    help='maximum question length (do not change!)',
)


def _build_string(tokens):
    """Builds string from token list."""

    return ' '.join(tokens)


def _color_context(context, answer_start, answer_end):
    """Colors answer span with bold + underline red within the context."""

    tokens = []

    i = 0
    while i < len(context):
        if i == answer_start:
            span = _build_string(context[answer_start:(answer_end + 1)])
            tokens.append(
                colored(span, 'red', attrs=['bold', 'underline']),
            )
            i = answer_end + 1
        else:
            tokens.append(context[i])
            i += 1

    lines = TEXT_WRAPPER.wrap(text=' '.join(tokens))

    return '\n'.join(lines)


def main(args):
    """Visualization of contexts, questions, and colored answer spans."""

    # Load dataset, and optionally shuffle.
    dataset = QADataset(args, args.path)
    samples = dataset.samples
    if args.shuffle:
        random.shuffle(samples)

    vis_samples = samples[:args.samples]

    print()
    print('-' * RULE_LENGTH)
    print()

    # Visualize samples.
    for (qid, context, question, answer_start, answer_end) in vis_samples[:10]:
      cxt = _build_string(context)
      print(cxt)
      stanza.download('en') 
      en_nlp = stanza.Pipeline('en')
      en_doc = en_nlp(cxt)

      for i, sent in enumerate(en_doc.sentences):
        print(f"[Sentence {i+1}")
        for word in sent.words:
          print("{:12s}\t{:12s}\t{:6s}\t{:d}\t{:12s}".format(word.text, word.lemma, word.pos, word.head, word.deprel))
          print("")

      print("Mention text\tType\tStart-End")
      for ent in en_doc.ents:
        print("{}\t{}\t{}-{}".format(ent.text, ent.type, ent.start_char, ent.end_char))
    


if __name__ == '__main__':
    main(parser.parse_args())
