#!/usr/bin/env python
"""Writes to STDOUT a transformation matrix for converting a 3x1 vector of
non-speech/speech/garbage posteriors to a 2x1 speech/non-speech pseudo-likelihood
vector.

Assuming the class priors are stored in ``post_output.vec`` as a Kaldi matrix in
text format, the following command will create the necessary transformation matrix:

    python local/segmentation/get_transform_probs_mat.py post_output.vec

E.g., if the non-speech prior is 0.2 and the speech prior 0.8, this will output:

     [
    5.000000 0.000000 0.000000
    0.000000 1.250000 0.000000
    0.000000 0.000000 0.000000 ]

By default, the transformation produces pseudo-likelihoods by dividing each class
posterior by its corresponding prior. However, you may optionally scale the
resulting speech pseudo-likelihood by passing a weighting factor via the
``--speech-likelihood-weight`` flag. E.g:

    python local/segmentation/get_transform_probs_mat.py --speech-likelihood-weight 10. post_output.vec

Values >1 will increase false alarm rate and decrease miss rate, while values <1
will decrase false alarm rate and increase miss rate.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, 'steps')

import numpy as np

import libs.common as common_lib


def get_args():
    parser = argparse.ArgumentParser(
        description='This script writes to STDOUT a transformation matrix'
                    'to convert a 3x1 probability vector of '
                    'non-speech/speech/garbage posteriors to a 2x1 '
                    'speech/non-speech pseudo-likelihood vector.',
        add_help=True)
    parser.add_argument(
        'priors', type=Path,
        help='path to Kaldi matrix containing priors vector; 3x1 vector of '
             'non-speech/speech/garbage priors')
    parser.add_argument(
        '--speech-likelihood-weight', metavar='WEIGHT', type=float, default=1.0,
        help='scale speech pseudo-likelihood by WEIGHT (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    return args


def run(args):
    # Load priors.
    # - priors[0]  --  prior probability of non-speech
    # - priors[1]  --  prior probability of speech
    # - priors[2]  --  prior probability of garbage; ignored
    priors = common_lib.read_matrix_ascii(args.priors)
    if len(priors) != 0 and len(priors[0]) != 3:
        raise RuntimeError(f'Invalid dimension for priors {priors}')
    priors = np.squeeze(np.array(priors, dtype=np.float64))

    # Create matrix that converts posteriors to likelihoods by dividing by
    # normalized priors.
    pmass = priors[0] + priors[1]  # Total mass devoted to speech/non-speech.
    priors /= pmass
    transform_mat = np.diag(1 / priors)
    transform_mat[2, 2] = 0.0  # Ignore garbage entirely
    transform_mat[1, 1] *= args.speech_likelihood_weight
    common_lib.write_matrix_ascii(sys.stdout, transform_mat)


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
