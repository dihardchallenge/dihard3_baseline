#!/usr/bin/env python
"""Prepare targets directory for SAD training from segments file.

Given a Kaldi segments file ``segments`` descibing the locations of all speech
segments and an ``utt2num_frames`` file, the following command will create a
directory ``targets_dir`` containg archives of frame-level targets for SAD
training:

    python3 prepare_sad_targets.py segments utt2num_frames targets_dir

The output targets_dir contains the following files:

- targets.scp  --  script file mapping recording ids to entries in
  ``targets.ark``
- targets.ark  --  Kaldi archive mapping recoding ids to matrices of
  frame-level targets

The targets matrices are NUM_FRAMES x 3 sized matrices whose rows are posterior
distributions over the following classes:

- COLUMN 0: speech
- COLUMN 1: non-speech
- COLUMN 2: garbage (i.e., everything else or data where or data for which we
  are unclear about the label)

By default, probability mass is never assigned to the garbage class. If
non-scoring regions are known and contained in a UEM file using the same
recording ids as in ``segments`` and ``utt2num_frames``, the script can be
directed to assign the corresponding frames to the garbage model:

    python3 prepare_sad_targets.py --uem all.uem segments utt2num_frames targets_dir

If the ``--subsampling-factor`` flag is provided with an integer ``N`` > 1,
then the frames will be subsampled by a factor of ``N`` by dividing the frames
into consecutive blocks of ``N`` frames and averaging within each block.

For more details regarding the structure of the targets archives and the
targets directory, see:

    https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/segmentation/lats_to_targets.sh
"""
from __future__ import division
import argparse
from collections import defaultdict, namedtuple
import numbers
from pathlib import Path
import sys

import kaldi_io
import numpy as np
from scipy.ndimage import convolve1d



def _seconds_to_frames(t, step):
    """Convert ``t`` from seconds to frames.

    Parameters
    ----------
    t : float
        Time in seconds.

    step : float
        Frame step size in seconds.

    Returns
    -------
    int
        Number of frames.
    """
    t = np.array(t, dtype=np.float32, copy=False)
    return np.array(t/step, dtype=np.int32)


def subsample_frames(frames, subsample_factor=1):
    """Subsample frames by a factor of ``subsample_factor``.

    Similar to simply taking every ``subsample_factor``-th frame except that
    it averages over blocks of ``subsample_factor`` frames rather than picking
    a single point.

    Parameters
    ----------
    frames : ndarray, (n_frames, n_dims)
        Frames to subsample.

    subsample_factor : int, optional
        Factor to subsample by.
        (Default: 1)

    Returns
    -------
    ndarray, (n_subsampled_frames, n_dims)
        Subsampled frames.

    References
    ----------
    https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/segmentation/internal/resample_targets.py
    """
    if not isinstance(subsample_factor, numbers.Integral):
        raise ValueError('"subsample_factor" must be integer >= 1.')
    if subsample_factor <= 1:
        return frames

    # Average frames using sliding windows of size subsample_factor with left
    # edge of first window starting at origin.
    filter = np.ones(subsample_factor) / subsample_factor
    fwidth = subsample_factor//2
    smoothed_frames = convolve1d(
        frames, filter, origin=-fwidth, axis=0, mode='nearest')

    # Select every subsample_factor-th smoothed frame.
    n_frames, dim = frames.shape
    n_frames_sub = (n_frames + subsample_factor - 1) // subsample_factor
    inds = np.arange(n_frames_sub)*subsample_factor + subsample_factor
    inds = np.minimum(inds, n_frames - 1)
    frames_sub = smoothed_frames[inds, ]

    return frames_sub


Segmentation = namedtuple('Segmentation', ['onsets', 'offsets'])


def load_speech_segments(segments_path, step=0.01):
    """Load speech segments for recordings.

    Parameters
    ----------
    segments_path : Path
        Path to Kaldi segments file.

    step : float, optonal
        Frame step size in seconds.
        (Default: 0.01)

    Returns
    -------
    segments : dict
        Mapping from recording ids to speech segments, stored as
        ``Segmentation`` instances.
    """
    onsets = defaultdict(list)
    offsets = defaultdict(list)
    with open(segments_path, 'r') as f:
        for line in f:
            utt_id, rec_id, onset, offset = line.strip().split()
            onsets[rec_id].append(float(onset))
            offsets[rec_id].append(float(offset))
    segments = {}
    for rec_id in onsets:
        segments[rec_id] = Segmentation(
            _seconds_to_frames(onsets[rec_id], step),
            _seconds_to_frames(offsets[rec_id], step))
    return segments


def load_scored_segments(uem_path, step=0.01):
    """Load scoring regions for recording from UEM file.

    Parameters
    ----------
    uem_path : Path
        Path to UEM file.

    step : float, optonal
        Frame step size in seconds.
        (Default: 0.01)

    Returns
    -------
    segments : dict
        Mapping from recording ids to nonscoring regions, stored as
        ``Segmentation`` instances.
    """
    onsets = defaultdict(list)
    offsets = defaultdict(list)
    with open(uem_path, 'r') as f:
        for line in f:
            rec_id, channel, onset, offset = line.strip().split()
            onsets[rec_id].append(float(onset))
            offsets[rec_id].append(float(offset))
    segments = {}
    for rec_id in onsets:
        segs = zip(onsets[rec_id], offsets[rec_id])
        segs = sorted(segs)
        onsets_, offsets_ = zip(*segs)
        segments[rec_id] = Segmentation(
            _seconds_to_frames(onsets_, step),
            _seconds_to_frames(offsets_, step))
    return segments


def main():
    parser = argparse.ArgumentParser(
        description='Prepare targets for SAD training.', add_help=True)
    parser.add_argument(
        'segments', type=Path, help='path to segments file')
    parser.add_argument(
        'utt2num_frames', metavar='utt2num-frames', type=Path,
        help='path to utt2num_frames file')
    parser.add_argument(
        'targets_dir', metavar='targets-dir', type=Path,
        help='path to output targets directory')
    parser.add_argument(
        '--uem', metavar='UEM', type=Path, default=None,
        help='path to UEM file')
    parser.add_argument(
        '--frame-step', metavar='STEP', type=float, default=0.01,
        help='time between frames is STEP seconds')
    parser.add_argument(
        '--subsampling-factor', metavar='FACTOR', type=int, default=1,
        help='subsample frames by factor of FACTOR')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    args.targets_dir.mkdir(parents=True, exist_ok=True)
    speech_segments = load_speech_segments(args.segments)
    if args.uem is not None:
        scored_segments = load_scored_segments(args.uem, args.frame_step)
    targets_scp_path = Path(args.targets_dir, 'targets.scp')
    targets_ark_path = Path(args.targets_dir, 'targets.ark')
    ark_scp_output = ('ark:| copy-feats --compress=true ark:- ark,scp:{0},{1}'.format(targets_ark_path,targets_scp_path))
    with open(args.utt2num_frames, 'r') as f:
        with kaldi_io.open_or_fd(ark_scp_output,'wb') as g:
            for line in f:
                rec_id, n_frames = line.strip().split()
                n_frames = int(n_frames)

                # Create targets.
                targets = np.ones(n_frames, dtype='int32')
                if rec_id in speech_segments:
                    speech_onsets, speech_offsets  = speech_segments[rec_id]
                    for onset, offset in zip(speech_onsets, speech_offsets):
                        targets[onset:offset+1] = 0
                if rec_id in scored_segments:
                    scored_onsets, scored_offsets = scored_segments[rec_id]
                    is_scored = np.zeros_like(targets, dtype=np.bool)
                    for onset, offset in zip(scored_onsets, scored_offsets):
                        is_scored[onset:offset+1] = True
                    targets[~is_scored] = 2

                # Convert targets to one-hot.
                targets = np.eye(3)[targets]

                # Subsample by requested factor.
                targets = subsample_frames(targets, args.subsampling_factor)

                # Write to script/archive files.
                kaldi_io.write_mat(g, targets, key=rec_id)


if __name__ == '__main__':
    main()
