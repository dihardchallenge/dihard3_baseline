#!/usr/bin/env python
"""Prepare targets directory for SAD training from segments file.

Given a Kaldi segments file ``segments`` describing the locations of all speech
segments and an ``utt2num_frames`` file, the following command will create a
directory ``targets_dir`` containing archives of frame-level targets for SAD
training:

    python prepare_sad_targets.py segments utt2num_frames targets_dir

The output targets_dir contains the following files:

- targets.scp  --  script file mapping recording ids to entries in
  ``targets.ark``
- targets.ark  --  Kaldi archive mapping recoding ids to matrices of
  frame-level targets

The targets matrices are NUM_FRAMES x 3 matrices whose rows are posterior
distributions over the following classes:

- COLUMN 0: non-speech
- COLUMN 1: speech
- COLUMN 2: garbage (i.e., everything else, which might include frames where we
  are unclear about the label)

By default, probability mass is never assigned to the garbage class. If the
recordings are only partially annotated, frames outside those regions can be
assigned to the garbage class by providing a second segments file listing the
annotated regions of each recording via the ``--annotated-segments`` flag; e.g.:

    python prepare_sad_targets.py --annotated-segments ann_segments segments utt2num_frames targets_dir

If the ``--subsampling-factor`` flag is provided with an integer ``N`` > 1,
then the frames will be subsampled by a factor of ``N`` by dividing the frames
into consecutive blocks of ``N`` frames and averaging within each block.

For more details regarding the structure of the targets archives and the
targets directory, see:

    https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/segmentation/lats_to_targets.sh
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
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
    t : ndarray
        Array of times in seconds.

    step : float
        Frame step size in seconds.

    Returns
    -------
    ndarray
        Frame indices corresponding to ``t``.
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
    fwidth = subsample_factor // 2
    smoothed_frames = convolve1d(
        frames, filter, origin=-fwidth, axis=0, mode='nearest')

    # Select every subsample_factor-th smoothed frame.
    n_frames, dim = frames.shape
    n_frames_sub = (n_frames + subsample_factor - 1) // subsample_factor
    inds = np.arange(n_frames_sub)*subsample_factor + subsample_factor
    inds = np.minimum(inds, n_frames - 1)
    frames_sub = smoothed_frames[inds, ]

    return frames_sub


@dataclass
class Segmentation:
    """Segmentation.

    Stores onsets/offsets of segments from a recording.

    Parameters
    ----------
    recording_id : str
        Recording segmentation is from.

    onsets: ndarray, (n_frames,)
        ``onsets[i]`` is the onset in frames of the ``i``-th segment.

    offsets: ndarray, (n_frames,)
        ``offsets[i]`` is the offset in frames of the ``i``-th segment

    step : float, optional
        Delta in seconds between onsets of consecutive frames.
        (Default: 0.01)
    """
    recording_id: str
    onsets: np.ndarray
    offsets: np.ndarray
    step: float=0.01

    def __post_init__(self):
        self.onsets = np.array(self.onsets, dtype=np.int32, copy=False)
        self.offsets = np.array(self.offsets, dtype=np.int32, copy=False)
        if len(self.onsets) != len(self.offsets):
            raise ValueError(
                f'"onsets" and "offsets" must have same length: '
                f'{len(self.onsets)} != {len(self.offsets)}.')
        n_bad = sum(self.durations <=0)
        if n_bad:
            raise ValueError(
                'One or more segments has non-positive duration.')

    @property
    def durations(self):
        """Segment durations in frames."""
        return self.offsets - self.onsets

    @property
    def num_segments(self):
        """Number of segments."""
        return len(self.onsets)

    @staticmethod
    def read_segments_file(segments_path, step=0.01):
        """Load speech segments for recordings from Kaldi ``segments`` file.

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
                utterance_id, recording_id, onset, offset = line.strip().split()
                onsets[recording_id].append(float(onset))
                offsets[recording_id].append(float(offset))
        segments = {}
        for recording_id in onsets:
            segments[recording_id] = Segmentation(
                recording_id,
                _seconds_to_frames(onsets[recording_id], step),
                _seconds_to_frames(offsets[recording_id], step),
                step)
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
        '--annotated-segments', metavar='PATH', type=Path, default=None,
        help='path to annotated segments file')
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

    # Load segmentations.
    speech_segments = Segmentation.read_segments_file(
        args.segments, step=args.frame_step)
    annotated_segments = {}
    if args.annotated_segments is not None:
        annotated_segments = Segmentation.read_segments_file(
            args.annotated_segments, step=args.frame_step)

    # Convert to targets.
    targets_scp_path = Path(args.targets_dir, 'targets.scp')
    targets_ark_path = Path(args.targets_dir, 'targets.ark')
    ark_scp_output = (f'ark:| copy-feats --compress=true ark:- '
                      f'ark,scp:{targets_ark_path},{targets_scp_path}')
    with open(args.utt2num_frames, 'r') as f:
        with kaldi_io.open_or_fd(ark_scp_output,'wb') as g:
            for line in f:
                recording_id, n_frames = line.strip().split()
                n_frames = int(n_frames)

                # Create targets.
                targets = np.zeros(n_frames, dtype='int32')
                if recording_id in speech_segments:
                    segmentation = speech_segments[recording_id]
                    for onset, offset in zip(
                            segmentation.onsets, segmentation.offsets):
                        targets[onset:offset+1] = 1
                if recording_id in annotated_segments:
                    segmentation = annotated_segments[recording_id]
                    is_scored = np.zeros_like(targets, dtype=np.bool)
                    for onset, offset in zip(
                            segmentation.onsets, segmentation.offsets):
                        is_scored[onset:offset+1] = True
                    targets[~is_scored] = 2

                # Convert targets to one-hot.
                targets = np.eye(3)[targets]

                # Subsample by requested factor.
                targets = subsample_frames(targets, args.subsampling_factor)

                # Write to script/archive files.
                kaldi_io.write_mat(g, targets, key=recording_id)


if __name__ == '__main__':
    main()
