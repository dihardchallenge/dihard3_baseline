#!/usr/bin/env python
"""This script resegments an initial diarization using a variant of the VB-HMM
method of Landini et al. (2020). Specifically, it has been modified to allow
posterior of the zeroth order statistics following Singh et al. (2019).

It wraps the module ``VB_diarization_v2.py``, which is based on ``VB_diarization.py``
from:

    http://www.fit.vutbr.cz/~burget/VB_diarization.zip     

The wrapper script is, itself, based on the ``VB_resegmentation.py`` script from
the Kaldi CALLHOME diarization recipe:

    https://github.com/kaldi-asr/kaldi/blob/master/egs/callhome_diarization/v1/diarization/VB_resegmentation.py


References
----------
- F. Landini, S. Wang, M. Diez, L. Burget et al. (2020)."BUT System for the
  Second DIHARD Speech Diarization Challenge". Proc. ofICASSP 2020.
- P. Singh, Harsha Vardhana M A, S. Ganapathy, A. Kanagasundaram. (2019).
  "LEAP Diarization System for the Second DIHARD Challenge".
  Proc. of Interspeech 2019.
"""
# Revision history
# ----------------
# - Zili Huang  --  original vesion
# - Prachi Singh  --  minor edits
# - Neville Ryant  --  major refactoring, improved documentation, and PEP8
#   adherence
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys

import kaldi_io
import numpy as np

import VB_diarization_v2 as VB_diarization


def load_dubm(fpath):
    """Load diagonal UBM parameters.

    Parameters
    ----------
    fpath : Path
        Path to pickled UBM model.

    Returns
    -------
    m
    iE
    w
    """
    with open(fpath, "rb") as f:
        params = pickle.load(f)
    m = params["<MEANS_INVVARS>"] / params["<INV_VARS>"]
    iE = params["<INV_VARS>"]
    w = params["<WEIGHTS>"]
    return m, iE, w


def load_ivector_extractor(fpath):
    """Load ivector extractor parameters.

    Parameters
    ----------
    fpath : Path
        Path to pickled ivector extractor model.

    Returns
    -------
    v
    """
    with open(fpath, "rb") as f:
        params = pickle.load(f)
    m = params["M"]
    v = np.transpose(m, (2, 0, 1))
    return v


def load_frame_counts(fpath):
    """Load mapping from URIs to frame counts from ``fpath``.

    The file is expected to be in the format of a Kaldi ``utt2num_frames`` file;
    that is, two space-delimited columns:

    - URI
    - frame count
    """
    frame_counts = {}
    with open(fpath, 'r') as f:
        for line in f:
            uri, n_frames = line.strip().split()
            n_frames = int(n_frames)
            frame_counts[uri] = n_frames
    return frame_counts


@dataclass
class Segment:
    """Speech segment.

    Parameters
    ----------
    recording_id : str
        URI for recording segment is from.

    onset : int
        Index in frames of segment onset (0-indexed).

    offset : int
        Index in frames of segment offset (0-indexed).

    speaker_id : str
        Speaker id.
    """
    recording_id: str
    onset: int
    offset: int
    speaker_id: str

    @property
    def duration(self):
        """Duration in frames of segment."""
        return self.offset - self.onset + 1


def load_rttm(rttm_path, frame_counts, step=0.01, target_rec_ids=None):
    """Load recording segmentations from RTTM file.

    Parameters
    ----------
    rttm_path : Path
        Path to RTTM file.

    frame_counts : dict
        Mapping from recording ids to lengths in frames.

    step : float, optional
        Duration in seconds between onsets of frames.
        (Default: 0.01)

    target_rec_ids : iterable of str, optional
        Filter segments from recordings whose recording ids are not in
        ``target_rec_ids``.
        (Default: None)

    Returns
    -------
    recordings : dict
        Mapping from recording ids to speech segments.
    """
    recordings = defaultdict(list)
    with open(rttm_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            recording_id = fields[1]

            # Skip segments from non-target recordings.
            if target_rec_ids and recording_id not in target_rec_ids:
                continue

            # Check sane segment boundaries.
            onset = float(fields[3])
            offset = onset + float(fields[4])
            onset_frames = int(onset/step)
            offset_frames = int(offset/step)
            n_frames = frame_counts[recording_id]
            if offset_frames >= n_frames:
                offset_frames = n_frames - 1
                print(
                    f"WARNING: Speaker turn extends past end of recording. "
                    f"LINE: {line}")
            if not 0 <= onset_frames <= offset_frames:
                # Note that offset_frames was previously truncated to
                # at most the actual length of the recording as we
                # anticipate the initial diarization may be sloppy at
                # the edges.
                raise ValueError(
                    f"Impossible segment boundaries. LINE: {line}")

            # Create new speech segment.
            speaker_id = fields[7]
            recordings[recording_id].append(
                Segment(recording_id, onset_frames, offset_frames, speaker_id))

    return recordings


def get_labels(segs, n_frames):
    """Return frame-wise labeling corresponding to a segmentation.

    The resulting labeling is an an array whose ``i``-th entry provides the label
    for frame ``i``, which can be one of the following integer values:

    - 0:   indicates no speaker present (i.e., silence)
    - 1:   indicates more than one speaker present (i.e., overlapped speech)
    - n>1: integer id of the SOLE speaker present in frame

    Speakers are assigned integer ids >1 based on their first turn in the
    recording.

    Parameters
    ----------
    segs : iterable of Segment
        Recording segments.

    n_frames : int
        Length of recording in frames.

    Returns
    -------
    ref : ndarray, (n_frames,)
        Framewise speaker labels.
    """
    # Induce mapping from string speaker ids to integers > 1s.
    n_speakers = 0
    speaker_dict = {}
    for seg in segs:
        if seg.speaker_id in speaker_dict:
            continue
        n_speakers += 1
        speaker_dict[seg.speaker_id] = n_speakers + 1

    # Create reference frame labeling:
    # - 0: non-speech
    # - 1: overlapping speech
    # - n>1: speaker n
    # We use 0 to denote silence frames and 1 to denote overlapping frames.
    ref = np.zeros(n_frames, dtype=np.int32)
    for seg in segs:
        # Integer id of speaker.
        speaker_label = speaker_dict[seg.speaker_id]

        # Assign this label to all frames in the segment that are not
        # already assigned.
        for ind in range(seg.onset, seg.offset+1):
            if ref[ind] == speaker_label:
                # This shouldn't happen, but being paranoid in case the
                # initialization contains overlapping segments by same speaker.
                continue
            elif ref[ind] == 0:
                label = speaker_label
            else:
                # Overlapped speech.
                label = 1
            ref[ind] = label

    return ref


def print_diagnostics(labels):
    """Print diagnostics for labeling."""
    n_speakers = labels.max() - 1
    print(f"# SPEAKERS: {n_speakers}")
    n_frames = len(labels)
    n_sil_frames = np.sum(labels == 0)
    sil_prop = 100.* n_sil_frames / n_frames
    n_overlap_frames = np.sum(labels == 1)
    overlap_prop = 100.* n_overlap_frames / n_frames
    print(f"TOTAL: {n_frames} frames, "
          f"SILENCE: {n_sil_frames} frames ({sil_prop:.0f}%), "
          f"OVERLAP {n_overlap_frames} frames ({overlap_prop:.0f}%)")
    speaker_hist = np.bincount(labels)[2:]
    speaker_dist = speaker_hist/speaker_hist.sum()
    print(f"SPEAKER FREQUENCIES (DISCOUNTING OVERLAPS): "
          f"{np.array2string(speaker_dist, precision=2, suppress_small=True)}")
    print("")



def write_rttm_file(rttm_path, labels, channel=0, step=0.01, precision=2):
    """Write RTTM file.

    Parameters
    ----------
    rttm_path : Path
        Path to output RTTM file.

    labels : ndarray, (n_frames,)
        Array of predicted speaker labels. See ``get_labels`` for explanation.

    channel : int, optional
        Channel (0-indexed) to output segments for.
        (Default: 0)

    step : float, optional
        Duration in seconds between onsets of frames.
        (Default: 0.01)

    precision : int, optional
        Output ``precision`` digits.
        (Default: 2)
    """
    rttm_path = Path(rttm_path)

    # Determine indices of onsets/offsets of speaker turns.
    is_cp = np.diff(labels, n=1, prepend=-999, append=-999) != 0
    cp_inds = np.nonzero(is_cp)[0]
    bis = cp_inds[:-1]  # Last changepoint is "fake".
    eis = cp_inds[1:] -1

    # Write turns to RTTM.
    with open(rttm_path, 'w') as f:
        for bi, ei in zip(bis, eis):
            label = labels[bi]
            if label < 2:
                # Ignore non-speech and overlapped speech.
                continue
            n_frames = ei - bi + 1
            duration = n_frames*step
            onset = bi*step
            recording_id = rttm_path.stem
            line = f'SPEAKER {recording_id} {channel} {onset:.{precision}f} {duration:.{precision}f} <NA> <NA> speaker{label} <NA> <NA>\n'
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description='VB Resegmentation')
    parser.add_argument(
        'data_dir', type=Path, help='Subset data directory')
    parser.add_argument(
        'init_rttm_filename', type=Path,
        help='The RTMM file to initialize the VB system from; usually the result '
             'from the AHC step')
    parser.add_argument(
        'output_dir', type=Path, help='Output directory')
    parser.add_argument(
        'dubm_model', type=Path, help='Path to the diagonal UBM model')
    parser.add_argument(
        'ie_model', type=Path, help='Path to the ivector extractor model')
    parser.add_argument(
        '--max-speakers', metavar='SPEAKERS', type=int, default=10,
        help='Set the maximum of speakers for a recording (default: %(default)s)')
    parser.add_argument(
        '--max-iters', metavar='ITER', type=int, default=10,
        help='Set maximum number of algorithm iterations (default: %(default)s)')
    parser.add_argument(
        '--downsample', metavar='FACTOR', type=int, default=25,
        help='Downsample input by FACTOR before applying VB-HMM '
             '(default: %(default)s)')
    parser.add_argument(
        '--alphaQInit', metavar='ALPHA', type=float, default=100.0,
        help='Initialize Q from Dirichlet distribution with concentration '
             'parameter ALPHA (default: %(default)s)')
    parser.add_argument(
        '--sparsityThr', metavar='SPARSITY', type=float, default=0.001,
        help='Set occupations smaller than SPARSITY to 0.0; saves memory as'
             'the posteriors are represented by sparse matrix '
             '(default: %(default)s)')
    parser.add_argument(
        '--epsilon', metavar='EPS', type=float, default=1e-6,
        help='Stop iterating if objective function improvement is <EPS '
             '(default: %(default)s)')
    parser.add_argument(
        '--minDur', metavar='FRAMES', type=int, default=1,
        help='Minimum number of frames between speaker turns. This constraint '
             'is imposed via linear chains of HMM states corresponding to each '
             'speaker. All the states in a chain share the same output '
             'distribution (default: %(default)s')
    parser.add_argument(
        '--loopProb', metavar='PROB', type=float, default=0.9,
        help='Probability of not switching speakers between frames '
             '(default: %(default)s)')
    parser.add_argument(
        '--statScale', metavar='FACTOR', type=float, default=0.2,
        help='Scaling factor for sufficient statistics collected using UBM '
             '(default: %(default)s)')
    parser.add_argument(
        '--llScale', metavar='FACTOR', type=float, default=1.0,
        help='Scaling factor for UBM likelihood; values <1.0 make atribution of '
             'frames to UBM componets more uncertain (default: %(default)s)')
    parser.add_argument(
        '--step', metavar='SECONDS', type=float, default=0.01,
        help='Duration in seconds between frame onsets (default: %(default)s)')
    parser.add_argument(
        '--channel', metavar='CHANNEL', type=int, default=0,
        help='In output RTTM files, set channel field to CHANNEL '
             '(default: %(default)s)')
    parser.add_argument(
        '--initialize', default=False, action='store_true',
        help='Initialize speaker posteriors from RTTM')
    parser.add_argument(
        '--seed', metavar='SEED', type=int, default=1036527419,
        help='seed for RNG (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # Might as well log the paramater values.
    print(args)

    # Set NumPy RNG to ensure reproducibility.
    np.random.seed(args.seed)

    # Load the diagonal UBM and i-vector extractor.
    m, iE, w = load_dubm(args.dubm_model)
    V = load_ivector_extractor(args.ie_model)

    # Load the MFCC features
    feats_dict = {}
    feats_scp_path = Path(args.data_dir, "feats.scp")
    for key,mat in kaldi_io.read_mat_scp(str(feats_scp_path)):
        feats_dict[key] = mat

    # Load segments for target recordings.
    frame_counts = load_frame_counts(
        Path(args.data_dir, "utt2num_frames"))
    recording_ids = sorted(frame_counts.keys())
    recordings = load_rttm(
        args.init_rttm_filename, frame_counts, step=args.step,
        target_rec_ids=recording_ids)

    print("------------------------------------------------------------------------")
    print("")

    # Resegment the recordings serially.
    for recording_id in recording_ids:
        # Convert the initial segmentation (e.g., from AHC) into an array of
        # frame-level integer labels, where:
        # - 0: non-speech
        # - 1: overlapping speech
        # - n>1: speaker n
        # We use 0 to denote silence frames and 1 to denote overlapping frames.
        init_labels = get_labels(
            recordings[recording_id], frame_counts[recording_id])
        print(f'INITIAL SEGMENTATION DIAGNOSTICS for "{recording_id}"')
        print_diagnostics(init_labels)


        # Grab the corresponding features.
        X = feats_dict[recording_id].astype(np.float64)

        # Drop frames corresponding to silence and overlapped speech.
        mask = (init_labels >= 2)
        X_masked = X[mask]
        init_labels_masked = init_labels[mask]
        init_labels_masked -= 2  # So the labeling starts at 0.
        if len(init_labels) == 0:
            print(
                f"Warning: the initial segmentation for {recording_id} has no "
                f"non-overlapping speech.")
            continue

        # Initialize the posterior of each speaker based on the initial
        # segmentation.
        if args.initialize:
            q = VB_diarization.frame_labels2posterior_mx(
                init_labels_masked, args.max_speakers)
        else:
            q = None
            print("RANDOM INITIALIZATION\n")

        # Perform VB resegmentation of the NON-OVERLAPPED speech.
        #
        # q  - S x T matrix of speaker posteriors whose i-th row is the
        #      attribution of frame i to the S possible speakers
        #      (args.max_speakers). T is the total number of frames.
        # sp - S dimensional column vector of ML learned speaker priors. Ideally,
        #      these should allow to estimate # of speaker in the recording as the
        #      probabilities of the redundant speaker should converge to zero.
        # Li - values of auxiliary function (and DER and frame cross-entropy
        #      between q and reference if 'ref' is provided) over iterations.
        q_out, sp_out, L_out = VB_diarization.VB_diarization(
            X_masked, recording_id, m, iE, w, V, sp=None, q=q,
            maxSpeakers=args.max_speakers, maxIters=args.max_iters, VtiEV=None,
            downsample=args.downsample, alphaQInit=args.alphaQInit,
            sparsityThr=args.sparsityThr, epsilon=args.epsilon,
            minDur=args.minDur, loopProb=args.loopProb, statScale=args.statScale,
            llScale=args.llScale, ref=None, plot=False)

        # Reconstruct a labeling relative to the original UNMASKED frames. Note
        # that in the following, we simply ignore overlap frames entirely and
        # treat them as silence. When the initial segmentation has no overlaps
        # (e..g, output of AHC) this is not problematic, but it is less than
        # ideal if there initial segmentation accounts for overlaps.
        predicted_labels_masked = np.argmax(q_out, 1) + 2
        predicted_labels = (np.zeros(len(mask))).astype(int)
        predicted_labels[mask] = predicted_labels_masked

        # More diagnostics.
        print(f'RESEGMENTATION DIAGNOSTICS for "{recording_id}"')
        print_diagnostics(predicted_labels)
        print(f"LEARNED SPEAKER PRIORS: "
              f"{np.array2string(sp_out, precision=3, suppress_small=True)}")
        aux_loss = np.squeeze(L_out)
        print('AUX LOSS VALUES')
        for n, l in enumerate(aux_loss):
            print(f"ITER: {n}, LOSS: {l}")


        # Create the output rttm file and compute the DER after re-segmentation.
        write_rttm_file(
            Path(args.output_dir, "per_file_rttm", recording_id + '.rttm'),
            predicted_labels, channel=args.channel, step=args.step, precision=2)
        print("")
        print("------------------------------------------------------------------------")
        print("")


if __name__ == "__main__":
    main()
