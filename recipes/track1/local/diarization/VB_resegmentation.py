#!/usr/bin/env python
import argparse
from collections import namedtuple
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



Segment = namedtuple('Segment', ['onset', 'offset', 'speaker_id'])

def create_ref_file(recording_id, rec2num_frames, full_rttm_path, step=0.01):
    """Return frame-wise labeling for  based on the initial diarization.

    The resulting labeling is an an array whose ``i``-th entry provides the label
    for frame ``i``, which can be one of the following integer values:

    - 0:   indicates no speaker present (i.e., silence)
    - 1:   indicates more than one speaker present (i.e., overlapped speech)
    - n>1: integer id of the SOLE speaker present in frame

    Speakers are assigned integer ids >1 based on their first turn in the
    recording.

    Parameters
    ----------
    recording_id : str
        URI of recording to extract labeling for.

    rec2num_frames : dict
        Mapping from recording URIs to lengths in frames.

    full_rttm_filename : Path
        Path to RTTM containing **ALL** segments for **ALL** recordings.

    step : float, optional
        Duration in seconds between onsets of frames.
        (Default: 0.01)

    Returns
    -------
    ref : ndarray, (n_frames,)
        Framewise speaker labels.
    """
    n_frames = rec2num_frames[recording_id]

    # Load speech segments for target recording from RTTM.
    segs = []
    with open(full_rttm_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if fields[1] != recording_id:
                # Skip segments from other recordings.
                continue
            onset = float(fields[3])
            offset = onset + float(fields[4])
            onset_frames = int(onset/step)
            offset_frames = int(offset/step)
            if offset_frames >= n_frames:
                offset_frames = n_frames - 1
                print(
                    f"WARNING: Speaker turn extends past end of recording. "
                    f"LINE: {line}")
            speaker_id = fields[7]
            if not 0 <= onset_frames <= offset_frames:
                # Note that offset_frames was previously truncated to
                # at most the actual length of the recording as we
                # anticipate the initial diarization may be sloppy at
                # the edges.
                raise ValueError(
                    f"Impossible segment boundaries. LINE: {line}")
            segs.append(
                Segment(onset_frames, offset_frames, speaker_id))

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

    # Diagnostics.
    print(f"{n_speakers} SPEAKERS IN {recording_id}")
    n_ref_frames = len(ref)
    n_sil_frames = np.sum(ref == 0)
    sil_prop = 100.* n_sil_frames / n_ref_frames
    n_overlap_frames = np.sum(ref == 1)
    overlap_prop = 100.* n_overlap_frames / n_ref_frames
    print(f"{n_ref_frames} TOTAL, {n_sil_frames} SILENCE({sil_prop:.0f}%), "
          f"{n_overlap_frames} OVERLAPPING({overlap_prop:.0f}%)")
    speaker_hist = np.bincount(ref)[2:]
    speaker_dist = speaker_hist/speaker_hist.sum()
    print(f"SPEAKER FREQUENCIES (DISCOUNTING OVERLAPS) "
          f"{np.array2string(speaker_dist, precision=2)}")
    print("")

    return ref


def write_rttm_file(rttm_path, labels, channel=0, step=0.01, precision=2):
    """Write RTTM file.

    Parameters
    ----------
    rttm_path : Path
        Path to output RTTM file.

    labels : ndarray, (n_frames,)
        Array of predicted speaker labels. See ``create_ref_file`` for explanation.

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

    # Paths to files in the data directory that we will be referring to:
    # - feats.scp  --  script file mapping recording ids to features spanning
    #   them
    # - utt2num_frames  --  mapping from recording ids to frame counts of
    #   corresponding features.
    utt2num_frames_filename = Path(args.data_dir, "utt2num_frames")
    feats_scp_filename = Path(args.data_dir, "feats.scp")

    # utt_list
    frame_counts = load_frame_counts(utt2num_frames_filename)
    recording_ids = sorted(frame_counts.keys())

    print("------------------------------------------------------------------------")
    print("")
    sys.stdout.flush()

    # Load the diagonal UBM and i-vector extractor.
    m, iE, w = load_dubm(args.dubm_model)
    V = load_ivector_extractor(args.ie_model)

    # Load the MFCC features
    feats_dict = {}

    for key,mat in kaldi_io.read_mat_scp(str(feats_scp_filename)):
        feats_dict[key] = mat

    for recording_id in recording_ids:
        # Get the alignments from the clustering result.
        # In init_ref, 0 denotes the silence silence frames
        # 1 denotes the overlapping speech frames, the speaker
        # label starts from 2.
        init_ref = create_ref_file(
            recording_id, frame_counts, args.init_rttm_filename, args.step)

        # Ground truth of the diarization.
        X = feats_dict[recording_id]
        X = X.astype(np.float64)

        # Keep only the voiced frames (0 denotes the silence
        # frames, 1 denotes the overlapping speech frames). Since
        # our method predicts single speaker label for each frame
        # the init_ref doesn't contain 1.
        mask = (init_ref >= 2)
        X_voiced = X[mask]
        init_ref_voiced = init_ref[mask] - 2

        if X_voiced.shape[0] == 0:
            print(
                f"Warning: {recording_id} has no voiced frames in the "
                f"initialization file")
            continue

        # Initialize the posterior of each speaker based on the clustering result.
        if args.initialize:
            # args.max_speakers=np.unique(init_ref_voiced)
            q = VB_diarization.frame_labels2posterior_mx(init_ref_voiced, args.max_speakers)
        else:
            q = None
            print("RANDOM INITIALIZATION\n")

        # VB resegmentation

        # q  - S x T matrix of posteriors attribution each frame to one of S
        #      possible speakers, where S is given by opts.maxSpeakers
        # sp - S dimensional column vector of ML learned speaker priors. Ideally,
        #      these should allow to estimate # of speaker in the recording as the
        #      probabilities of the redundant speaker should converge to zero.
        # Li - values of auxiliary function (and DER and frame cross-entropy
        #      between q and reference if 'ref' is provided) over iterations.
        q_out, sp_out, L_out = VB_diarization.VB_diarization(
            X_voiced, recording_id, m, iE, w, V, sp=None, q=q,
            maxSpeakers=args.max_speakers, maxIters=args.max_iters, VtiEV=None,
            downsample=args.downsample, alphaQInit=args.alphaQInit,
            sparsityThr=args.sparsityThr, epsilon=args.epsilon,
            minDur=args.minDur, loopProb=args.loopProb, statScale=args.statScale,
            llScale=args.llScale, ref=None, plot=False)

        predicted_label_voiced = np.argmax(q_out, 1) + 2
        predicted_label = (np.zeros(len(mask))).astype(int)
        predicted_label[mask] = predicted_label_voiced

        duration_list = []
        for i in range(args.max_speakers):
            num_frames = np.sum(predicted_label == (i + 2))
            if num_frames == 0:
                continue
            else:
                duration_list.append(1.0 * num_frames / len(predicted_label))
        duration_list.sort()
        duration_list = list(map(lambda x: '{0:.2f}'.format(x), duration_list))
        n_speakers = len(duration_list)
        dur_dist = " ".join(duration_list)
        print(f"PREDICTED {n_speakers} SPEAKERS")
        print(f"DISTRIBUTION {dur_dist}")
        print("sp_out", sp_out)
        print("L_out", L_out)

        # Create the output rttm file and compute the DER after re-segmentation.
        write_rttm_file(
            Path(args.output_dir, "per_file_rttm", recording_id + '.rttm'),
            predicted_label, channel=args.channel, step=args.step, precision=2)
        print("")
        print("------------------------------------------------------------------------")
        print("")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
