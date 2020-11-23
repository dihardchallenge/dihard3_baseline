#!/usr/bin/env python
import argparse
from pathlib import Path
import pickle
import sys

import kaldi_io
import numpy as np

import VB_diarization_v2 as VB_diarization



def get_utt_list(utt2spk_filename):
    utt_list = []
    with open(utt2spk_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt_list.append(line_split[0])
    print(f"{len(utt_list)} UTTERANCES IN TOTAL")
    return utt_list


def utt_num_frames_mapping(utt2num_frames_filename):
    utt2num_frames = {}
    with open(utt2num_frames_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2num_frames[line_split[0]] = int(line_split[1])
    return utt2num_frames


def create_ref_file(uttname, utt2num_frames, full_rttm_filename, temp_dir, rttm_filename):
    utt_rttm_fn = Path(temp_dir, rttm_filename)
    utt_rttm_file = open(utt_rttm_fn, 'w')

    num_frames = utt2num_frames[uttname]

    # We use 0 to denote silence frames and 1 to denote overlapping frames.
    ref = np.zeros(num_frames)
    speaker_dict = {}
    num_spk = 0

    with open(full_rttm_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname_line = line_split[1]
        if uttname != uttname_line:
            continue
        else:
            utt_rttm_file.write(line + "\n")
        start_time = int(float(line_split[3]) * 100)
        duration_time = int(float(line_split[4]) * 100)
        end_time = start_time + duration_time
        spkname = line_split[7]
        if spkname not in speaker_dict.keys():
            spk_idx = num_spk + 2
            speaker_dict[spkname] = spk_idx
            num_spk += 1

        for i in range(start_time, end_time):
            if i < 0:
                raise ValueError(line)
            elif i >= num_frames:
                print(f"{line} EXCEED NUM_FRAMES")
                break
            else:
                if ref[i] == 0:
                    ref[i] = speaker_dict[spkname]
                else:
                    ref[i] = 1 # The overlapping speech is marked as 1.
    ref = ref.astype(int)

    print(f"{num_spk} SPEAKERS IN {uttname}")
    n_ref_frames = len(ref)
    n_sil_frames = np.sum(ref == 0)
    sil_prop = 100.* n_sil_frames / n_ref_frames
    n_overlap_frames = np.sum(ref == 1)
    overlap_prop = 100.* n_overlap_frames / n_ref_frames
    print(f"{n_ref_frames} TOTAL, {n_sil_frames} SILENCE({sil_prop:.0f}%), "
          f"{n_overlap_frames} OVERLAPPING({overlap_prop:.0f}%)")

    duration_list = []
    for i in range(num_spk):
        duration_list.append(1.0 * np.sum(ref == (i + 2)) / len(ref))
    duration_list.sort()
    duration_list = map(lambda x: '{0:.2f}'.format(x), duration_list)
    dur_dist = " ".join(duration_list)
    print(f"DISTRIBUTION OF SPEAKER {dur_dist}")
    print("")
    sys.stdout.flush()
    utt_rttm_file.close()
    return ref


def create_rttm_output(uttname, predicted_label, output_dir, channel):
    num_frames = len(predicted_label)

    start_idx = 0
    idx_list = []

    last_label = predicted_label[0]
    for i in range(num_frames):
        if predicted_label[i] == last_label: # The speaker label remains the same.
            continue
        else: # The speaker label is different.
            if last_label != 0: # Ignore the silence.
                idx_list.append([start_idx, i, last_label])
            start_idx = i
            last_label = predicted_label[i]
    if last_label != 0:
        idx_list.append([start_idx, num_frames, last_label])

    rttmf = Path(output_dir, f"{uttname}_predict.rttm")
    with open(rttmf, 'w') as fh:
        for start_frame, end_frame, label in idx_list:
            onset = start_frame / 100.
            duration = (end_frame - start_frame) / 100.
            line = (f'SPEAKER {uttname} {channel} {onset:.2f} {duration:.2f} <NA> <NA> {label} <NA> <NA>\n')
            fh.write(line)
    return 0


def match_DER(string):
    string_split = string.split('\n')
    for line in string_split:
        if "OVERALL SPEAKER DIARIZATION ERROR" in line:
            return line
    return 0


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
        help='Set the maximum of speakers for an utterance (default: %(default)s)')
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
    
    # The data directory should contain wav.scp, spk2utt, utt2spk and feats.scp
    utt2spk_filename = Path(args.data_dir, "utt2spk")
    utt2num_frames_filename = Path(args.data_dir, "utt2num_frames")
    feats_scp_filename = Path(args.data_dir, "feats.scp")
    temp_dir = Path(args.output_dir, "tmp")
    rttm_dir = Path(args.output_dir, "rttm")

    utt_list = get_utt_list(utt2spk_filename)
    utt2num_frames = utt_num_frames_mapping(utt2num_frames_filename)
    print("------------------------------------------------------------------------")
    print("")
    sys.stdout.flush()

    # Load the diagonal UBM and i-vector extractor
    with open(args.dubm_model, 'rb') as fh:
        dubm_para = pickle.load(fh)
    with open(args.ie_model, 'rb') as fh:
        ie_para = pickle.load(fh)

    DUBM_WEIGHTS = None
    DUBM_MEANS_INVVARS = None
    DUBM_INV_VARS = None
    IE_M = None

    for key in dubm_para.keys():
        if key == "<WEIGHTS>":
            DUBM_WEIGHTS = dubm_para[key]
        elif key == "<MEANS_INVVARS>":
            DUBM_MEANS_INVVARS = dubm_para[key]
        elif key == "<INV_VARS>":
            DUBM_INV_VARS = dubm_para[key]
        else:
            continue

    for key in ie_para.keys():
        if key == "M":
            IE_M = np.transpose(ie_para[key], (2, 0, 1))
    m = DUBM_MEANS_INVVARS / DUBM_INV_VARS
    iE = DUBM_INV_VARS
    w = DUBM_WEIGHTS
    V = IE_M

    # Load the MFCC features
    feats_dict = {}

    for key,mat in kaldi_io.read_mat_scp(str(feats_scp_filename)):
        feats_dict[key] = mat

    for utt in utt_list:
        # Get the alignments from the clustering result.
        # In init_ref, 0 denotes the silence silence frames
        # 1 denotes the overlapping speech frames, the speaker
        # label starts from 2.
        init_ref = create_ref_file(
            utt, utt2num_frames, args.init_rttm_filename, temp_dir,
            f"{utt}.rttm")

        # Ground truth of the diarization.
        X = feats_dict[utt]
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
                f"Warning: {utt} has no voiced frames in the initialization file")
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
        #      these should allow to estimate # of speaker in the utterance as the
        #      probabilities of the redundant speaker should converge to zero.
        # Li - values of auxiliary function (and DER and frame cross-entropy
        #      between q and reference if 'ref' is provided) over iterations.
        q_out, sp_out, L_out = VB_diarization.VB_diarization(
            X_voiced,utt, m, iE, w, V, sp=None, q=q,
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

        # Create the output rttm file and compute the DER after re-segmentation
        create_rttm_output(utt, predicted_label, rttm_dir, args.channel)
        print("")
        print("------------------------------------------------------------------------")
        print("")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    main()
