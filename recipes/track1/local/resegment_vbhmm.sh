#!/usr/bin/env bash
set -e -u -o pipefail


################################################################################
# Configuration
################################################################################
# Stage to start from IN THE SCRIPT.
stage=0

# Number of parallel jobs to use during extraction/scoring/clustering.
nj=40

# Scaling factor for sufficient statistics collected usign UBM.
statscale=10

# Probability of NOT switching speakers between frames.
loop=0.45

# Maximum number of iterations.
max_iters=1


# Set occupations below this threshold to 0.0. Since the algorithm uses sparse
# matrices, this will reduces memory footprint and speed up maxtrix operations.
sparsityThr=0.001


################################################################################
# Parse options, etc.
################################################################################
if [ -f path.sh ]; then
    . ./path.sh;
fi
if [ -f cmd.sh ]; then
    . ./cmd.sh;
fi
. utils/parse_options.sh || exit 1;
if [ $# != 5 ]; then
  echo "usage: $0 <data-dir> <init-rttm> <dubm-model> <ie-model> <out-dir>"
  echo "e.g.: $0 data/dev/ exp/diarization_dev/rttm exp/ivec/diag_ubm.pkl exp/ivec/ie.pkl exp/diarization_vbhmm_dev/"
  echo "  --nj <n|40>         # number of jobs"
  echo "  --stage <stage|0>   # current stage; controls partial reruns"
  exit 1;
fi


# Data directory containing data to be diarized.
data_dir=$1

# Path to RTTM containing diarization to initialize from.
init_rttm=$2

# Path to diagonal UBM.
dubm_model=$3

# Path to i-vector extractor
ie_model=$4

# Output directory for diarization.
out_dir=$5


###############################################################################
# Convert to whole data directory.
#
# Since the data may already be segmented (e.g., from the first pass
# diarization), we first create a new data directory in which each recording is
# spanned by a single segment.
###############################################################################
whole_data_dir=${data_dir}_vbhmm_whole
if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi


###############################################################################
# Extract 24-D MFCCs
###############################################################################
name=$(basename $whole_data_dir)
if [ $stage -le 1 ]; then
  echo "$0: Extracting MFCCs...."
  steps/make_mfcc.sh \
    --nj $nj --cmd "$decode_cmd" --write-utt2num-frames true  \
    --mfcc-config conf/mfcc_vb.conf \
    $whole_data_dir  exp/make_mfcc/$name exp/make_mfcc/$name
    utils/fix_data_dir.sh ${whole_data_dir}
fi


###############################################################################
# Perform VB-HMM resegmentation
###############################################################################
if [ $stage -le 2 ]; then
  echo "$0: Performing VH-HMM resegmentation..."
  local/diarization/VB_resegmentation.sh \
    --nj $nj --cmd "$train_cmd" \
    --initialize true --max_iters $max_iters \
    --statScale $statscale --loopProb $loop \
    --channel 1 --sparsityThr $sparsityThr \
    $whole_data_dir $init_rttm $out_dir $dubm_model $ie_model
  cat $out_dir/per_file_rttm/*.rttm | sort > $out_dir/rttm
fi
