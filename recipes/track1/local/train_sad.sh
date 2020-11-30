#!/usr/bin/env bash

set -e -u -o pipefail


##############################################################################
# Configuration                                                                ##############################################################################
nj=50


# Stage to start from IN THIS SCRIPT:
#
#   0:  convert data directory to "whole" data directory
#   1:  extract MFCCs for whole recordings
#   2:  prepare targets for SAD training
#   3:  train model
stage=0

# Stage to start neural network training prep from:
#
#   0:  generate neural net config
#   1:  train neural network
#   2:  estimate speech/non-speech priors using trained network
nstage=0  # Neural network training stage.

# The following controls at which stage we start neural network training:
#
#   -4:  initialize network
#   -3:  generate training examples
#   -2:  compute preconditioning matrix for input features
#   -1:  prepare initial network
#    0:  start training network
#   >0:  restart training from iteration <train_stage>
train_stage=-10  # Stage for nnet3/train_raw_rnn.py



##############################################################################
# Parse options, etc.
##############################################################################
. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh
if [ $# != 2 ]; then
    echo "usage: local/train_sad.sh <data-dir> <model-dir>"
    echo "e.g.: local/train_sad.sh data/dh_dev exp/sad_stats"
    echo "main options (for others, see top of script file)"
    echo "--nnet-type <net>  # neural network type (stats or lstm)"
    echo "--nj <nj>          # number of parallel jobs."
    echo "--stage <stage>    # stage to do partial re-run from."
    exit 1
fi

# Data directory containing training data. Must includes:
# - wav.scp
# - segments  --  speech segments
data_dir=$1

# Output directory for trained model.
dir=$2


##############################################################################
# Convert data directory to a "whole" data directory in which segments are
# removed and recording themselves are used as utterances.
###############################################################################
whole_data_dir=${data_dir}_whole
whole_data_id=$(basename $whole_data_dir)
if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi


###############################################################################
# Extract features for the whole data directory. We extract 40-D MFCCs.
###############################################################################
if [ $stage -le 1 ]; then
    steps/make_mfcc.sh \
	--nj $nj --cmd "$train_cmd"  \
	--mfcc-config conf/mfcc_sad.conf --write-utt2num-frames true \
	${whole_data_dir} exp/make_mfcc/${whole_data_id}
  steps/compute_cmvn_stats.sh ${whole_data_dir} exp/make_mfcc/${whole_data_id}
  utils/fix_data_dir.sh ${whole_data_dir}
fi


###############################################################################
# Prepare SAD targets for recordings
###############################################################################
mkdir -p $dir
targets_dir=exp/${whole_data_id}_sad_targets
if [ $stage -le 2 ]; then
  local/segmentation/prepare_sad_targets.py \
      --frame-step 0.010 \
      --subsampling-factor 3 \
      $data_dir/segments $whole_data_dir/utt2num_frames $targets_dir
fi


###############################################################################
# Train a neural network for SAD
###############################################################################
if [ $stage -le 3 ]; then
    # Train a STATS-pooling network for SAD
    local/segmentation/train_stats_sad_1a.sh \
	--stage $nstage --train-stage $train_stage \
	--num-epochs 40 \
	--targets-dir ${targets_dir} \
	--data-dir ${whole_data_dir} \
	--dir $dir
fi

exit 0;
