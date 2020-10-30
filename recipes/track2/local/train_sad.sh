#!/usr/bin/env bash

# Copyright  2017  Nagendra Kumar Goel
#            2017  Vimal Manohar
#            2019  Desh Raj
# Apache 2.0

# This script is based on local/run_asr_segmentation.sh script in the
# Aspire recipe. It demonstrates nnet3-based speech activity detection for
# segmentation.
# This script:
# 1) Prepares targets (per-frame labels) for a subset of training data 
#    using GMM models
# 2) Trains TDNN+Stats or TDNN+LSTM neural network using the targets 
# 3) Demonstrates using the SAD system to get segments of dev data

data_dir=
                            # If not provided, a new one will be created using $lang_test


nj=50
reco_nj=40
nstage=-10
train_stage=-10
stage=0

. ./cmd.sh
. ./conf/sad.conf

if [ -f ./path.sh ]; then . ./path.sh; fi

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 0 ]; then
  exit 1
fi

dir=exp/segmentation${affix}
sad_work_dir=exp/sad${affix}_${nnet_type}/
sad_nnet_dir=$dir/tdnn_${nnet_type}_sad_1a

mkdir -p $dir
mkdir -p ${sad_work_dir}



# The training data may already be segmented, so we first prepare
# a "whole" training data (not segmented) for training the SAD
# system.
whole_data_dir=${data_dir}_whole
whole_data_id=$(basename $whole_data_dir)
echo $whole_data_dir

if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi


###############################################################################
# Extract features for the whole data directory. We extract 40-D MFCCs.
###############################################################################
if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --nj $reco_nj --cmd "$train_cmd"  --write-utt2num-frames true \
    --mfcc-config conf/mfcc_sad.conf \
    ${whole_data_dir} exp/make_mfcc/${whole_data_id}
  steps/compute_cmvn_stats.sh ${whole_data_dir} exp/make_mfcc/${whole_data_id}
  utils/fix_data_dir.sh ${whole_data_dir}
fi


###############################################################################
# Prepare SAD targets for recordings
#
# The output targets directory must contain the following file:
#
#     targets.scp
#
# which is a script file mapping recordings to targets for the subsampled frames.
# Three classes are used:
#
# - speech --> 0
# - nonspeech --> 1
# - garbage --> 2
#
# and are encoded using a one-hot encoding so that the resulting targets matrix for
# a recording REC-ID has dimensions NUM_SUBSAMPLED_FRAMES x 3.
#
# For more details, see:
#
#     https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/segmentation/lats_to_targets.sh
#
###############################################################################
targets_dir=$dir/${whole_data_id}_combined_targets
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
    if [ $nnet_type == "stats" ]; then
	# Train a STATS-pooling network for SAD
	local/segmentation/train_stats_sad_1a.sh \
	    --stage $nstage --train-stage $train_stage \
	    --targets-dir ${targets_dir} \
	    --data-dir ${whole_data_dir} --affix "1a" || exit 1
	
    elif [ $nnet_type == "lstm" ]; then
	# Train a TDNN+LSTM network for SAD
	local/segmentation/train_lstm_sad_1a.sh \
	    --stage $nstage --train-stage $train_stage \
	    --targets-dir ${targets_dir} \
	    --data-dir ${whole_data_dir} --affix "1a" || exit 1

  fi
fi

exit 0;
