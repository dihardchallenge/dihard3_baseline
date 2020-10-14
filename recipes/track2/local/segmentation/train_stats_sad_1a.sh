#!/usr/bin/env bash
# Copyright 2017   Nagendra Kumar Goel
#           2018   Vimal Manohar
# Apache 2.0

# This is a script to train a TDNN for speech activity detection (SAD)
# using statistics pooling for long-context information.
#
# Based on:
#
#     egs/chime6/s5_track2/local/segmentation/tuning/train_stats_sad_1a.sh



###############################################################################
# Global config
###############################################################################
stage=0
train_stage=-10
get_egs_stage=-10
nj=40


###############################################################################
# Context options
###############################################################################
chunk_width=20

# The context is chosen to be around 1 second long. The context at test time
# is expected to be around the same.
extra_left_context=79
extra_right_context=21


###############################################################################
# NNet config
###############################################################################
relu_dim=256


###############################################################################
# NNet training
###############################################################################
num_epochs=1
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=4
num_jobs_final=6
max_param_change=0.2  # Small max-param change for small network

# If following is true, remove intermediate models, configs/, egs/, and other
# files/directories created by training that are not needed to use the model.
cleanup=true


###############################################################################
# Directories
###############################################################################
data_dir=     # Data directory containing input features.
targets_dir=  # Directory containing labels for frames.
egs_dir=      # Working directory for examples during training.
dir=          # Output directory for trained model.


###############################################################################
# Path and other checks
###############################################################################
. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

set -e -o pipefail
set -u

if [ -z "$dir" ]; then
  dir=exp/segmentation_1a/tdnn_stats_sad
fi

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi



###############################################################################
# Generate model config.
###############################################################################
mkdir -p $dir

samples_per_iter=`perl -e "print int(400000 / $chunk_width)"`
#TODO: should be false for both
cmvn_opts="--norm-means=false --norm-vars=false"
echo $cmvn_opts > $dir/cmvn_opts

if [ $stage -le 0 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=`feat-to-dim scp:$data_dir/feats.scp -` name=input
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat

  relu-renorm-layer name=tdnn1 input=lda dim=$relu_dim add-log-stddev=true
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1,2) dim=$relu_dim add-log-stddev=true
  relu-renorm-layer name=tdnn3 input=Append(-3,0,3,6) dim=$relu_dim add-log-stddev=true
  stats-layer name=tdnn3_stats config=mean+count(-99:3:9:99)
  relu-renorm-layer name=tdnn4 input=Append(tdnn3@-6,tdnn3@0,tdnn3@6,tdnn3@12,tdnn3_stats) add-log-stddev=true dim=$relu_dim
  stats-layer name=tdnn4_stats config=mean+count(-108:6:18:108)
  relu-renorm-layer name=tdnn5 input=Append(tdnn4@-12,tdnn4@0,tdnn4@12,tdnn4@24,tdnn4_stats) dim=$relu_dim

  output-layer name=output include-log-softmax=true dim=3 learning-rate-factor=0.1 input=tdnn5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/

  cat <<EOF >> $dir/configs/vars
num_targets=3
EOF
fi



###############################################################################
# Train network.
###############################################################################
if [ $stage -le 1 ]; then
    # Set number of recordings to use for validation. Ideally, we'd set this to
    # 0 as this data isn't used by the optimization itself (e.g., for early
    # stopping, adjusting learning rate, etc), but 1 is the lowest we could do.
    #
    # NOTE: Be careful adusting this parameter or attempting to make inferences
    # about generalization from results computed on this subset. Because DIHARD
    # recording are very much not homogenous, this number needs to be relatively
    # high to give an accurate estimate of generalization to the EVAL set.
    num_utts_subset=1

    # Train. Apologies for so many command-line flags.
    steps/nnet3/train_raw_rnn.py \
	--stage=$train_stage \
	--cmd="$decode_cmd" --nj $nj \
	--use-gpu=true \
	--dir=$dir \
	--feat-dir=$data_dir --feat.cmvn-opts="$cmvn_opts" \
	--use-dense-targets=true \
	--targets-scp="$targets_dir/targets.scp" \
	--egs.chunk-width=$chunk_width \
	--egs.dir="$egs_dir" --egs.stage=$get_egs_stage \
	--egs.chunk-left-context=$extra_left_context \
	--egs.chunk-right-context=$extra_right_context \
	--egs.chunk-left-context-initial=0 \
	--egs.chunk-right-context-final=0 \
	--egs.opts="--frame-subsampling-factor 3 --num-utts-subset $num_utts_subset" \
	--trainer.num-epochs=$num_epochs \
	--trainer.samples-per-iter=20000 \
	--trainer.optimization.do-final-combination=false \
	--trainer.optimization.num-jobs-initial=$num_jobs_initial \
	--trainer.optimization.num-jobs-final=$num_jobs_final \
	--trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
	--trainer.optimization.final-effective-lrate=$final_effective_lrate \
	--trainer.rnn.num-chunk-per-minibatch=128,64 \
	--trainer.optimization.momentum=0.5 \
	--trainer.deriv-truncate-margin=10 \
	--trainer.max-param-change=$max_param_change \
	--trainer.compute-per-dim-accuracy=true \
	--cleanup=true \
	--cleanup.remove-egs=true \
	--cleanup.preserve-model-interval=10
fi



###############################################################################
# Estimate class posteriors for use in generating pseudo likelihoods
# at decoding time
###############################################################################
if [ $stage -le 2 ]; then
  # Set to actual distribution of speech/non-speech in DEV set:
  # - speech: 80%
  # - nonspeech: 20%  
  echo " [ 1 4 0.0001 ]" > $dir/post_output.vec
  echo 3 > $dir/frame_subsampling_factor
fi



###############################################################################
# Clean up training directory
###############################################################################
if [ $stage -le 3 -a $cleanup = "true" ]; then
    # Retains following:
    # - accuraccy.report
    # - cmvn_opts
    # - final.raw
    # - frame_subsampling_factor
    # - lda.mat
    # - lda_stats
    # - log/
    # - post_output.vec
    # - srand
    cp $dir/final.raw $dir/final.raw.keep
    rm $dir/*.raw
    mv $dir/final.raw.keep $dir/final.raw
    rm $dir/cache*
    rm -fr $dir/{configs,egs}
fi
