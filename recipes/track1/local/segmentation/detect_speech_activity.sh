#!/usr/bin/env bash

# Copyright 2016-17  Vimal Manohar
#              2017  Nagendra Kumar Goel
# Apache 2.0.

# This script does nnet3-based speech activity detection given an input
# kaldi data directory and outputs a segmented kaldi data directory.
#
# Adapted from:
#
#     wsj/s5/steps/segmentation/detect_speech_activity.sh
set -e
set -o pipefail
set -u

if [ -f ./path.sh ]; then . ./path.sh; fi

###############################################################################
# Global config
###############################################################################
stage=-1
cmd=run.pl
nj=40


###############################################################################
# Context config
###############################################################################
# TDNN already has such a wide context window, the values set here have next to
# no impact for performance.
extra_left_context=0
extra_right_context=0  
extra_left_context_initial=-1
extra_right_context_final=-1
frames_per_chunk=150


###############################################################################
# NNet config
###############################################################################
# Use final model.
iter=final


###############################################################################
# Decoding config
###############################################################################
# These parameters control the construction of the decoding graph and can be
# used to prevent Viterbi decoding from ever outputting speech or non-speech
# segments with durations below some threshold. However, these parameters have
# no impact on performance unless set to extreme values such as minimum
# non-speech duration of 1 second or minimum speech duration of 2 seconds.
min_sil_dur=0.030     # Minimum allowed duration in seconds of non-speech
                      # segments.
min_speech_dur=0.240  # Minimum allowed duration in seconds of speech segments.
max_speech_dur=60.0   # Maximum allowed duration in seconds of speech segments.

# Scaling factor for acoustic likelihoods.
acwt=0.3

# Scaling factor for speech likelihood. Values > 1 will increase false alarm rate
# and reduces miss rate. Values < 1 will increase miss rate and decrease false
# alarm rate.
speech_weight=1.0

graph_opts="--min-silence-duration=${min_sil_dur} --min-speech-duration=${min_speech_dur} --max-speech-duration=${max_speech_dur}"


###############################################################################
# Postprocessing config
###############################################################################
# All speech segments will be padded on either side by up to this many seconds.
# If extending the segment onset/offset by the full padding duration would result
# in the segment extending past the edge of the recording or overlapping an
# adjacent speech segment, then it will be truncated.
# If exte
segment_padding=0.0

# Minimum duration in seconds of speech segments. Any segments shorter than this
# threshold will be eliminated PRIOR to the padding step.
min_segment_dur=0

# Max duration in seconds of merged segments. Overlapping speech segments will be
# merged by an iterable process as long as the duration of the merged segment
# does not exceed this threshold. There are two special cases:
# - 0: perform no merging
# - inf: no limit on duration of merged segments
merge_consecutive_max_dur=inf


##############################################################################
# Parse options, etc.
##############################################################################
echo $*

set -e
set -o pipefail
set -u

if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "This script does nnet3-based speech activity detection given an input "
  echo "Kaldi data directory and outputs a new Kaldi data directory where the "
  echo "segments file contains the results of SAD."
  echo ""
  echo "Main options are documeted below; see script for details of the other "
  echo "options."
  echo "Usage: $0 <src-data-dir> <sad-nnet-dir> <mfcc-dir> <work-dir> <out-data-dir>"
  echo " e.g.: $0 ~/workspace/egs/ami/s5b/data/sdm1/dev exp/nnet3_sad_snr/nnet_tdnn_j_n4 \\"
  echo "    mfcc_sad exp/segmentation_sad_snr/nnet_tdnn_j_n4 data/ami_sdm1_dev"
  echo ""
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <num-job>                                 # number of parallel jobs to run."
  echo "  --stage <stage>                                # stage to do partial re-run from."
  echo "  --extra-left-context  <context|0>   # Set to some large value, typically 40 for LSTM (must match training)"
  echo "  --extra-right-context  <context|0>   # For BLSTM or statistics pooling"
  echo "--speech-weight <weight|1>  # Scaling factor for speech pseudo-likelihood"
  exit 1
fi

src_data_dir=$1   # The input data directory that needs to be segmented.
sad_nnet_dir=$2   # The SAD neural network
mfcc_dir=$3       # The directory to store the features
dir=$4            # Work directory
data_dir=$5       # The output data directory will be ${data_dir}_seg


data_id=`basename $src_data_dir`
posts_dir=${dir}/posts
seg_dir=${dir}/segmentation


whole_data_dir=data/${data_id}_whole_mfcc_sad
  if [ $stage -le 0 ]; then
    rm -fr ${whole_data_dir}
    utils/data/convert_data_dir_to_whole.sh $src_data_dir $whole_data_dir
  fi


###############################################################################
## Extract input features
###############################################################################  
mfcc_config=conf/mfcc_sad.conf
if [ $stage -le 1 ]; then
  utils/fix_data_dir.sh $whole_data_dir
  steps/make_mfcc.sh \
      --mfcc-config $mfcc_config --nj $nj --cmd "$cmd" \
      --write-utt2num-frames true \
      $whole_data_dir exp/make_mfcc/${data_id}_whole $mfcc_dir
  steps/compute_cmvn_stats.sh $whole_data_dir exp/make_mfcc/${data_id}_whole $mfcc_dir
  utils/fix_data_dir.sh ${whole_data_dir}
fi


###############################################################################
## Forward pass through the network network and dump the log-likelihoods.
###############################################################################
frame_subsampling_factor=1
if [ -f $sad_nnet_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $sad_nnet_dir/frame_subsampling_factor)
fi

mkdir -p $dir
if [ $stage -le 2 ]; then
    echo "$0: Computing non-speech/speech/garbage posteriors..."
    if [ "$(readlink -f $sad_nnet_dir)" != "$(readlink -f $dir)" ]; then
	cp $sad_nnet_dir/cmvn_opts $dir || exit 1
    fi
    cp $sad_nnet_dir/$iter.raw $dir/

    # Actual forward pass.
    steps/nnet3/compute_output.sh \
	--nj $nj --cmd "$cmd" --iter ${iter} \
	--extra-left-context $extra_left_context \
	--extra-right-context $extra_right_context \
	--extra-left-context-initial $extra_left_context_initial \
	--extra-right-context-final $extra_right_context_final \
	--frames-per-chunk $frames_per_chunk --apply-exp true \
	--frame-subsampling-factor $frame_subsampling_factor \
	${whole_data_dir} $dir $posts_dir || exit 1
fi


###############################################################################
## Prepare FST we search to make speech/silence decisions.
###############################################################################
utils/data/get_utt2dur.sh \
    --nj $nj --cmd "$cmd" $whole_data_dir || exit 1
frame_shift=$(utils/data/get_frame_shift.sh $whole_data_dir) || exit 1

graph_dir=${dir}/graph
if [ $stage -le 3 ]; then
    echo "$0: Preparing SAD decoding graph..."
    mkdir -p $graph_dir

    # 1 for silence and 2 for speech
    cat <<EOF > $graph_dir/words.txt
<eps> 0
silence 1
speech 2
EOF

    $cmd $graph_dir/log/make_graph.log \
        steps/segmentation/internal/prepare_sad_graph.py \
	    $graph_opts \
	    --frame-shift=$(perl -e "print $frame_shift * $frame_subsampling_factor") - \| \
	    fstcompile --isymbols=$graph_dir/words.txt --osymbols=$graph_dir/words.txt '>' \
	    $graph_dir/HCLG.fst
fi


###############################################################################
## Do Viterbi decoding to create per-frame alignments.
###############################################################################
post_vec=$sad_nnet_dir/post_output.vec
mkdir -p $seg_dir
if [ $stage -le 4 ]; then
    echo "$0: Running Viterbi decoding..."
    local/segmentation/get_transform_probs_mat.py \
        --speech-likelihood-weight $speech_weight  $post_vec \
	> $seg_dir/transform_probs.mat
    steps/segmentation/decode_sad.sh \
	--cmd "$cmd" --nj $nj \
	--acwt $acwt --transform "$seg_dir/transform_probs.mat" \
	$graph_dir $posts_dir $seg_dir
fi


###############################################################################
## Post-process segmentation to create kaldi data directory.
###############################################################################
if [ $stage -le 5 ]; then
    echo "$0: Post-processing Viterbi segmentation..."
    steps/segmentation/post_process_sad_to_segments.sh \
	--cmd "$cmd" --nj $nj \
	--segment-padding $segment_padding --min-segment-dur $min_segment_dur \
	--merge-consecutive-max-dur $merge_consecutive_max_dur \
	--frame-shift $(perl -e "print $frame_subsampling_factor * $frame_shift") \
	${whole_data_dir} ${seg_dir} ${seg_dir}
fi


if [ $stage -le 6 ]; then
    echo "$0: Generating new data directory from SAD..."
    utils/data/subsegment_data_dir.sh \
	${whole_data_dir} ${seg_dir}/segments ${data_dir}
    for bn in wav.scp rttm reco2num_spk; do
	if [ -f $src_data_dir/$bn ]; then
	    cp $src_data_dir/$bn ${data_dir}
	fi
    done
    utils/fix_data_dir.sh ${data_dir}
    echo "$0: Segmented kaldi data directory located at ${data_dir}."
fi

exit 0
