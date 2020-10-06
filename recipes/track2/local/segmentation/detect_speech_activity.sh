#!/usr/bin/env bash

# Copyright 2016-17  Vimal Manohar
#              2017  Nagendra Kumar Goel
# Apache 2.0.

# This script does nnet3-based speech activity detection given an input 
# kaldi data directory and outputs a segmented kaldi data directory.
# This script can also do music detection and other similar segmentation
# using appropriate options such as --output-name output-music.
#
# Adapted from:
#
#     wsj/s5/steps/segmentation/detect_speech_activity.sh
set -e 
set -o pipefail
set -u

if [ -f ./path.sh ]; then . ./path.sh; fi

nj=32
cmd=run.pl
stage=-1

# Feature options (Must match training)
mfcc_config=conf/mfcc_hires.conf

# SAD network config
iter=final  # Model iteration to use

# Contexts must ideally match training for LSTM models, but
# may not necessarily for stats components
extra_left_context=0  # Set to some large value, typically 40 for LSTM (must match training)
extra_right_context=0  
extra_left_context_initial=-1
extra_right_context_final=-1
frames_per_chunk=150

# Decoding options
graph_opts="--min-silence-duration=0.030 --min-speech-duration=0.240 --max-speech-duration=60.0"
acwt=0.3

# These <from>_in_<to>_weight represent the fraction of <from> probability 
# to transfer to <to> class.
# e.g. --speech-in-sil-weight=0.0 --garbage-in-sil-weight=0.0 --sil-in-speech-weight=0.0 --garbage-in-speech-weight=0.3
transform_probs_opts=""

# Postprocessing options
segment_padding=0.0   # Duration (in seconds) of padding added to segments 
min_segment_dur=0   # Minimum duration (in seconds) required for a segment to be included
                    # This is before any padding. Segments shorter than this duration will be removed.
                    # This is an alternative to --min-speech-duration above.
merge_consecutive_max_dur=0   # Merge consecutive segments as long as the merged segment is no longer than this many
                              # seconds. The segments are only merged if their boundaries are touching.
                              # This is after padding by --segment-padding seconds.
                              # 0 means do not merge. Use 'inf' to not limit the duration.

echo $* 

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "This script does nnet3-based speech activity detection given an input kaldi "
  echo "data directory and outputs an output kaldi data directory."
  echo "See script for details of the options to be supplied."
  echo "Usage: $0 <src-data-dir> <sad-nnet-dir> <mfcc-dir> <work-dir> <out-data-dir>"
  echo " e.g.: $0 ~/workspace/egs/ami/s5b/data/sdm1/dev exp/nnet3_sad_snr/nnet_tdnn_j_n4 \\"
  echo "    mfcc_hires exp/segmentation_sad_snr/nnet_tdnn_j_n4 data/ami_sdm1_dev"
  echo ""
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <num-job>                                 # number of parallel jobs to run."
  echo "  --stage <stage>                                # stage to do partial re-run from."
  echo "  --output-name <name>    # The output node in the network"
  echo "  --extra-left-context  <context|0>   # Set to some large value, typically 40 for LSTM (must match training)"
  echo "  --extra-right-context  <context|0>   # For BLSTM or statistics pooling"
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


whole_data_dir=data/${data_id}_whole_mfcc_hires
  if [ $stage -le 0 ]; then
    rm -fr ${whole_data_dir}
    utils/data/convert_data_dir_to_whole.sh $src_data_dir $whole_data_dir
  fi
  

###############################################################################
## Extract input features 
###############################################################################
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
if [ ! -f $sad_nnet_dir/post_output.vec ]; then
  if [ ! -f $sad_nnet_dir/post_output.txt ]; then
    echo "$0: Could not find $sad_nnet_dir/post_output.vec. "
    echo "Re-run the corresponding stage in the training script possibly "
    echo "with --compute-average-posteriors=true or compute the priors "
    echo "from the training labels"
    exit 1
  else
    post_vec=$sad_nnet_dir/post_output.txt
  fi
fi

mkdir -p $seg_dir
if [ $stage -le 4 ]; then
    echo "$0: Running Viterbi decoding..."
    steps/segmentation/internal/get_transform_probs_mat.py \
	--priors="$post_vec" \
	$transform_probs_opts > $seg_dir/transform_probs.mat
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
    cp $src_data_dir/{wav.scp,rttm,reco2num_spk} ${data_dir}
    utils/fix_data_dir.sh ${data_dir}
    echo "$0: Segmented kaldi data directory located at ${data_dir}."
fi

exit 0
