#!/bin/bash
# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#  Apache 2.0.



# This script obtains phone posteriors from a trained chain model, using either
# the xent output or the forward-backward posteriors from the denominator fst.
# The phone posteriors will be in matrices where the column index can be
# interpreted as phone-index - 1.

# You may want to mess with the compression options.  Be careful: with the current
# settings, you might sometimes get exact zeros as the posterior values.

# CAUTION!  This script isn't very suitable for dumping features from recurrent
# architectures such as LSTMs, because it doesn't support setting the chunk size
# and left and right context.  (Those would have to be passed into nnet3-compute
# or nnet3-chain-compute-post).

# Begin configuration section.
stage=0

nj=1  # Number of jobs to run.
cmd=run.pl
remove_word_position_dependency=false
use_xent_output=false
online_ivector_dir=
use_gpu=false
count_smoothing=1.0  # this should be some small number, I don't think it's critical;
                     # it will mainly affect the probability we assign to phones that
                     # were never seen in training.  note: this is added to the raw
                     # transition-id occupation counts, so 1.0 means, add a single
                     # frame's count to each transition-id's counts.

# End configuration section.

set -e -u
echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <chain-model-dir> <data-dir> <out-dir> <layer-name>"
  echo " e.g.: $0 --online-ivector-dir exp/nnet3/ivectors_test_eval92_hires \\"
  echo "       exp/chain/tdnn1a_sp data/test_eval92_hires exp/chain/tdnn1a_sp_post_eval92 blstm3-forward.rp"
  echo " ... you'll normally want to set the --nj and --cmd options as well."
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (run.pl|queue.pl|... <queue opts>)    # how to run jobs."
  echo "  --config <config-file>                      # config containing options"
  echo "  --stage <stage>                             # stage to do partial re-run from."
  echo "  --nj <N>                                    # Number of parallel jobs to run, default:1"
  echo "                                              # (default: false, will use chain denominator FST)"
  echo "  --online-ivector-dir <dir>                  # Directory where we dumped online-computed"
  echo "                                              # ivectors corresponding to the data in <data>"
  echo "  --use-gpu <bool>                            # Set to true to use GPUs (not recommended as the"
  echo "                                              # binary is very poorly optimized for GPU use)."
  exit 1;
fi



model_dir=$1
data=$2
dir=$3
layer_name=$4


for f in $model_dir/final.mdl $model_dir/frame_subsampling_factor $model_dir/den.fst \
         $data/feats.scp; do
  [ ! -f $f ] && echo "train_sat.sh: no such file $f" && exit 1;
done

sdata=$data/split${nj}utt
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data $nj || exit 1;

use_ivector=false

cmvn_opts=$(cat $model_dir/cmvn_opts)
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

if [ ! -z "$online_ivector_dir" ];then
  steps/nnet2/check_ivectors_compatible.sh $model_dir $online_ivector_dir || exit 1;
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_feats="scp:utils/filter_scp.pl $sdata/JOB/utt2spk $online_ivector_dir/ivector_online.scp |"
  ivector_opts="--online-ivector-period=$ivector_period --online-ivectors='$ivector_feats'"
else
  ivector_opts=
fi

if $use_gpu; then
  gpu_queue_opt="--gpu 1"
  gpu_opt="--use-gpu=yes"
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
    exit 1
  fi
else
  gpu_queue_opts=
  gpu_opt="--use-gpu=no"
fi
frame_subsampling_factor=$(cat $model_dir/frame_subsampling_factor)

mkdir -p $dir/log
cp $model_dir/frame_subsampling_factor $dir/



if [ $stage -le 2 ]; then

  # note: --compression-method=3 is kTwoByteAuto: Each element is stored in two
  # bytes as a uint16, with the representable range of values chosen
  # automatically with the minimum and maximum elements of the matrix as its
  # edges.
  compress_opts="--compress=true --compression-method=3"

    model="nnet3-copy '--nnet-config=echo output-node name=output input=$layer_name|' --edits=remove-orphans '--edits-config=echo remove-output-nodes name=output-xent|' $model_dir/final.mdl -|"

    $cmd $gpu_queue_opts JOB=1:$nj $dir/log/get_bnfeats.JOB.log \
       nnet3-compute $gpu_opt $ivector_opts \
       --frame-subsampling-factor=$frame_subsampling_factor --apply-exp=true \
       "$model" "$feats" ark:- \| \
       copy-feats $compress_opts ark:- ark,scp:$dir/bnfeats.JOB.ark,$dir/bnfeats.JOB.scp

  sleep 5
  # Make a single .scp file, for convenience.
  for n in $(seq $nj); do cat $dir/bnfeats.$n.scp; done > $dir/bnfeats.scp

fi
