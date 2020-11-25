#!/bin/bash

# Wrapper for VB_resegmentation.py. Based on the "VB_resegmentation.sh" script
# from the Kaldi CALLHOME diarization recipe:
#
#     https://github.com/kaldi-asr/kaldi/blob/master/egs/callhome_diarization/v1/diarization/VB_resegmentation.sh
#
# Revision history
# ----------------
# - Zili Huang  --  original version
# - Neville Ryant  --  minor edits for formatting


################################################################################
# Configuration
################################################################################
# Stage to start from IN THE SCRIPT.
stage=0

# Number of parallel jobs.
nj=40

# Run command.
cmd=run.pl

# Python to use.
PYTHON=python

# Upper bound on number of speakers hypothesized.
max_speakers=10

# Maximum number of iterations to perform.
max_iters=10

# Downsample input by this factor before doing resegmentation.
downsample=25

# Concentration parameters for Dirichlet distribution used to initialize Q.
alphaQInit=100.0

# Set occupations below this threshold to 0.0. Since the algorithm uses sparse
# matrices, this will reduces memory footprint and speed up maxtrix operations.
sparsityThr=0.001

# Stop iterating if objective function threshold is < epsildon.
epsilon=1e-6

# Minimum number of frames allowed between speaker turns.
minDur=1

# Probability of NOT switching speakers between frames.
loopProb=0.9

# Scaling factor for sufficient statistics collected using UBM
statScale=0.2

# Scalign factor for UBM likelihoods.
llScale=1.0

# Channel number output in RTTM files.
channel=0

# If true, initialize speaker posteriors from the initial segmentation.
initialize=true


################################################################################
# Logging
################################################################################
echo "$0 $@"  # Print the command line for logging


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
  echo "Usage: local/VB_resegmentation.sh <data_dir> <init_rttm_filename> <output_dir> <dubm_model> <ie_model>"
  echo "Variational Bayes Re-segmenatation"
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # How to run jobs."
  echo "  --nj <num-jobs|20>                               # Number of parallel jobs to run."
  echo "  --true-rttm-filename <string|None>               # The true rttm label file"
  echo "  --max-speakers <n|10>                            # Maximum number of speakers"
  echo "                                                   # expected in the utterance"
  echo "                               # (default: 10)"
  echo "  --max-iters <n|10>                               # Maximum number of algorithm"
  echo "                                                   # iterations (default: 10)"
  echo "  --downsample <n|25>                              # Perform diarization on input"
  echo "                                                   # downsampled by this factor"
  echo "                                                   # (default: 25)"
  echo "  --alphaQInit <float|100.0>                       # Dirichlet concentraion"
  echo "                                                   # parameter for initializing q"
  echo "  --sparsityThr <float|0.001>                      # Set occupations smaller that"
  echo "                                                   # this threshold to 0.0 (saves"
  echo "                                                   # memory as the posteriors are"
  echo "                                                   # represented by sparse matrix)"
  echo "  --epsilon <float|1e-6>                           # Stop iterating, if obj. fun."
  echo "                                                   # improvement is less than"
  echo "                                   # epsilon"
  echo "  --minDur <n|1>                                   # Minimum number of frames"
  echo "                                                   # between speaker turns imposed"
  echo "                                                   # by linear chains of HMM"
  echo "                                                   # state corresponding to each"
  echo "                                                   # speaker. All the states in"
  echo "                                                   # a chain share the same output"
  echo "                                                   # distribution"
  echo "  --loopProb <float|0.9>                           # Probability of not switching"
  echo "                                                   # speakers between frames"
  echo "  --statScale <float|0.2>                          # Scale sufficient statistics"
  echo "                                                   # collected using UBM"
  echo "  --llScale <float|1.0>                            # Scale UBM likelihood (i.e."
  echo "                                                   # llScale < 1.0 make"
  echo "                                                   # attribution of frames to UBM"
  echo "                                                   # componets more uncertain)"
  echo "  --channel <n|0>                                  # Channel information in the rttm file"
  echo "  --initialize <true|false>                        # Whether to initalize the"
  echo "                                                   # speaker posterior (if not)"
  echo "                                                   # the speaker posterior will be"
  echo "                                                   # randomly initilized"

  exit 1;
fi

data_dir=$1
init_rttm_filename=$2
output_dir=$3
dubm_model=$4
ie_model=$5



################################################################################
# Call VB_resegmentation.py
################################################################################
sdata=$data_dir/split$nj;
utils/split_data.sh $data_dir $nj || exit 1;
JOB=1
if [ $stage -le 0 ]; then
  mkdir -p $output_dir/per_file_rttm
  args=""
  if [ $initialize == true ]; then
      args="--initialize"
  fi
  $cmd JOB=1:$nj $output_dir/log/VB_resegmentation.JOB.log \
    $PYTHON local/diarization/VB_resegmentation.py \
      $args \
      --max-speakers $max_speakers \
      --max-iters $max_iters --downsample $downsample --alphaQInit $alphaQInit \
      --sparsityThr $sparsityThr --epsilon $epsilon --minDur $minDur \
      --loopProb $loopProb --statScale $statScale --llScale $llScale \
      --channel $channel \
      $sdata/JOB $init_rttm_filename $output_dir $dubm_model $ie_model || exit 1;

fi
