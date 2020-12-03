#!/usr/bin/env bash

set -e -u -o pipefail


################################################################################
# Configuration
################################################################################
# Stage to start from IN THE SCRIPT.
stage=0

# Number of parallel jobs to use during extraction/scoring/clustering.
nj=40

# Proportion of energy to retain when performing the conversation-dependent
# PCA projection. Usual default in Kaldi diarization recipes is 10%, but we use
# 30%, which was found to give better performance by Diez et al. (2020).
#
#   Diez, M. et. al. (2020). "Optimizing Bayesian HMM based x-vector clustering
#   for the Second DIHARD Speech Diarization Challenge." Proceedings of
#   ICASSP 2020.
target_energy=0.3

# AHC threshold.
thresh=-0.2

# If true, ignore "thresh" and instead tune the AHC threshold using the
# reference RTTM. The tuning stage repeats clustering for a range of thresholds
# and selects the one that  yields the lowest DER. Requires that the data
# directory contains a file named "rttm" with the reference diarization. If this
# file is absent, tuning will be skipped and the threshold will default to
# "thresh".
tune=false


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
if [ $# != 4 ]; then
  echo "usage: $0 <model-dir> <plda-path> <data-dir> <out-dir>"
  echo "e.g.: $0 exp/xvector_nnet_1a exp/xvector_nnet_1a/plda data/dev exp/diarization_nnet_1a_dev"
  echo "  --nj <n|40>         # number of jobs"
  echo "  --stage <stage|0>   # current stage; controls partial reruns"
  echo "  --thresh <t|-0.2>   # AHC threshold"
  echo "  --tune              # tune AHC threshold; data directory must contain reference RTTM"
  exit 1;
fi

# Directory containing the trained x-vector extractor.
model_dir=$1

# Path to the PLDA model to use in scoring.
plda_path=$2

# Data directory containing data to be diarized. If performing AHC tuning (i.e.,
# "--tune true"), must contain a file named "rttm" containing reference
# diarization.
data_dir=$3

# Output directory for x-vectors and diarization.
out_dir=$4


###############################################################################
# Extract 40-D MFCCs
###############################################################################
name=$(basename $data_dir)
if [ $stage -le 0 ]; then
  echo "$0: Extracting MFCCs...."
  set +e # We expect failures for short segments.
  steps/make_mfcc.sh \
    --nj $nj --cmd "$decode_cmd" --write-utt2num-frames true  \
    --mfcc-config conf/mfcc.conf \
    $data_dir  exp/make_mfcc/$name exp/make_mfcc/$name
  set -e
fi


###############################################################################
# Prepare feats for x-vector extractor by performing sliding window CMN.
###############################################################################
if [ $stage -le 1 ]; then
  echo "$0: Preparing features for x-vector extractor..."
  local/nnet3/xvector/prepare_feats.sh \
    --nj $nj --cmd "$decode_cmd" \
    data/$name data/${name}_cmn exp/make_mfcc/${name}_cmn/
  if [ -f data/$name/vad.scp ]; then
    echo "$0: vad.scp found; copying it"
    cp data/$name/vad.scp data/${name}_cmn/
  fi
  if [ -f data/$name/segments ]; then
    echo "$0: segments found; copying it"
    cp data/$name/segments data/${name}_cmn/
  fi
  utils/fix_data_dir.sh data/${name}_cmn
fi


###############################################################################
# Extract sliding-window x-vectors for all segments.
###############################################################################
if [ $stage -le 2 ]; then
  echo "$0: Extracting x-vectors..."
  local/diarization/nnet3/xvector/extract_xvectors.sh \
    --nj $nj --cmd "$decode_cmd" \
    --window 1.5 --period 0.25 --apply-cmn false \
    --min-segment 0.25 \
    $model_dir data/${name}_cmn $out_dir/xvectors
fi


###############################################################################
# Perform PLDA scoring for x-vectors.
###############################################################################
plda_dir=$out_dir/plda_scores
if [ $stage -le 3 ]; then
  echo "$0: Performing PLDA scoring..."

  # Use specified PLDA model + whitening computed from actual xvectors.
  plda_model_dir=$out_dir/plda
  mkdir $plda_model_dir
  cp $plda_path $plda_model_dir/plda
  cp $out_dir/xvectors/{mean.vec,transform.mat} $plda_model_dir
  local/diarization/nnet3/xvector/score_plda.sh \
    --nj $nj --cmd "$decode_cmd" \
    --target-energy $target_energy \
    $plda_model_dir $out_dir/xvectors $plda_dir
fi


###############################################################################
# Determine clustering threshold.
###############################################################################
tuning_dir=$out_dir/tuning
if [ $stage -le 4 ]; then
  mkdir -p $tuning_dir
  ref_rttm=$data_dir/rttm
  echo "$0: Determining AHC threshold..."
  if [[ $tune == true ]] && [[ -f $ref_rttm ]]; then
    echo "$0: Tuning threshold using reference diarization stored in: ${ref_rttm}"
    best_der=1000
    best_thresh=0
    for thresh in -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0; do
      echo "$0: Clustering with AHC threshold ${thresh}..."
      cluster_dir=$tuning_dir/plda_scores_t${thresh}
      mkdir -p $cluster_dir
      local/diarization/cluster.sh \
        --nj $nj --cmd "$decode_cmd" \
	--threshold $thresh --rttm-channel 1 \
	$plda_dir $cluster_dir
      perl local/diarization/md-eval.pl \
        -r $ref_rttm -s $cluster_dir/rttm \
        > $tuning_dir/der_t${thresh} \
	2> $tuning_dir/der_t${thresh}.log
      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
	      $tuning_dir/der_t${thresh})
      if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
          best_der=$der
          best_thresh=$thresh
      fi
    done
    echo "$best_thresh" > $tuning_dir/thresh_best
    echo "$0: ***** Results of tuning *****"
    echo "$0: *** Best threshold is: $best_thresh"
    echo "$0: *** DER using this threshold: $best_der"
  else
    echo "$thresh" > $tuning_dir/thresh_best
  fi
fi


###############################################################################
# Cluster using selected threshold
###############################################################################
if [ $stage -le 5 ]; then
  best_thresh=$(cat $tuning_dir/thresh_best)
  echo "$0: Performing AHC using threshold ${best_thresh}..."
  local/diarization/cluster.sh \
    --nj $nj --cmd "$decode_cmd" \
    --threshold $best_thresh --rttm-channel 1 \
    $plda_dir $out_dir
  local/diarization/split_rttm.py \
    $out_dir/rttm $out_dir/per_file_rttm
fi
