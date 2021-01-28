#!/bin/bash

set -e  # Exit on error.

PYTHON=python  # Python to use; defaults to system Python.


################################################################################
# Configuration
################################################################################
nj=40
stage=0
diarization_stage=0
vb_hmm_stage=0


################################################################################
# Paths to DIHARD III releases
################################################################################
DIHARD_DEV_DIR=/data/working/nryant/dihard3/delivery/builds/LDC2020E12_Third_DIHARD_Challenge_Development_Data
DIHARD_EVAL_DIR=/data/working/nryant/dihard3/delivery/builds/LDC2021E02_Third_DIHARD_Challenge_Evaluation_Data_Complete


. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh


################################################################################
# Prepare data directories
################################################################################
if [ $stage -le 0 ]; then
  echo "$0: Preparing data directories..."

  # dev
  local/make_data_dir.py \
  --rttm-dir $DIHARD_DEV_DIR/data/rttm \
    data/dihard3_dev \
    $DIHARD_DEV_DIR/data/flac \
    $DIHARD_DEV_DIR/data/sad
  utils/utt2spk_to_spk2utt.pl \
    data/dihard3_dev/utt2spk > data/dihard3_dev/spk2utt
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/dihard3_dev/

  # eval
  local/make_data_dir.py \
    --rttm-dir $DIHARD_EVAL_DIR/data/rttm \
    data/dihard3_eval \
    $DIHARD_EVAL_DIR/data/flac \
    $DIHARD_EVAL_DIR/data/sad
  utils/utt2spk_to_spk2utt.pl \
    data/dihard3_eval/utt2spk > data/dihard3_eval/spk2utt
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/dihard3_eval/
fi


################################################################################
# Perform first-pass diarization using AHC.
################################################################################
if [ $stage -le 1 ]; then
  echo "$0: Performing first-pass diarization of DEV..."
  local/diarize.sh \
    --nj $nj --stage $diarization_stage \
    --tune true \
    exp/xvector_nnet_1a/ exp/xvector_nnet_1a/plda_track1 \
    data/dihard3_dev/ exp/dihard3_diarization_nnet_1a_dev
fi


if [ $stage -le 2 ]; then
  echo "$0: Performing first-pass diarization of EVAL using threshold "
  echo "$0: obtained by tuning on DEV..."
  thresh=$(cat exp/dihard3_diarization_nnet_1a_dev/tuning/thresh_best)
  local/diarize.sh \
    --nj $nj --stage $diarization_stage \
    --thresh $thresh --tune false \
    exp/xvector_nnet_1a/ exp/xvector_nnet_1a/plda_track1 \
    data/dihard3_eval/ exp/dihard3_diarization_nnet_1a_eval
fi


################################################################################
# Evaluate first-pass diarization.
################################################################################
if [ $stage -le 3 ]; then
  echo "$0: Scoring first-pass diarization on DEV..."
  local/diarization/score_diarization.sh \
    --scores-dir exp/dihard3_diarization_nnet_1a_dev/scoring \
    $DIHARD_DEV_DIR exp/dihard3_diarization_nnet_1a_dev/per_file_rttm
fi


if [ $stage -le 4 ] && [ -d $DIHARD_EVAL_DIR/data/rttm ]; then
  echo "$0: Scoring first-pass diarization on EVAL..."
  local/diarization/score_diarization.sh \
    --scores-dir exp/dihard3_diarization_nnet_1a_eval/scoring \
    $DIHARD_EVAL_DIR exp/dihard3_diarization_nnet_1a_eval/per_file_rttm
fi



################################################################################
# Refined first-pass diarization using VB-HMM resegmentation
################################################################################
dubm_model=exp/xvec_init_gauss_1024_ivec_400/model/diag_ubm.pkl
ie_model=exp/xvec_init_gauss_1024_ivec_400/model/ie.pkl

if [ $stage -le 5 ]; then
  echo "$0: Performing VB-HMM resegmentation of DEV..."
  local/resegment_vbhmm.sh \
      --nj $nj --stage $vb_hmm_stage \
      data/dihard3_dev exp/dihard3_diarization_nnet_1a_dev/rttm \
      $dubm_model $ie_model exp/dihard3_diarization_nnet_1a_vbhmm_dev/
fi


if [ $stage -le 6 ]; then
  echo "$0: Performing VB-HMM resegmentation of EVAL..."
  local/resegment_vbhmm.sh \
      --nj $nj --stage $vb_hmm_stage \
      data/dihard3_eval exp/dihard3_diarization_nnet_1a_eval/rttm \
      $dubm_model $ie_model exp/dihard3_diarization_nnet_1a_vbhmm_eval/
fi



################################################################################
# Evaluate VB-HMM resegmentation.
################################################################################
if [ $stage -le 7 ]; then
  echo "$0: Scoring VB-HMM resegmentation on DEV..."
  local/diarization/score_diarization.sh \
    --scores-dir exp/dihard3_diarization_nnet_1a_vbhmm_dev/scoring \
    $DIHARD_DEV_DIR exp/dihard3_diarization_nnet_1a_vbhmm_dev/per_file_rttm
fi


if [ $stage -le 8 ] && [ -d $DIHARD_EVAL_DIR/data/rttm ]; then
  if [ -d $DIHARD_EVAL_DIR/data/rttm/ ]; then
    echo "$0: Scoring VB-HMM resegmentation on EVAL..."
    local/diarization/score_diarization.sh \
      --scores-dir exp/dihard3_diarization_nnet_1a_vbhmm_eval/scoring \
      $DIHARD_EVAL_DIR exp/dihard3_diarization_nnet_1a_vbhmm_eval/per_file_rttm
  fi
fi
