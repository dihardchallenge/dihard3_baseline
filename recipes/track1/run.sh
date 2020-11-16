#!/bin/bash

set -e  # Exit on error.

PYTHON=python  # Python to use; defaults to system Python.


#####################################
# Configuration
#####################################
NJOBS=40
stage=0
diarization_stage=0


#####################################
# Paths to DIHARD III releases
#####################################
DIHARD_DEV_DIR=/data/working/nryant/dihard3/delivery/builds/LDC2020E12_Third_DIHARD_Challenge_Development_Data
DIHARD_EVAL_DIR=/data/working/nryant/dihard3/delivery/builds/LDC2020E13_Third_DIHARD_Challenge_Evaluation_Data


. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh


#####################################
# Prepare data directories
#####################################
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
exit 1


#####################################                                             # Perform diarization using
# pretrained model.
#####################################
if [ $stage -le 1]; then
    echo "$0: Diarizing DEV/EVAL..."
    ./alltracksrun.sh \
	--tracknum 1 --DIHARD_DEV_DIR $DIHARD_DEV_DIR \
	--plda_path exp/xvector_nnet_1a/plda_track1 \
	--DSCORE_DIR $DSCORE_DIR --njobs $NJOBS
