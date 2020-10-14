#!/bin/bash

set -e  # Exit on error.

PYTHON=python  # Python to use; defaults to system Python.

#####################################
# Configuration
#####################################
nj=40
decode_nj=40
PYTHON=python
stage=0
sad_train_stage=0
sad_decode_stage=0
diarization_stage=0

# If following is True, then SAD output will be evaluated against reference
# following decoding stage. This step requires the following Python packages be
# installed:
#
# - pyannote.core
# - pyannote.metrics
# - pandas
eval_sad=true


#####################################
# Paths to DIHARD II releases
#####################################
DIHARD_DEV_DIR=/data/working/nryant/dihard2/delivery/releases/LDC2019E31_Second_DIHARD_Challenge_Development_Data/
DIHARD_EVAL_DIR=/data/working/nryant/dihard2/delivery/releases/LDC2019E32_Second_DIHARD_Challenge_Evaluation_Data


. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh


#####################################
# Check dependencies
#####################################


#####################################
# Prepare data directories
#####################################
if [ $stage -le 0 ]; then
    echo "$0: Preparing data directories..."

    # dev
    local/make_data_dir.py \
	--rttm-dir $DIHARD_DEV_DIR/data/single_channel/rttm \
	data/dihard2_dev \
	$DIHARD_DEV_DIR/data/single_channel/flac \
	$DIHARD_DEV_DIR/data/single_channel/sad
    utils/utt2spk_to_spk2utt.pl \
	data/dihard2_dev/utt2spk > data/dihard2_dev/spk2utt
    ./utils/validate_data_dir.sh \
	--no-text --no-feats data/dihard2_dev/

    # eval
    local/make_data_dir.py \
        --rttm-dir $DIHARD_EVAL_DIR/data/single_channel/rttm \
        data/dihard2_eval \
        $DIHARD_EVAL_DIR/data/single_channel/flac \
        $DIHARD_EVAL_DIR/data/single_channel/sad
    utils/utt2spk_to_spk2utt.pl \
        data/dihard2_eval/utt2spk > data/dihard2_eval/spk2utt
    ./utils/validate_data_dir.sh \
        --no-text --no-feats data/dihard2_eval/
fi


#####################################
# Train SAD system.
#####################################
if [ $stage -le 1 ]; then
    echo "$0: Training SAD..."
    local/train_sad.sh \
	--nj $nj --stage $sad_train_stage \
	data/dihard2_dev exp/dihard2_sad_tdnn_stats
fi


#####################################
# SAD decoding.
#####################################
if [ $stage -le 2 ]; then
    echo "$0: Applying SAD model to DEV/EVAL..."
    for dset in dev eval; do
	local/segmentation/detect_speech_activity.sh \
	    --nj $nj --stage $sad_decode_stage \
	    data/dihard2_${dset} exp/dihard2_sad_tdnn_stats \
	    mfcc exp/sad_tdnn_stats_decode_${dset} \
       	    data/dihard2_${dset}_seg
    done
fi


#####################################
# Evaluate SAD output.
#####################################
if [ $stage -le 3  -a  $eval_sad = "true" ]; then
    echo "$0: Scoring SAD output on DEV set..."
    local/segmentation/score_sad.py \
	--n-jobs $nj --collar 0.0\
	data/dihard2_dev/segments \
	data/dihard2_dev_seg/segments \
	local/dihard2_dev_recordings.tbl
    echo ""
    echo ""

    echo "$0: Scoring SAD output on EVAL set..."
    local/segmentation/score_sad.py \
        --n-jobs $nj --collar 0.0\
        data/dihard2_eval/segments \
        data/dihard2_eval_seg/segments \
        local/dihard2_eval_recordings.tbl

fi
