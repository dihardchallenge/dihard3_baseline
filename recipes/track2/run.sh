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
eval_sad=false
train_sad=false

#####################################
# Paths to DIHARD III releases
#####################################
DIHARD_DEV_DIR=/home/prachis/Dihard_2020/LDC2020E12_Third_DIHARD_Challenge_Development_Data/
#DIHARD_EVAL_DIR=/data/working/nryant/dihard3/delivery/builds/LDC2020E13_Third_DIHARD_Challenge_Evaluation_Data


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
	--rttm-dir $DIHARD_DEV_DIR/data/rttm \
	data/dihard3_dev \
	$DIHARD_DEV_DIR/data/flac \
	$DIHARD_DEV_DIR/data/sad
    utils/utt2spk_to_spk2utt.pl \
	data/dihard3_dev/utt2spk > data/dihard3_dev/spk2utt
    ./utils/validate_data_dir.sh \
	--no-text --no-feats data/dihard3_dev/

    # eval
#    local/make_data_dir.py \
#        --rttm-dir $DIHARD_EVAL_DIR/data/rttm \
#        data/dihard3_eval \
#        $DIHARD_EVAL_DIR/data/flac \
#        $DIHARD_EVAL_DIR/data/sad
#    utils/utt2spk_to_spk2utt.pl \
#        data/dihard3_eval/utt2spk > data/dihard3_eval/spk2utt
#    ./utils/validate_data_dir.sh \
#        --no-text --no-feats data/dihard3_eval/
fi
#exit

#####################################
# Train SAD system.
#####################################
if [ $stage -le 1 -a $train_sad = "true" ]; then
    echo "$0: Training SAD..."
    local/train_sad.sh \
	--nj $nj --stage $sad_train_stage \
	data/dihard3_dev exp/dihard3_sad_tdnn_stats
fi
#exit

#####################################
# SAD decoding.
#####################################
if [ $stage -le 2 ]; then
    echo "$0: Applying SAD model to DEV/EVAL..."
    for dset in dev; do
	local/segmentation/detect_speech_activity.sh \
	    --nj $nj --stage $sad_decode_stage \
	    data/dihard3_${dset} exp/dihard3_sad_tdnn_stats \
	    mfcc exp/dihard3_sad_tdnn_stats_decode_${dset} \
       	    data/dihard3_${dset}_seg
    done
fi


#####################################
# Evaluate SAD output.
#####################################
if [ $stage -le 3  -a  $eval_sad = "true" ]; then
    echo "$0: Scoring SAD output on DEV set..."
    local/segmentation/score_sad.py \
	--n-jobs $nj --collar 0.0\
	data/dihard3_dev/segments \
	data/dihard3_dev_seg/segments \
	$DIHARD_DEV_DIR/docs/recordings.tbl
fi

mv data/dihard3_dev_seg data/dihard_dev_2020_track2 # Renaming data folder for consistency

echo "Diarizing..."

./alltracksrun.sh --tracknum 2 --DIHARD_DEV_DIR $DIHARD_DEV_DIR \
        --plda_path exp/xvector_nnet_1a/plda_track1 \
        --DSCORE_DIR $DSCORE_DIR --njobs $NJOBS
