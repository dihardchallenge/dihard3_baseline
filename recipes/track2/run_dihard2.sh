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
sad_stage=0
diarization_stage=0


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
