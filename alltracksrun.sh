. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/vad

tracknum=-1
DIHARD_DEV_DIR=default
DSCORE_DIR=dscore-master
nnet_dir=exp/xvector_nnet_1a
plda_path=default
njobs=40
stage=0

. parse_options.sh || exit 1;

if [ $# != 0 -o "$plda_path" = "default" -o "$tracknum" = "-1" ]; then
  echo "Usage: $0 --tracknum <1|2> --plda_path <path of plda file>"
  echo "main options (for others, see top of script file)"
  echo "  --tracknum <track number>         # number associated with the track to be run"
  echo "  --plda_path <plda-file>           # path of PLDA file"
  echo "  --njobs <n|40>                    # number of jobs"
  echo "  --stage <stage|0>                 # current stage; controls partial reruns"
  exit 1;
fi

if [[ !( "$tracknum" == "1" || "$tracknum" == "2" || "$tracknum" == "2_den" ||
         "$tracknum" == "3" || "$tracknum" == "4" || "$tracknum" == "4_den") ]]; then
    echo "ERROR: Unrecognized track."
    exit 1
fi

TRACK_DIR=recipes/track$tracknum
echo "Track: $TRACK_DIR"

echo "Running baseline for Track ${tracknum}..."
track=track$tracknum
dihard_dev=dihard_dev_2020_${track}
dihard_eval=dihard_eval_2020_${track}

# Determine max num jobs for each of DEV/EVAL.
dev_nfiles=`wc -l < data/${dihard_dev}/wav.scp`
dev_njobs=$((${njobs}<${dev_nfiles}?${njobs}:${dev_nfiles}))
#eval_nfiles=`wc -l < data/${dihard_eval}/wav.scp`
#eval_njobs=$((${njobs}<${eval_nfiles}?${njobs}:${eval_nfiles}))

# Extract MFCCs.
if [ $stage -le 0 ]; then
    echo "Extracting MFCCs...."
    for name in ${dihard_dev}; do #${dihard_eval}; do
	if [[ "$name" == "$dihard_dev" ]]; then
	    njobs=$dev_njobs
	else
	    njobs=$eval_njobs
	fi
	set +e # We expect failures for short segments.
	steps/make_mfcc.sh \
	    --cmd "$train_cmd --max-jobs-run 20" --nj $njobs \
	    --write-utt2num-frames true --mfcc-config conf/mfcc.conf \
	    data/${name} exp/make_mfcc/$name $mfccdir
	set -e
	utils/fix_data_dir.sh data/${name}
    done
    echo "MFCC extraction finished. See $PWD/exp/make_mfcc for logs."
fi

# Perform CMN.
if [ $stage -le 1 ]; then
    echo "Performing cepstral mean normalisation (CMN)..."
    for name in ${dihard_dev}; do # ${dihard_eval}; do
        if [[ "$name" == "$dihard_dev" ]]; then
            njobs=$dev_njobs
        else
            njobs=$eval_njobs
        fi
	local/nnet3/xvector/prepare_feats.sh \
	    --nj $njobs --cmd "$train_cmd" \
	    data/$name data/${name}_cmn exp/${name}_cmn
	if [ -f data/$name/vad.scp ]; then
	    echo "vad.scp found .. copying it"
	    cp data/$name/vad.scp data/${name}_cmn/
	fi
	if [ -f data/$name/segments ]; then
	    echo "Segments found .. copying it"
	    cp data/$name/segments data/${name}_cmn/
	fi
	utils/fix_data_dir.sh data/${name}_cmn
    done
    echo "CMN finished."
fi

# Extract x-vectors for DIHARD 2019 development and evaluation set.
DEV_XVEC_DIR=$nnet_dir/xvectors_${dihard_dev}
EVAL_XVEC_DIR=$nnet_dir/xvectors_${dihard_eval}
if [ $stage -le 2 ]; then
    echo "Extracting x-vectors for DEV..."
    cmn_dir=data/${dihard_dev}_cmn
    diarization/nnet3/xvector/extract_xvectors.sh \
	--cmd "$train_cmd --mem 5G" --nj $dev_njobs \
	--window 1.5 --period 0.25 --apply-cmn false \
	--min-segment 0.25 $nnet_dir \
	$cmn_dir $DEV_XVEC_DIR
    echo "X-vector extraction finished for DEV. See $DEV_XVEC_DIR/log for logs."
fi

if [ $stage -eq -7 ]; then
    echo "Extracting x-vectors for EVAL..."
    cmn_dir=data/${dihard_eval}_cmn
    diarization/nnet3/xvector/extract_xvectors.sh \
	--cmd "$train_cmd --mem 5G" --nj $eval_njobs \
	--window 1.5 --period 0.25 --apply-cmn false \
        --min-segment 0.25 $nnet_dir \
	$cmn_dir $EVAL_XVEC_DIR
    echo "X-vector extraction finished for EVAL. See $EVAL_XVEC_DIR/log for logs."
fi

#./train_plda.sh --tracknum 1

# Perform PLDA scoring
PLDA_DIR=$DEV_XVEC_DIR
DEV_SCORE_DIR=$DEV_XVEC_DIR/plda_scores
EVAL_SCORE_DIR=$EVAL_XVEC_DIR/plda_scores
if [ $stage -le 3 ]; then
    cp $plda_path $PLDA_DIR/plda

    echo "Performing PLDA scoring for DEV..."
    diarization/nnet3/xvector/score_plda.sh \
	    --cmd "$train_cmd --mem 4G" --nj $dev_njobs \
	    --target_energy 0.3 \
	    $PLDA_DIR $DEV_XVEC_DIR $DEV_SCORE_DIR
    echo "PLDA scoring finished for DEV. See $DEV_SCORE_DIR/log for logs."
fi
if [ $stage -eq -7 ]; then
    echo "Performing PLDA scoring for EVAL..."
    diarization/nnet3/xvector/score_plda.sh \
	--cmd "$train_cmd --mem 4G" --nj $eval_njobs \
	$PLDA_DIR $EVAL_XVEC_DIR $EVAL_SCORE_DIR
    echo "PLDA scoring finished for EVAL, See $EVAL_SCORE_DIR/log for logs."
fi

# Tune clustering threshold.
if [ $stage -le 4 ]; then
    mkdir -p $nnet_dir/tuning_$track
    echo "Tuning clustering threshold using DEV..."
    best_der=100
    best_threshold=0
    for threshold in -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0; do
	echo "Clustering with threshold $threshold..."
	cluster_dir=${DEV_XVEC_DIR}/plda_scores_t${threshold}
	diarization/cluster.sh \
	    --cmd "$train_cmd --mem 4G" --nj $dev_njobs \
	    --threshold $threshold --rttm-channel 1 \
	    $DEV_SCORE_DIR $cluster_dir
	perl md-eval.pl -r data/${dihard_dev}/rttm \
	     -s $cluster_dir/rttm \
	     2> $nnet_dir/tuning_$track/${dihard_dev}_t${threshold}.log \
	     > $nnet_dir/tuning_$track/${dihard_dev}_t${threshold}
	der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
		   $nnet_dir/tuning_$track/${dihard_dev}_t${threshold})
	if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
            best_der=$der
            best_threshold=$threshold
	fi
    done
    echo "Threshold tuning finished. See $DEV_XVEC_DIR/plda_scores_t*/log for logs."
    echo "*** Best threshold is: $best_threshold. PLDA scores of eval x-vectors will "
    echo "**  be clustered using this threshold"
    echo "*** DER on dev set using best threshold is: $best_der"
    echo "$best_threshold" > $nnet_dir/tuning_$track/${dihard_dev}_best
fi

# Cluster.
if [ $stage -le 5 ]; then
    best_threshold=$(cat $nnet_dir/tuning_$track/${dihard_dev}_best)
       
    echo "Performing agglomerative hierarchical clustering (AHC) using threshold $best_threshold for DEV..."
    diarization/cluster.sh \
	--cmd "$train_cmd --mem 4G" --nj $dev_njobs \
	--threshold $best_threshold --rttm-channel 1 \
	$DEV_SCORE_DIR $DEV_SCORE_DIR
    echo "Clustering finished for DEV. See $DEV_SCORE_DIR/log for logs."
fi

if [ $stage -eq -7 ]; then
    echo "Performing agglomerative hierarchical clustering (AHC) using threshold $best_threshold for EVAL..."
    diarization/cluster.sh \
	--cmd "$train_cmd --mem 4G" --nj $eval_njobs \
	--threshold $best_threshold --rttm-channel 1 \
	$EVAL_SCORE_DIR $EVAL_SCORE_DIR
    echo "Clustering finished for EVAL. See $EVAL_SCORE_DIR/log for logs."
fi

# Score Dev
#DEV_RTTM_DIR=${TRACK_DIR}/rttm_dev_${band}
DEV_RTTM_DIR=${DEV_SCORE_DIR}/filewise_rttms
EVAL_RTTM_DIR=${EVAL_SCORE_DIR}/filewise_rttms
if [ $stage -le 6 ]; then
    echo "Extracting RTTM files..."
    local/split_rttm.py \
        ${DEV_SCORE_DIR}/rttm ${DEV_RTTM_DIR}
    #local/split_rttm.py \
    #    ${EVAL_SCORE_DIR}/rttm ${EVAL_RTTM_DIR}

    # Score system outputs for DEV set against reference.
    echo "Scoring DEV set RTTM..."
    $PYTHON $DSCORE_DIR/score.py \
        -u $DIHARD_DEV_DIR/data/uem_scoring/full/all.uem \
        -r $DIHARD_DEV_DIR/data/rttm/*.rttm \
        -s $DEV_RTTM_DIR/*.rttm \
        > ${TRACK_DIR}/metrics_dev.stdout 2> ${TRACK_DIR}/metrics_dev.stderr
fi
#exit;
# VBHMM Resegmentation
if [ $stage -le 7 ]; then
    echo "Performing VB Resegmentation..."
    isvb=_vb

    init_rttm_path=${DEV_RTTM_DIR}
    cat $init_rttm_path/*.rttm > $init_rttm_path/rttm

    DEV_RTTM_DIR=${DEV_RTTM_DIR}${isvb}
    #EVAL_RTTM_DIR=${TRACK_DIR}/rttm_eval
    ./run_vb_dihard3.sh --stage 0 --data_dir ${dihard_dev} --nnet_dir $nnet_dir --init_rttm_path $init_rttm_path --output_dir_overall $DEV_RTTM_DIR

    # Score system outputs for DEV set against reference.
    echo "Scoring DEV set RTTM..."
    $PYTHON $DSCORE_DIR/score.py \
        -u $DIHARD_DEV_DIR/data/uem_scoring/full/all.uem \
        -r $DIHARD_DEV_DIR/data/rttm/*.rttm \
        -s $DEV_RTTM_DIR/*.rttm \
        > ${TRACK_DIR}/metrics_dev${isvb}.stdout 2> ${TRACK_DIR}/metrics_dev${isvb}.stderr

    grep OVERALL ${TRACK_DIR}/metrics_dev${isvb}.stdout
    echo "Filewise DER in ${TRACK_DIR}/metrics_dev${isvb}.stdout"

fi

echo "Run finished successfully."
