INSTRUCTION=all # all or features or xvectors or VBx or score
SET=dev # dev or eval
USE_GMM=0
USE_INT=0
# DIHARD_DIR=$3 # directory containing the DIHARD data as provided by the organizers

# if [[ $SET = "dev" ]]; then
# 	DATA_DIR=$DIHARD_DIR/LDC2019E31_Second_DIHARD_Challenge_Development_Data
# elif [[ $SET = "eval" ]]; then
# 	DATA_DIR=$DIHARD_DIR/LDC2019E32_Second_DIHARD_Challenge_Evaluation_Data_V1.1
# else
# 	echo "The set has to be 'dev' or 'eval'"
# 	exit -1
# fi

#TMP_DIR=./tmp_dir_$SET
TMP_DIR=.
PLDA_DIR=./plda
OUT_DIR=./out_dir_${SET}
SCORE_DIR=dscore_master # directory with scoring tool: https://github.com/nryant/dscore
GT_DIR=data/dihard_2019_dev_track1
mkdir -p $OUT_DIR 

thr=0.0

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

echo $TMP_DIR
echo $PLDA_DIR
echo $OUT_DIR
echo $thr
echo $USE_GMM
echo $USE_INT
#if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "features" ]]; then
	# For recordings listed in file $1
	# - load lab file with VAD information from directory $2
	# - load original or WPE processed flac file from directory $3
	# - extract speech segment according to VAD
	# - each speech segment split into 1.5s subsegments with shift 0.25s
	# - extract Kaldi compatible cmv normalized fbank features for each subsegment
	# - save all subsegment feature matrices to ark file $4
	# - save subsegment timing information to $5
#	./compute_fbanks_cmn.py \
#	  list_$SET \
#	  $DATA_DIR/data/single_channel/sad \
#	  $DATA_DIR/data/single_channel/flac \
#	  $TMP_DIR/fbank_cmn.ark \
#	  $TMP_DIR/segments
#fi


# if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "xvectors" ]]; then
# 	feats_ark=$TMP_DIR/mfcc_feats_xvec12.ark
# 	model_init=xvector_extractor.txt
# 	feats_len=`wc -l $TMP_DIR/segments | awk '{print $1}'`
# 	arkfile=$TMP_DIR/xvectors.ark

# 	./extract.py --feats-ark $feats_ark \
#                 --feats-len $feats_len \
#                 --ark-file $arkfile \
#                 --batch-size 1 \
#                 --model-init $model_init
# fi


if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "VBx" ]]; then
	alpha=0.55
	# thr=0.0
	# echo $thr
	
	tareng=0.3
	smooth=5.0
	lda_dim=220
	Fa=0.4
	Fb=11
	loopP=0.80

	# x-vector clustering using VBHMM based diarization
	# for i in {1..40}; do
	# qsub -q all.q -V -cwd -e Log/vbhmm.${i}.err -o Log/vbhmm.${i}.out -S 
	python diarization_PLDAadapt_AHCxvec_BHMMxvec.py \
	 					$OUT_DIR \
	 					$TMP_DIR/xvectors.ark \
	 					$TMP_DIR/segments \
	 					$TMP_DIR/mean.vec \
	 					$TMP_DIR/transform.mat \
	 					$PLDA_DIR \
	 					$PLDA_DIR \
	 					$alpha \
	 					$thr \
	 					$tareng \
	 					$smooth \
	 					$lda_dim \
	 					$Fa \
	 					$Fb \
	 					$loopP \
	 					$USE_INT \
	 					$USE_GMM
	# done;
fi


if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "score" ]]; then
	python $SCORE_DIR/score.py \
		--collar 0.0 \
		-r $GT_DIR/rttm \
		-s $OUT_DIR/*.rttm \
		> $OUT_DIR/DER
#-u $DATA_DIR/data/single_channel/uem/all.uem \
		#-r $DATA_DIR/data/single_channel/rttm/*.rttm \
		#-s $OUT_DIR/*.rttm

fi
