. ./cmd.sh
. ./path.sh

set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
data_root=/export/corpora5/LDC
sre_root=/export/corpora5/SRE
stage=0
nnet_dir=exp/xvector_nnet_1a/
#nnet_dir=/home/data1/prachis/Dihard_2019/v2/exp/xvector_nnet_1a/

data_dir=data/dihard_dev_2019_track1
dataset=dihard_2019_dev_full
whole_data_dir=data/$dataset
init_rttm_path=rttm_dev
#Variational Bayes resegmentation options
VB_resegmentation=true
num_gauss=1024
ivec_dim=400

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi

if [ $stage -le 1 ]; then
  # The script local/make_callhome.sh splits callhome into two parts, called
  # callhome1 and callhome2.  Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  for name in $dataset; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc_vb.conf --nj 15 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    cp $data_dir/rttm data/${name}/rttm
    utils/fix_data_dir.sh data/$name
  done
fi

if [ $VB_resegmentation ]; then
  # Variational Bayes method for smoothing the Speaker segments at frame-level
  output_dir=exp/xvec_init_gauss_${num_gauss}_ivec_${ivec_dim}

if [ $stage -eq 12 ]; then
  # Apply cmn and adding deltas will harm the performance on the callhome dataset. So we just use the 20-dim raw MFCC feature.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G --max-jobs-run 6" \
            --nj 10 --num-threads 4  --subsample 1 --delta-order 0 --apply-cmn false \
            data/swbd_sre_32k $num_gauss \
            exp/diag_ubm_gauss_${num_gauss}_delta_0_cmn_0
fi

if [ $stage -eq 13 ]; then
  # Train the i-vector extractor. The UBM is assumed to be diagonal.
  diarization/train_ivector_extractor_diag.sh --cmd "$train_cmd --mem 45G --max-jobs-run 20" \
                    --ivector-dim ${ivec_dim} \
                    --num-iters 5 \
                    --apply-cmn false \
                    --num-threads 1 --num-processes 1 --nj 10 \
                    exp/diag_ubm_gauss_${num_gauss}_delta_0_cmn_0/final.dubm data/swbd_sre \
                    exp/extractor_gauss_${num_gauss}_delta_0_cmn_0_ivec_${ivec_dim}
fi

if [ $stage -eq 14 ]; then
  # Convert the Kaldi UBM and T-matrix model to numpy array.
  mkdir -p $output_dir
  mkdir -p $output_dir/tmp
  mkdir -p $output_dir/log
  mkdir -p $output_dir/model

  # Dump the diagonal UBM model into text format.
  # "$train_cmd" $output_dir/log/convert_diag_ubm.log \
     gmm-global-copy --binary=false \
     exp/diag_ubm_gauss_${num_gauss}_delta_0_cmn_0/final.dubm \
     $output_dir/tmp/dubm.tmp || exit 1;

  # Dump the ivector extractor model into text format.
  # This method is not currently supported by Kaldi,
  # so please use my kaldi.
  # "$train_cmd" $output_dir/log/convert_ie.log \
     ivector-extractor-copy --binary=false \
     exp/extractor_gauss_${num_gauss}_delta_0_cmn_0_ivec_${ivec_dim}/final.ie \
     $output_dir/tmp/ie.tmp || exit 1;

  # diarization/dump_model.py $output_dir/tmp/dubm.tmp $output_dir/model
  # diarization/dump_model.py $output_dir/tmp/ie.tmp $output_dir/model

  /home/prachis/miniconda3/envs/py27/bin/python diarization/dump_model.py $output_dir/tmp/dubm.tmp $output_dir/model
  /home/prachis/miniconda3/envs/py27/bin/python diarization/dump_model.py $output_dir/tmp/ie.tmp $output_dir/model
fi


if [ $stage -le 15 ]; then

  echo "code"
  # /home/prachis/miniconda3/bin/python diarization/dump_model.py $output_dir/tmp/dubm.tmp $output_dir/model
  # /home/prachis/miniconda3/bin/python diarization/dump_model.py $output_dir/tmp/ie.tmp $output_dir/model
  output_data_dir=$output_dir/$dataset
  echo $output_data_dir

  mkdir -p $output_data_dir/results
  mkdir -p $output_data_dir/log
  
  
  init_rttm_file=$init_rttm_path/rttm
  label_rttm_file=data/$dataset/rttm
  # cat $nnet_dir/xvectors_dihard_2019_dev/plda_scores_energy0.3/rttm > $init_rttm_file

  # echo "code2"
  # # Compute the DER before VB resegmentation
  perl md-eval.pl -r $label_rttm_file -s $init_rttm_file 2> $output_data_dir/log/DER_init.log \
          > $output_data_dir/results/DER_init.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
         $output_data_dir/results/DER_init.txt)

  echo "der" $der

  stat=10.0
  loop=0.45
  
  echo "Stat:${stat}, Loop ${loop} : "
  predict=predict_stat${stat}_loop_${loop}.rttm
  # VB resegmentation. In this script, I use the x-vector result to
  # initialize the VB system. You can also use i-vector result or random
  # initize the VB system.
  diarization/VB_resegmentation.sh --nj 40 --cmd "$train_cmd" \
                       --initialize 1 --max_iters 1 --statScale $stat --loopProb $loop \
                 data/$dataset $init_rttm_file $output_data_dir $output_dir/model/diag_ubm.pkl $output_dir/model/ie.pkl || exit 1;

  # Compute the DER after VB resegmentation
  cat $output_data_dir/rttm/* > $output_data_dir/$predict
  md-eval.pl -1 -c 0.25 -r $label_rttm_file -s $output_data_dir/$predict 2> $output_data_dir/log/DER.log \
         > $output_data_dir/results/DER.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
         $output_data_dir/results/DER.txt)
  
  md-eval.pl -r $label_rttm_file -s $output_data_dir/predict_stat${stat}_loop_${loop}.rttm 2> /dev/null \
         > $output_data_dir/results/DER_overlap.txt
  # After VB resegmentation, DER: %
  echo "After VB resegmentation, DER: $der%"
  echo "Considering overlap"
  grep OVERALL $output_data_dir/results/DER_overlap.txt
  rm -r $output_data_dir/rttm  
fi # VB resegmentation part ends here.
fi
