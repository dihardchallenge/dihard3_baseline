
. ./cmd.sh
. ./path.sh

set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
data_root=/export/corpora5/LDC
sre_root=/export/corpora5/SRE
stage=0
nnet_dir=exp/xvector_nnet_1a/
data_dir=dihard_dev_2020_track
init_rttm_path=rttm_dev
#Variational Bayes resegmentation options
VB_resegmentation=true
num_gauss=1024
ivec_dim=400
output_dir_overall=default
njobs_init=80
. utils/parse_options.sh
echo $data_dir;

dataset=${data_dir}_whole
whole_data_dir=data/$dataset

if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh data/$data_dir $whole_data_dir
fi

if [ $stage -le 1 ]; then
  # The script local/make_callhome.sh splits callhome into two parts, called
  # callhome1 and callhome2.  Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  for name in $dataset; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc_vb.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    cp data/${data_dir}/rttm data/${name}/rttm
    utils/fix_data_dir.sh data/$name
  done
fi

nfiles=`wc -l < $whole_data_dir/wav.scp`
njobs=$((${njobs_init}<${nfiles}?${njobs_init}:${nfiles}))
if [ $VB_resegmentation ]; then
  # Variational Bayes method for smoothing the Speaker segments at frame-level
  output_dir=exp/xvec_init_gauss_${num_gauss}_ivec_${ivec_dim}


    if [ $stage -le 2 ]; then

      echo "code"
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
      stat=10
      loop=0.45
      max_iters=1
      echo "Stat:${stat}, Loop ${loop} max_iters $max_iters: "
      predict=predict_stat${stat}_loop${loop}_max_iters${max_iters}.rttm
      DER_resullts=DER_stat${stat}_loop${loop}_max_iters${max_iters}
      # VB resegmentation. In this script, we use the x-vector result to
      # initialize the VB system. You can also use i-vector result or random
      # initize the VB system.

      diarization/VB_resegmentation.sh --nj $njobs --cmd "$train_cmd" \
                           --initialize 1 --max_iters $max_iters --statScale $stat --loopProb $loop --channel 1 \
                     $whole_data_dir $init_rttm_file $output_data_dir $output_dir/model/diag_ubm.pkl $output_dir/model/ie.pkl || exit 1;

      # Compute the DER after VB resegmentation
      cat $output_data_dir/rttm/* > $output_data_dir/$predict
      md-eval.pl -r $label_rttm_file -s $output_data_dir/$predict 2> $output_data_dir/log/DER.log \
             > $output_data_dir/results/${DER_resullts}.txt
      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
             $output_data_dir/results/${DER_resullts}.txt)
      
      # After VB resegmentation, DER: %
      echo "After VB resegmentation, DER: $der%"
      cp -r $output_data_dir/rttm/ $output_dir_overall 
      echo "filewise rttm in $output_dir_overall"
  fi
fi

