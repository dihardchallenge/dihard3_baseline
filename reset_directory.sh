for name in mfcc exp/xvector_nnet_1a/xvectors_dihard_*/ exp/xvector_nnet_1a/q exp/xvector_nnet_1a/log exp/xvector_nnet_1a/tuning_track1 exp/xvec_init_gauss_1024_ivec_400/dihard_* exp/dihard* exp/make_mfcc ./*.out ./*.err recipes/track1/metrics* data/dihard*; do
	rm -r $name;
done;


