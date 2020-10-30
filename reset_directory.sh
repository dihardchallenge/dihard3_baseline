for name in mfcc exp/*decode* exp/xvector_nnet_1a/xvectors_dihard_*/ exp/xvector_nnet_1a/q exp/xvector_nnet_1a/log exp/xvector_nnet_1a/tuning_track* exp/xvec_init_gauss_1024_ivec_400/dihard* exp/dihard_* exp/make_mfcc ./*.out ./*.err recipes/track*/metrics* data/dihard*; do
	rm -r $name;
done;


