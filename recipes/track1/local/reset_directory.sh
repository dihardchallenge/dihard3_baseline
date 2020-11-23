#!/bin/bash
# Restore data/ and exp/ directories to virgin state.
rm -fr data
for dpath in $(ls -d exp/*/); do
    dname=$(basename $dpath)
    if [ $dname == "xvector_nnet_1a" ] || [ $dname == "xvec_init_gauss_1024_ivec_400" ]; then
	continue
    fi
    rm -fr $dpath
done

