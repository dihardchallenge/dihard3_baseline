#!/bin/bash
# Installation script for Kaldi.
set -e


#######################
# Config
#######################
NJOBS=20  # Number of parallel jobs for make.


#######################
# Clone repo.
#######################
KALDI_GIT=https://github.com/kaldi-asr/kaldi
SCRIPT_DIR=$(realpath $(dirname "$0"))
KALDI_DIR=$SCRIPT_DIR/kaldi
KALDI_REVISION=76a97983a

if [ ! -d $KALDI_DIR ]; then
    git clone $KALDI_GIT $KALDI_DIR
    cd $KALDI_DIR
    git checkout $KALDI_REVISION
    cd ..
fi


#######################
# Build tools.
#######################
cd $KALDI_DIR/tools
if [ ! -f install.succeeded ]; then
    # Force Kaldi to use system Python.
    mkdir -p "python"
    touch "python/.use_default_python"

    # Compile.
    make -j $NJOBS

    # Compile OpenBLAS. For some reason the default install script doesn't put
    # components in the places expected by src/configure, so move some things
    # around.
    extras/install_openblas.sh
    mkdir OpenBLAS/lib
    mv OpenBLAS/*.so OpenBLAS/lib
    mkdir OpenBLAS/include
    cp OpenBLAS/*.h OpenBLAS/include

    touch install.succeeded

    # Clean up object files.
    echo $PWD
    find .  -type f -name "*.o" -exec rm {} \;
    find .  -type f -name "*.lo" -exec rm {} \;
fi


#######################
# Build Kaldi.
#######################
cd $KALDI_DIR/src
if [ ! -f install.succeeded ]; then
    # Configure.
    if [ -z "$CUDA_HOME" ]; then
	# Use default CUDA installlocation if CUDA_HOME not set.
	CUDA_HOME=/usr/local/cuda
    fi
    if [ -z "$MKL_ROOT" ]; then
	# Use default MKL install location if MKL_ROOT not set.
	MKL_ROOT=/opt/intel/mkl
    fi
    if [ -d "$MKL_ROOT" ]; then
	./configure \
            --shared --mathlib=MKL --mkl-root=$MKL_ROOT \
	    --use-cuda=yes --cudatk-dir=$CUDA_HOME
    else
	echo "Cannot find MKL library directory. Defaulting to OpenBLAS."
	echo "If you wish to use MKL:"
	echo "    - set MKL_ROOT to the directory where MKL was installed "
	echo "    - run \"rm kaldi/src/install.succeeded\""
	echo "    - rerun this script"
	./configure \
	    --shared \
	    --mathlib=OPENBLAS --openblas-root=../tools/OpenBLAS/install \
	    --use-cuda=yes --cudatk-dir=$CUDA_HOME
    fi

    # Build. May take a while.
    make clean
    make -j $NJOBS depend
    make -j $NJOBS

    # Clean up object files.
    find .  -type f -name "*.o" -exec rm {} \;

    touch install.succeeded
fi


echo "Successfully installed Kaldi."
