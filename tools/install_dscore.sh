#!/bin/bash
# Installation script for dscore.

set -e


# Clone repo.
echo "Installing dscore."
DSCORE_GIT=https://github.com/nryant/dscore.git
SCRIPT_DIR=$(realpath $(dirname "$0"))
DSCORE_DIR=$SCRIPT_DIR/dscore
DSCORE_REVISION=824f126
if [ ! -d $DSCORE_DIR ]; then
    git clone $DSCORE_GIT $DSCORE_DIR
    cd $DSCORE_DIR
    git checkout $DSCORE_REVISION
    cd ..
else
    echo "$DSCORE_DIR already exists!"
fi


echo "Successfully installed dscore."
