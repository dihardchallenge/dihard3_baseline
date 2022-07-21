export TOOLS_DIR=$PWD/../../tools
# dscore.
export PATH=$TOOLS_DIR/dscore:$PATH
# Kaldi.
#export KALDI_ROOT=$TOOLS_DIR/kaldi/
export KALDI_ROOT=/media/ssd3/xuechen/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

for f in steps utils; do ln -sf $KALDI_ROOT/egs/wsj/s5/$f .; done
ln -sf $KALDI_ROOT/egs/sre08/v1/sid .
