export KALDI_ROOT="/path/to/kaldi_directory/kaldi"
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH:$KALDI_ROOT/src/nnet3bin
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PATH="/state/partition1/softwares/kaldi/src/featbin:$PATH"
export PATH="/state/partition1/softwares/kaldi/tools/sph2pipe_v2.5:$PATH"
export PATH="/state/partition1/softwares/kaldi/src/bin:$PATH"
export PATH="/state/partition1/softwares/kaldi/src/ivectorbin:$PATH"
LMBIN=$KALDI_ROOT/tools/irstlm/bin
SRILM=$KALDI_ROOT/tools/srilm/bin/i686-m64
BEAMFORMIT=$KALDI_ROOT/tools/BeamformIt

export PATH=$PATH:$LMBIN:$BEAMFORMIT:$SRILM
