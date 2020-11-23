#!/usr/bin/env bash

set -e -u -o pipefail

################################################################################
# Configuration
################################################################################
# Use a no scoring collar of +/ "collar" seconds around each boundary.
collar=0.0

# Step size in seconds to use in computation of JER.
step=0.010

# If provided, output full scoring logs to this directory. It will contain the
# following files:
# - metrics_full.stdout  --  per-file and overall metrics for full test set; see
#   dscore documentation for explanation
# - metrics_full.stderr  --  warnings/errors produced by dscore for full test set
# - metrics_core.stdout  --  per-file and overall metrics for core test set
# - metrics_core.stderr  --  warnings/errors produced by dscore for full test set
scores_dir=


################################################################################
# Parse options, etc.
################################################################################
if [ -f path.sh ]; then
    . ./path.sh;
fi
if [ -f cmd.sh ]; then
    . ./cmd.sh;
fi
. utils/parse_options.sh || exit 1;
if [ $# != 2 ]; then
  echo "usage: $0 <release-dir> <rttm-dir>"
  echo "e.g.: $0 /data/corpora/LDC2020E12 exp/diarization_dev/rttms"
  exit 1;
fi

# Root of official LDC release; e.g., /data/corpora/LDC2020E12.
release_dir=$1

# Directory containing RTTMs to be scored.
rttm_dir=$2


################################################################################
# Score.
################################################################################
# Create temp directory for dscore outputs.
tmpdir=$(mktemp -d -t dh3-dscore-XXXXXXXX)

# Score FULL test set.
score.py \
  --collar $collar --step $step \
  -u $release_dir/data/uem_scoring/full/all.uem \
  -r $release_dir/data/rttm/*.rttm \
  -s $rttm_dir/*.rttm \
  >  $tmpdir/metrics_full.stdout \
  2> $tmpdir/metrics_full.stderr


# Score CORE test set.
score.py \
  -u $release_dir/data/uem_scoring/core/all.uem \
  -r $release_dir/data/rttm/*.rttm \
  -s $rttm_dir/*.rttm \
  >  $tmpdir/metrics_core.stdout \
  2> $tmpdir/metrics_core.stderr


# Report.
core_der=$(grep OVERALL $tmpdir/metrics_core.stdout | awk '{print $4}')
core_jer=$(grep OVERALL $tmpdir/metrics_core.stdout | awk '{print $5}')
full_der=$(grep OVERALL $tmpdir/metrics_full.stdout | awk '{print $4}')
full_jer=$(grep OVERALL $tmpdir/metrics_full.stdout | awk '{print $5}')
echo "$0: ******* SCORING RESULTS *******"
echo "$0: *** DER (full): ${full_der}"
echo "$0: *** JER (full): ${full_jer}"
echo "$0: *** DER (core): ${core_der}"
echo "$0: *** JER (core): ${core_jer}"
if [ ! -z "scores_dir" ]; then
 echo "$0: ***"
 echo "$0: *** Full results are located at: ${scores_dir}"
fi


# Clean up.
if [ ! -z "scores_dir" ]; then
  mkdir -p $scores_dir
  cp $tmpdir/* $scores_dir
fi
rm -fr $tmpdir
