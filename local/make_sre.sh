#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# See README.txt for more info on data required.

set -e

data_root=$1
data_dir=$2

wget -P data/local/ http://www.openslr.org/resources/15/speaker_list.tgz
tar -C data/local/ -xvf data/local/speaker_list.tgz
sre_ref=data/local/speaker_list

local/make_sre.pl $data_root/SRE04_TEST/test2/export/corpora5/LDC/LDC2006S44 \
   sre2004 $sre_ref $data_dir/sre2004
echo "SRE 2004 Done"
local/make_sre.pl $data_root/SRE05_TRAIN/LDC2011S01\
  sre2005 $sre_ref $data_dir/sre2005_train
echo "SRE 2005_1 Done"
local/make_sre.pl $data_root/SRE05_TEST/LDC2011S04 \
  sre2005 $sre_ref $data_dir/sre2005_test
echo "SRE 2005_2 Done"
local/make_sre.pl $data_root/SRE06_TRAIN/export/corpora5/LDC/LDC2011S09 \
  sre2006 $sre_ref $data_dir/sre2006_train
echo "SRE 2006 Done"
local/make_sre.pl $data_root/SRE08_TRAIN/LDC2011S05 \
  sre2008 $sre_ref $data_dir/sre2008_train
echo "SRE 2008_1 Done"

local/make_sre.pl $data_root/SRE08_TEST/export/corpora5/LDC/LDC2011S08 \
  sre2008 $sre_ref $data_dir/sre2008_test
echo "SRE 2008_2 Done"


  utils/combine_data.sh $data_dir/sre \
  $data_dir/sre2004 $data_dir/sre2005_train \
  $data_dir/sre2005_test $data_dir/sre2006_train \
  $data_dir/sre2008_train $data_dir/sre2008_test 

utils/validate_data_dir.sh --no-text --no-feats $data_dir/sre
utils/fix_data_dir.sh $data_dir/sre
rm data/local/speaker_list.*
