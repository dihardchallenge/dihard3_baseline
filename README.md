Implementation of diarization baseline for the [Third DIHARD Speech Diarization Challenge (DIHARD III)](https://dihardchallenge.github.io/dihard3/). The baseline is based on [LEAP Lab's](http://leap.ee.iisc.ac.in/) submission to [DIHARD II](https://dihardchallenge.github.io/dihard2/):

- Singh, Prachi, et. al. (2019). "LEAP Diarization System for Second DIHARD Challenge." Proceedings of INTERSPEECH 2019.
  ([paper](http://leap.ee.iisc.ac.in/navigation/publications/papers/DIHARD_2019_challenge_Prachi.pdf))


The x-vector extractor, PLDA model, UBM-GMM, and total variability matrix were trained on [VoxCeleb I and II](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/). Prior
to PLDA scoring and clustering, the x-vectors are centered and whitened using statistics estimated from the DIHARD III development and evaluation sets.

For further details about the training pipeline, please consult the companion paper to the challenge:

- Ryant, Neville et al. (2020). "The Third DIHARD Diarization Challenge."
  ([paper](http://arxiv.org/abs/2012.01477))




# Overview

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the baseline recipes](#running-the-baseline-recipes)
- [Expected results](#expected-results)
- [Reproducibility](#reproducibility)




# Prerequisites

The following packages are required to run the baseline.

- [Python](https://www.python.org/) >= 3.7
- [Kaldi](https://github.com/kaldi-asr/kaldi)
- [dscore](https://github.com/nryant/dscore)


Additionally, you will need to obtain the relevant data releases from [LDC](https://www.ldc.upenn.edu/):

- DIHARD III development set (**LDC2020E12**)

- DIHARD III evaluation set (**LDC2021E02**)


For instructions on obtaining these sets, please consult the [DIHARD III website](https://dihardchallenge.github.io/dihard3/).




# Installation

## Step 1: Clone the repo and create a new virtual environment

Clone the repo:

```bash
git clone https://github.com/dihardchallenge/dihard3_baseline.git
cd dihard3_baseline
```

While not required, we recommend running the recipes from a fresh virtual environment. If using ``virtualenv``:

```bash
virtualenv venv
source venv/bin/activate
```

Alternately, you could use ``conda`` or ``pipenv``. Make sure to activate the environment before proceeding.



## Step 2: Installing Python dependencies

Run the following command to install the required Python packages:

```bash
pip install -r requirements/core.txt
```

If you intend to enable [detailed scoring for SAD output](#sad-scoring) in the track 2 recipe, you will also need to install the ``pandas``, ``pyannote.core``, and ``pyannote.metrics`` packages:

```bash
pip install -r requirements/sad.txt
```


## Step 3: Installing remaining dependencies

We also need to install [Kaldi](https://github.com/kaldi-asr/kaldi) and [dscore](https://github.com/nryant/dscore). To do so, run the the installation scripts in the ``tools/`` directory:

```bash
cd tools
./install_kaldi.sh
./install_dscore.sh
cd ..
```

Please check the output of these scripts to ensure that installation has succeeded. If succesful, you should see ``Successfully installed {Kaldi,dscore}.`` printed at the end. If installation of a component fails, please consult the output of the relevant installation script for additional details.




# Running the baseline recipes

We include full recipes for reproducing the baseline results for both DIHARD III tracks:

- **Track 1**  —  Diarization using reference SAD.
- **Track 2**  —  Diarization from scratch (i.e., using system produced SAD).

These recipes are located under the ``recipes/`` directory, which has the following structure:

- ``track1/``  --  recipe for track 1
- ``track2/``  --  recipe for track 2 with SAD performed from original source audio


## Running the Track 1 recipe

To replicate the results for track 1, switch to ``recipes/track1/``:

```bash
cd recipes/track1
```

and open ``run.sh`` in a text editor. The first section of this script defines paths to the official DIHARD III DEV and EVAL releases and should look something like the following:

```bash
################################################################################
# Paths to DIHARD III releases
################################################################################
DIHARD_DEV_DIR=/data/working/nryant/dihard3/delivery/builds/LDC2020E12_Third_DIHARD_Challenge_Development_Data
DIHARD_EVAL_DIR=/data/working/nryant/dihard3/delivery/builds/LDC2021E02_Third_DIHARD_Challenge_Evaluation_Data_Complete
```

Change the variables ``DIHARD_DEV_DIR`` and ``DIHARD_EVAL_DIR`` so that they point to the roots of the DIHARD III DEV and EVAL releases on your filesystem. Save your changes, exit the editor, and run:

```bash
./run.sh
```

This will run the baseline, printing status updates to STDOUT with the first few lines of output resembling:

```bash
./run.sh: Preparing data directories...
./utils/validate_data_dir.sh: Successfully validated data-directory data/dihard3_dev/
./utils/validate_data_dir.sh: Successfully validated data-directory data/dihard3_eval/
./run.sh: Performing first-pass diarization of DEV...
local/diarize.sh: Extracting MFCCs....
steps/make_mfcc.sh --nj 40 --cmd run.pl --write-utt2num-frames true --mfcc-config conf/mfcc.conf data/dihard3_dev/ exp/make_mfcc/dihard3_dev exp/make_mfcc/dihard3_dev
utils/validate_data_dir.sh: Successfully validated data-directory data/dihard3_dev/
steps/make_mfcc.sh [info]: segments file exists: using that.
steps/make_mfcc.sh: It seems not all of the feature files were successfully procesed (39457 != 39743); consider using utils/fix_data_dir.sh data/dihard3_dev/
steps/make_mfcc.sh: Succeeded creating MFCC features for dihard3_dev
local/diarize.sh: Preparing features for x-vector extractor...
local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd run.pl data/dihard3_dev data/dihard3_dev_cmn exp/make_mfcc/dihard3_dev_cmn/
** split_data.sh: warning, #lines is (utt2spk,feats.scp) is (39743,39457); you can
**  use utils/fix_data_dir.sh data/dihard3_dev to fix this.
local/nnet3/xvector/prepare_feats.sh: Succeeded creating xvector features for dihard3_dev
local/diarize.sh: segments found; copying it
fix_data_dir.sh: kept 39457 utterances out of 39743
fix_data_dir.sh: old files are kept in data/dihard3_dev_cmn/.backup
local/diarize.sh: Extracting x-vectors..
```

If any stage fails, the script will immediately exit with status 1.


### RTTM files

Following the initial AHC segmentation step, RTTM files for the DEV set are written to:

    exp/dihard3_diarization_nnet_1a_dev/per_file_rttm

This directory will contain one RTTM for each recording in the DIHARD III DEV set; e.g.:

```bash
ls exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/*.rttm | head
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0001.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0002.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0003.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0004.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0005.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0006.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0007.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0008.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0009.rttm
exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/DH_DEV_0010.rttm

ls exp/dihard3_diarization_nnet_1a_dev/per_file_rttm/*.rttm | wc -l
254
```

Similarly, RTTM files for the EVAL set are written to:

    exp/dihard3_diarization_nnet_1a_eval/per_file_rttm

RTTM files will also be output after the VB-HMM resegmentation step, this time to:

- ``exp/dihard3_diarization_nnet_1a_vbhmm_dev/per_file_rttm``
- ``exp/dihard3_diarization_nnet_1a_vbhmm_eval/per_file_rttm``


### Scoring

DER and JER on the DEV set for the output of the AHC step will be printed to STDOUT:

```bash
./run.sh: Scoring first-pass diarization on DEV...
local/diarization/score_diarization.sh: ******* SCORING RESULTS *******
local/diarization/score_diarization.sh: *** DER (full): 20.71
local/diarization/score_diarization.sh: *** JER (full): 42.44
local/diarization/score_diarization.sh: *** DER (core): 21.05
local/diarization/score_diarization.sh: *** JER (core): 46.34
local/diarization/score_diarization.sh: ***
local/diarization/score_diarization.sh: *** Full results are located at: exp/dihard3_diarization_nnet_1a_dev/scoring
./run.sh: Scoring first-pass diarization on EVAL...
local/diarization/score_diarization.sh: ******* SCORING RESULTS *******
local/diarization/score_diarization.sh: *** DER (full): 20.75
local/diarization/score_diarization.sh: *** JER (full): 43.31
local/diarization/score_diarization.sh: *** DER (core): 21.66
local/diarization/score_diarization.sh: *** JER (core): 48.10
local/diarization/score_diarization.sh: ***
local/diarization/score_diarization.sh: *** Full results are located at: exp/dihard3_diarization_nnet_1a_eval/scoring
```

These scores are extracted from the original ``dscore`` logs, which are output to ``exp/dihard3_diarization_nnet_1a_dev/scoring``, which has the following structure:

- ``scoring/metrics_core.stdout``  --  output to STDOUT of ``dscore`` for CORE DEV set
- ``scoring/metrics_core.stderr``  --  output to STDERR of ``dscore`` for CORE DEV set
- ``scoring/metrics_full.stdout``  --  output to STDOUT of ``dscore`` for FULL DEV set
- ``scoring/metrics_full.stderr``  --  output to STDERR of ``dscore`` for FULL DEV set


Following VB-HMM resegmentation, DER and JER  for the DEV set are again printed to STDOUT:

```bash
./run.sh: Scoring VB-HMM resegmentation on DEV...
local/diarization/score_diarization.sh: ******* SCORING RESULTS *******
local/diarization/score_diarization.sh: *** DER (full): 19.41
local/diarization/score_diarization.sh: *** JER (full): 41.66
local/diarization/score_diarization.sh: *** DER (core): 20.25
local/diarization/score_diarization.sh: *** JER (core): 46.02
local/diarization/score_diarization.sh: ***
local/diarization/score_diarization.sh: *** Full results are located at: exp/dihard3_diarization_nnet_1a_vbhmm_dev/scoring
./run.sh: Scoring VB-HMM resegmentation on EVAL...
local/diarization/score_diarization.sh: ******* SCORING RESULTS *******
local/diarization/score_diarization.sh: *** DER (full): 19.25
local/diarization/score_diarization.sh: *** JER (full): 42.45
local/diarization/score_diarization.sh: *** DER (core): 20.65
local/diarization/score_diarization.sh: *** JER (core): 47.74
local/diarization/score_diarization.sh: ***
local/diarization/score_diarization.sh: *** Full results are located at: exp/dihard3_diarization_nnet_1a_vbhmm_eval/scoring
```

and detailed results written to ``exp/dihard3_diarization_nnet_1a_vbhmm_dev/scoring``.


### Running on a grid

**NOTE** that by default the Kaldi scripts are configured for execution on a single machine. To run on a grid using a submission engine such as SGE or Slurm, make sure to edit ``recipes/track{1,2}/cmd.sh`` accordingly.


## Running the Track 2 recipe

The track 2 recipe is almost identical to the track 1 recipe, except that it begins by training a SAD model on the DIHARD III DEV set, which it then uses to segment both the DEV and EVAL data. This SAD output will then be used by the diarization system in lieu of the reference SAD that was used for track 1.

To run the recipe, switch to ``recipes/track2/``:

```bash
cd recipes/track2
```

edit the variables ``DIHARD_DEV_DIR`` and ``DIHARD_EVAL_DIR`` just as for track 1, and run:

```bash
./run.sh
```

As before, status updates will periodically be printed to STDOUT and a failure at any stage will result in the script immediately exiting with status 1.


### RTTM files

As was the case for track 1, following the AHC step RTTM files will be output to:

- ``exp/dihard3_diarization_nnet_1a_dev/per_file_rttm``
- ``exp/dihard3_diarization_nnet_1a_eval/per_file_rttm``


and following the VB-HMM resegmentation step, RTTM files will be output to:

- ``exp/dihard3_diarization_nnet_1a_vbhmm_dev/per_file_rttm``
- ``exp/dihard3_diarization_nnet_1a_vbhmm_eval/per_file_rttm``


### Scoring

DER and JER on the DEV set will be printed to STDOUT after the AHC step:

```bash
./run.sh: Scoring first-pass diarization on DEV...
local/diarization/score_diarization.sh: ******* SCORING RESULTS *******
local/diarization/score_diarization.sh: *** DER (full): 24.08
local/diarization/score_diarization.sh: *** JER (full): 45.61
local/diarization/score_diarization.sh: *** DER (core): 24.06
local/diarization/score_diarization.sh: *** JER (core): 49.17
local/diarization/score_diarization.sh: ***
local/diarization/score_diarization.sh: *** Full results are located at: exp/dihard3_diarization_nnet_1a_dev/scoring
./run.sh: Scoring first-pass diarization on EVAL...
local/diarization/score_diarization.sh: ******* SCORING RESULTS *******
local/diarization/score_diarization.sh: *** DER (full): 28.00
local/diarization/score_diarization.sh: *** JER (full): 49.35
local/diarization/score_diarization.sh: *** DER (core): 29.51
local/diarization/score_diarization.sh: *** JER (core): 53.82
local/diarization/score_diarization.sh: ***
local/diarization/score_diarization.sh: *** Full results are located at: exp/dihard3_diarization_nnet_1a_eval/scoring
```

and again following the VB-HMM resegmetation step:


```bash
./run.sh: Scoring VB-HMM resegmentation on DEV...
local/diarization/score_diarization.sh: ******* SCORING RESULTS *******
local/diarization/score_diarization.sh: *** DER (full): 21.71
local/diarization/score_diarization.sh: *** JER (full): 43.66
local/diarization/score_diarization.sh: *** DER (core): 22.28
local/diarization/score_diarization.sh: *** JER (core): 47.75
local/diarization/score_diarization.sh: ***
local/diarization/score_diarization.sh: *** Full results are located at: exp/dihard3_diarization_nnet_1a_vbhmm_dev/scoring
local/diarization/score_diarization.sh: ******* SCORING RESULTS *******
local/diarization/score_diarization.sh: *** DER (full): 25.36
local/diarization/score_diarization.sh: *** JER (full): 46.95
local/diarization/score_diarization.sh: *** DER (core): 27.34
local/diarization/score_diarization.sh: *** JER (core): 51.91
local/diarization/score_diarization.sh: ***
local/diarization/score_diarization.sh: *** Full results are located at: exp/dihard3_diarization_nnet_1a_vbhmm_eval/scoring
```

As was the case for the track 1 recipe, the original ``dscore`` logs will be output to:

- ``exp/dihard3_diarization_nnet_1a_dev/scoring``
- ``exp/dihard3_diarization_nnet_1a_vbhmm_dev/scoring``.


### SAD scoring

Optionally, the script will score the SAD output for the DEV set against the reference SAD and compute the following metrics overall and on a per-domain basis:

- [detection cost function (DCF)](https://www.nist.gov/system/files/documents/2018/11/05/opensat19_evaluation_plan_v2_11-5-18.pdf)
- [detection error rate (DER)](http://herve.niderb.fr/download/pdfs/Bredin2017a.pdf)
- accuracy
- false alarm rate
- miss rate

These metrics will be output to STDOUT as a table with the following 10 columns:

- domain  --  recording domain; overall results are reported under the domain ``OVERALL``
- dcf  --  detection cost function in percent
- der  --  detection error rate in percent
- accuracy  --  accuracy in percent
- fa  --  false alarm rate in percent
- miss  --  miss rate in percent
- fa duration  --  the total duration in seconds of all false alarms
- miss duration  --  the total duration in seconds of all misses
- speech duration  --  the total duration in seconds of speech according to
  the reference segmentation
- nonspeech duration  --  the total duration in seconds of non-speech
  according to the reference segmentation

For example:

```
bash
./run.sh: Scoring SAD output on CORE DEV set...
domain                 dcf    der    accuracy     fa    miss    fa duration    miss duration    speech duration    nonspeech duration
-------------------  -----  -----  ----------  -----  ------  -------------  ---------------  -----------------  --------------------
audiobooks            1.13   1.38       98.90   1.64    0.95          24.56            54.79            5743.97               1499.00
broadcast_interview   1.65   2.05       98.39   2.44    1.39          38.68            81.05            5832.72               1583.28
clinical              3.21   4.98       96.99   2.17    3.56          63.02           157.76            4432.29               2900.77
court                 1.71   1.84       98.46   3.10    1.25          37.11            78.56            6298.96               1196.41
cts                   3.67   3.01       97.32   9.04    1.87          78.76           129.89            6928.48                870.90
maptask               2.19   3.32       97.75   2.85    1.96          83.46           121.07            6167.54               2924.84
meeting               6.07   2.42       97.74  21.42    0.95         120.53            78.69            8249.17                562.65
restaurant            3.27   2.53       97.77   9.34    1.25          82.51            79.91            6409.74                883.52
socio_field           2.73   3.77       97.26   2.83    2.70          56.40           141.74            5253.77               1989.93
socio_lab             2.84   3.84       97.15   3.82    2.52          94.12           179.53            7134.36               2464.94
webvideo              2.90   3.88       97.10   3.56    2.69          60.78           136.46            5080.94               1709.45
OVERALL               2.37   2.93       97.70   3.98    1.84         739.96          1239.45           67531.94              18585.67


./run.sh: Scoring SAD output on FULL DEV set...
domain                 dcf    der    accuracy     fa    miss    fa duration    miss duration    speech duration    nonspeech duration
-------------------  -----  -----  ----------  -----  ------  -------------  ---------------  -----------------  --------------------
audiobooks            1.13   1.38       98.90   1.64    0.95          24.56            54.79            5743.97               1499.00
broadcast_interview   1.65   2.05       98.39   2.44    1.39          38.68            81.05            5832.72               1583.28
clinical              3.26   5.06       96.98   2.09    3.65         128.07           332.43            9099.55               6124.68
court                 1.71   1.84       98.46   3.10    1.25          37.11            78.56            6298.96               1196.41
cts                   3.90   2.95       97.36  10.43    1.72         402.84           562.68           32733.15               3863.92
maptask               2.19   3.32       97.75   2.85    1.96          83.46           121.07            6167.54               2924.84
meeting               6.07   2.42       97.74  21.42    0.95         120.53            78.69            8249.17                562.65
restaurant            3.27   2.53       97.77   9.34    1.25          82.51            79.91            6409.74                883.52
socio_field           2.73   3.77       97.26   2.83    2.70          56.40           141.74            5253.77               1989.93
socio_lab             2.84   3.84       97.15   3.82    2.52          94.12           179.53            7134.36               2464.94
webvideo              2.90   3.88       97.10   3.56    2.69          60.78           136.46            5080.94               1709.45
OVERALL               2.55   3.04       97.58   4.55    1.88        1129.09          1846.90           98003.88              24802.61


./run.sh: Scoring SAD output on CORE EVAL set...
domain                 dcf    der    accuracy     fa    miss    fa duration    miss duration    speech duration    nonspeech duration
-------------------  -----  -----  ----------  -----  ------  -------------  ---------------  -----------------  --------------------
audiobooks            2.28   2.88       97.77   3.69    1.81          60.69           103.30            5702.96               1644.45
broadcast_interview   4.53   5.64       95.63   9.60    2.84         158.27           160.56            5654.08               1648.38
clinical              6.81  11.34       92.91   8.47    6.25         235.15           289.17            4624.34               2775.04
court                 3.84   4.02       96.66   8.41    2.32         104.27           141.51            6109.74               1239.62
cts                   5.96   4.47       96.03  16.82    2.33         148.04           161.44            6919.28                880.10
maptask               3.08   5.29       96.60   5.40    2.30         143.53           110.79            4810.31               2658.97
meeting              10.01  10.36       91.48  25.64    4.80         307.61           265.70            5532.99               1199.51
restaurant           19.00  17.14       84.90  41.35   11.55         363.93           752.86            6517.08                880.21
socio_field          10.77  13.18       89.78  26.92    5.39         492.11           340.96            6322.04               1828.22
socio_lab             5.40   7.01       94.67  10.92    3.56         191.64           198.07            5557.63               1755.27
webvideo             15.80  22.04       83.87  29.21   11.33         585.27           619.27            5464.26               2003.76
OVERALL               7.50   9.39       92.74  15.07    4.97        2790.50          3143.63           63214.72              18513.52


./run.sh: Scoring SAD output on FULL EVAL set...
domain                 dcf    der    accuracy     fa    miss    fa duration    miss duration    speech duration    nonspeech duration
-------------------  -----  -----  ----------  -----  ------  -------------  ---------------  -----------------  --------------------
audiobooks            2.28   2.88       97.77   3.69    1.81          60.69           103.30            5702.96               1644.45
broadcast_interview   4.53   5.64       95.63   9.60    2.84         158.27           160.56            5654.08               1648.38
clinical              6.96  11.86       92.68   9.01    6.28         535.63           602.36            9596.82               5945.99
court                 3.84   4.02       96.66   8.41    2.32         104.27           141.51            6109.74               1239.62
cts                   6.14   4.54       95.95  17.29    2.43         690.99           790.68           32600.88               3996.22
maptask               3.08   5.29       96.60   5.40    2.30         143.53           110.79            4810.31               2658.97
meeting              10.01  10.36       91.48  25.64    4.80         307.61           265.70            5532.99               1199.51
restaurant           19.00  17.14       84.90  41.35   11.55         363.93           752.86            6517.08                880.21
socio_field          10.77  13.18       89.78  26.92    5.39         492.11           340.96            6322.04               1828.22
socio_lab             5.40   7.01       94.67  10.92    3.56         191.64           198.07            5557.63               1755.27
webvideo             15.80  22.04       83.87  29.21   11.33         585.27           619.27            5464.26               2003.76
OVERALL               6.93   8.22       93.49  14.65    4.35        3633.93          4086.06           93868.79              24800.59
```

This behavior is enabled by setting ``eval_sad=true`` in the preamble of ``run.sh``. If you elect to do so, make sure you also install the ``pandas``, ``pyannote.core``, and ``pyannote.metrics`` packages [as described above](#installation).




# Expected results

Expected DER and JER for the baseline system on the DIHARD III DEV and EVAL sets are presented in Tables 1 and 2.


**Table 1: Baseline diarization results for the DIHARD III development and evaluation set using just AHC.**

| Track   | Partition   | DER (dev)   | DER (eval)   | JER (dev)   | JER (eval)   |
| ------- | ----------- | ----------- | ------------ | ----------- | ------------ |
| Track 1 | core        | 21.05       | 21.66        | 46.34       | 48.10        |
| Track 1 | full        | 20.71       | 20.75        | 42.44       | 43.31        |
| Track 2 | core        | 24.06       | 29.51        | 49.17       | 53.82        |
| Track 2 | full        | 24.08       | 28.00        | 45.61       | 49.35        |


**Table 2: Baseline diarization results for the DIHARD III development and evaluation set using AHC followed by VB-HMM resegmentation.**

| Track   | Partition   | DER (dev)   | DER (eval)   | JER (dev)   | JER (eval)   |
| ------- | ----------- | ----------- | ------------ | ----------- | ------------ |
| Track 1 | core        | 20.25       | 20.65        | 46.02       | 47.74        |
| Track 1 | full        | 19.41       | 19.25        | 41.66       | 42.45        |
| Track 2 | core        | 22.28       | 27.34        | 47.75       | 51.91        |
| Track 2 | full        | 21.71       | 25.36        | 43.66       | 46.95        |


## Detailed scoring logs

For debugging purposes, for both tracks we provide the detailed scoring logs output by ``dscore`` for the DEV set:

- ``recipes/track1/expected_outputs/exp/dihard3_diarization_nnet_1a_dev/scoring``
- ``recipes/track1/expected_outputs/exp/dihard3_diarization_nnet_1a_vbhmm_dev/scoring``
- ``recipes/track2/expected_outputs/exp/dihard3_diarization_nnet_1a_dev/scoring``
- ``recipes/track2/expected_outputs/exp/dihard3_diarization_nnet_1a_vbhmm_dev/scoring``


## RTTMs

We also provide the RTTMs produced by the AHC and VB-HMM resegmentation steps:

- ``recipes/track{1,2}/expected_outputs/exp/dihard3_diarization_nnet_1a_dev/per_file_rttm``
- ``recipes/track{1,2}/expected_outputs/exp/dihard3_diarization_nnet_1a_vbhmm_dev/per_file_rttm``
- ``recipes/track{1,2}/expected_outputs/exp/dihard3_diarization_nnet_1a_eval/per_file_rttm``
- ``recipes/track{1,2}/expected_outputs/exp/dihard3_diarization_nnet_1a_vbhmm_eval/per_file_rttm``


## SAD output

For track 2, the ``segments`` files output by the SAD system are also provided:

- ``recipes/track2/expected_outputs/exp/dihard3_sad_tdnn_stats_decode_dev/segmentation/segments``
- ``recipes/track2/expected_outputs/exp/dihard3_sad_tdnn_stats_decode_eval/segmentation/segments``




# Reproducibility

The above results were produced using the provided recipes with their default settings running on a single machine with the following specs:

- 2 x Intel E5-2680 v4
- 6 x Nvidia GTX 1080
- 384 GB RAM
- Ubuntu 18.04 LTS
-  g++ 7.4.0
- libstdc++.so.6.0.25
- CUDA 10.0

While we have attempted to make the pipeline as deterministic as possible, there are two possible sources of non-determinism:

- **MFCC extraction**

  Because dithering is applied during MFCC extraction, running the recipes on a grid or with a different number of splits (controlled by  ``nj``) will affect the RNG and, consequently, the dithering.
- **CUDA**

  The Track 2 recipe uses GPUs to train the TDNN+stats network used for SAD. Many CUDA kernels are non-deterministic, so results for training this network will differ run-to-run even on the same machine.


## Pretrained SAD model

We have placed a copy of the TDNN+stats SAD model used to produce these results on [Zenodo](https://zenodo.org/). To use this model, download and unarchive the [tarball](https://zenodo.org/record/4299009), then move it to ``recipes/track2/exp``. You will also need to modify ``recipes/track2/run.sh`` to skip stage 1.
