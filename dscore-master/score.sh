
# min=3
# d=25
# find /home/data1/prachis/Dihard_2019/VB_HMM_xvec_Init_2019/rttm_gen_xvec${1}D/${min}frame_400D_downsample_${d}_loop_99.0_statScale_${2}.0/*.rttm  > ../../lists/${min}frame_400D_downsample_${d}_loop_99.0_statScale_${2}.0             

# python score.py -R /home/data1/prachis/Dihard_2019/VB_HMM_xvec_Init_2019/lists/rttmAllGround -S ../../lists/${min}frame_400D_downsample_${d}_loop_99.0_statScale_${2}.0 > temp.scr

# awk '{print $2}' temp.scr 
# grep OVERALL temp.scr

min=3
d=50
var=withstats0.5
var=nosparse
find /home/data1/prachis/Dihard_2019/VB_HMM_xvec_Init_2019/rttm_gen_xvec${1}D/${min}frame_400D_downsample_${d}_loop_90.0_statScale_${2}.0/*.rttm  > ../../lists/${min}frame_400D_downsample_${d}_loop_90.0_statScale_${2}.0           

python score.py -R /home/data1/prachis/Dihard_2019/VB_HMM_xvec_Init_2019/lists/rttmAllGround -S ../../lists/${min}frame_400D_downsample_${d}_loop_90.0_statScale_${2}.0 > temp.scr

awk '{print $2}' temp.scr 
grep OVERALL temp.scr



