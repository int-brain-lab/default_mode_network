#!/bin/bash
# Full-data run of all BWM DMN analyses (after 10_extract_full.py finished).
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate iblenv
cd "$(dirname "$0")"
F="Wayland|Warning|return |c /=|nanmean|nanquantile|X -= |d = np"
L=output/run_full.log
: > $L
run() { echo "### $*" >> $L; "$@" 2>&1 | grep -Ev "$F" > /dev/null; echo "exit ${PIPESTATUS[0]}" >> $L; }
# A
run python 20_full_A.py A_full 60 1.5
run env TASKSET=motor_init,move python 20_full_A.py A_full_nofb 60 1.5
run python 20_full_A.py A_full_strict 40 2.0
run env TASKSET=stim python 20_full_A.py A_full_stim 60 1.5
run python 26_A_subsets.py A_full
run python 26_A_subsets.py A_full_nofb
run python 26_A_subsets.py A_full_strict
# B
run python 21_full_B.py spont motion 1
run python 21_full_B.py spont motion_pupil 1
run python 21_full_B.py iti motion 0.5 still
run python 21_full_B.py iti motion 1 still
for b in spont_motion_1s spont_motion_pupil_1s iti_motion_0.5s_still iti_motion_1s_still; do
  run python 22_compare_full.py $b A_full RI_iti z
  run python 24_cv_network.py $b A_full RI_iti
  run python 24_cv_network.py $b A_full suppressed
  run python 25_within_session.py $b all
  run python 25_within_session.py $b core
done
# ITI activity vs behaviour
run python 23_iti_behaviour.py D_full
run python 23_iti_behaviour.py D_full_pupil pupil
echo "ALL DONE" >> $L
