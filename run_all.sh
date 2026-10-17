#!/bin/bash
# Resume full DMN pipeline: BWM extraction (2 workers, RAM-limited) -> BWM battery -> meso.
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate iblenv
D="$(cd "$(dirname "$0")" && pwd)"
L=$D/run_all.log
echo "start $(date)" >> $L
cd $D/bwm
python 10_extract_full.py 2 >> output/extract_full.log 2>&1; echo "bwm extract exit $? $(date)" >> $L
bash run_full.sh; echo "bwm battery done $(date)" >> $L
cd $D/meso
python 10_extract_meso.py 2 >> output/extract_meso.log 2>&1; echo "meso extract exit $? $(date)" >> $L
python 30_meso_C.py motion 1.0 > output/30_full.log 2>&1; echo "meso C exit $? $(date)" >> $L
python 31_meso_stats.py motion_iti1 A_full > output/31_full.log 2>&1; echo "meso stats exit $? $(date)" >> $L
cd $D/meso && python 32_meso_checks.py motion_iti1 > output/32_full.log 2>&1; echo "meso checks exit $? $(date)" >> $L
echo "ALL DONE $(date)" >> $L
