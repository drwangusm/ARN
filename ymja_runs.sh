#!/bin/bash
python3 src/run_protocol.py ARN_inter_aug configs/YMJA/ARN_inter.cfg YMJA -n 5
python3 src/run_protocol.py ARN_intra_aug configs/YMJA/ARN_intra.cfg YMJA -n 5
python3 src/run_protocol.py ARN_inter_intra configs/YMJA/ARN_inter+intra.cfg YMJA -F middle -n 5
python3 src/run_protocol.py ARN_fc1_inter_intra configs/YMJA/ARN-fc1_inter+intra.cfg YMJA -F middle -n 5