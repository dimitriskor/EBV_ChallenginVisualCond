#!/bin/bash

python3 TS_dataset.py --split glare --TS e2f_35 --classes 1
cd ../../runs/
python3 yolo_TS.py -c e2f_35 -s x -a -d 1 -p
