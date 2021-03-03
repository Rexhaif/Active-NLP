#!/bin/bash
python active_cls.py \
    --usemodel=$MODEL \
    --pretrnd=wordvectors/mts_dataset_300.vec \
    --acquiremethod=mnlp \
    --seed=$SEED