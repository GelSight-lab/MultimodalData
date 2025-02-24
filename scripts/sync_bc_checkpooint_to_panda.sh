#!/usr/bin/env bash
# sync evaluation data from panda workstation
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rsync -avz --progress --exclude={'*.egg-info','.git','data','outputs','output','wandb','checkpoints','optimizer_AdamW.pt'} -e 'ssh -p 2022' "$SCRIPT_DIR/../checkpoints_supercloud" "azhao@gelsight-panda-0.csail.mit.edu:/home/azhao/src/FoundationTactile/TheProbe/ProbingPanda/"
