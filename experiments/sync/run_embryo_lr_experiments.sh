#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

#experiments=("test3")
#experiments=("test1" "test2" "test3")
#experiments=("fedavg_r100_e1_lr1e5" "fedavg_r100_e1_lr1e4" "fedavg_r100_e1_lr1e3")
#experiments=("fedavg_r100_e1_lr1e4" "fedavg_r100_e1_lr1e3")
experiments=("fedavg_r100_e1_lr1e5" "fedavg_r100_e1_lr5e5" "fedavg_r100_e1_lr1e4" "fedavg_r100_e1_lr5e4" "fedavg_r100_e1_lr1e3")
echo "=============================================================================================="
echo "Starting Learning rate experiments"
echo "=============================================================================================="
for experiment in ${experiments[@]}; do
    echo "Starting experiment <$experiment>"
    $base_path/../../venv/bin/python sync_base.py -c embryos/$experiment.yml
    echo "Experiment <$experiment> done"
    echo -e "\n\n\n\n\n"
    echo "=============================================================================================="
    echo -e "\n\n\n\n\n"
done
echo "Finished executing experiments"