#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

experiments=("fedavg_r50_e2_lr1e4" "fedavg_r20_e5_lr1e4" "fedavg_r100_e2_lr1e4" "fedavg_r100_e5_lr1e4")
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