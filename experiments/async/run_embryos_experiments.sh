#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

declare -a experiments=(
#"proposal_1"
#"proposal_2"
#"proposal_3"
"clients=10_interval=30_func=const"
#"clients=10_interval=30_func=const_06"
#"clients=10_interval=30_func=poly(0,5)"
#"clients=10_interval=30_func=hinge(1,2)"
#"clients=10_interval=30_func=hinge(10,2)"
#"clients=10_interval=30_func=poly(1)"
#"clients=10_interval=30_func=poly(10)"
#"clients=10_interval=40_func=const"
#"clients=10_interval=40_func=poly(1)"
#"clients=10_interval=40_func=poly(10)"
#"clients=10_interval=50_func=const"
#"clients=10_interval=50_func=poly(1)"
#"clients=10_interval=50_func=poly(10)"
#"clients=10_interval=60_func=const"
#"clients=10_interval=60_func=poly(1)"
#"clients=10_interval=60_func=poly(10)"
#"clients=23_interval=60_func=const"
#"clients=10_interval=60_func=hinge(10,0)"
#"clients=10_interval=60_func=hinge(1,0)"
            )

echo "=============================================================================================="
echo "Starting Embryos Async experiments"
echo "=============================================================================================="
for experiment in ${experiments[@]}; do
    echo "Starting experiment <$experiment>"
    $base_path/../../venv/bin/python async_base.py -c embryos/$experiment.yml
    echo "Experiment <$experiment> done"
    echo -e "\n\n\n\n\n"
    echo "=============================================================================================="
    echo -e "\n\n\n\n\n"
done
echo "Finished executing experiments"