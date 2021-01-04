#!/bin/bash
#NODELIST=$1
JOB_NAME=$1
GYM_ID=$2
LOGS_DIR='/home/krzywicki/logs'
#SBATCH --nodelist="$NODELIST"

(
cat << EOF
#!/bin/bash
#SBATCH --job-name="$JOB_NAME"
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:1
#SBATCH --output="$LOGS_DIR/$JOB_NAME.txt"

bash -c ". ~/anaconda3/etc/profile.d/conda.sh;conda activate rlpyt;python run_model_shaping.py --gym_id '$GYM_ID' --run_ID $3"
EOF
) | sbatch
echo "Spawned training process."
