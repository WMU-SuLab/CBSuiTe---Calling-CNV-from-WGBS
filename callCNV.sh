source activate cbsuite  #or your env name
mode="germline"

python -u scripts/cbsuite_call.py \
        --model $mode \
        --input ./processed_data \
        --output CNV_${mode} \
        --normalize ${mode}_stats.txt \
        --gpu yes
