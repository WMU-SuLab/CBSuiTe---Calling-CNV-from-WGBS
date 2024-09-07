mode="germline"
filenames="./demo_data/bam/*.bam"
source activate cbsuite #or your env name
mkdir -p processed_data
mkdir -p demo_data/depth

#for filename in $filenames; do
#    echo $filename
#    f="$(basename -- $filename)"
#    samtools index $filename
#    sambamba depth region -t 4 -L hg38_bin100.bed $filename > ./demo_data/depth/$f.txt
#done

python -u scripts/cal_depth.py -i ./demo_data/bam -o ./demo_data/depth -t 4

if [ "$mode" == "germline" ]; then
    python -u scripts/preprocess_sample_call_germline.py \
        -rd ./demo_data/depth/ \
        -methy ./demo_data/methylation/ \
        -o ./processed_data

elif [ "$mode" == "somatic" ]; then
    python -u scripts/preprocess_sample_call_somatic.py -h
else
    echo "Invalid mode specified. Please use 'germline' or 'somatic'."
fi

