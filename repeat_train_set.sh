FOLDER=${1}
N_REPS=${2}

if [ -z ${FOLDER} ] || [ -z ${N_REPS} ]
then
    echo Missing args
    exit 1
fi

for f in targets_train.txt midi_train.txt
do
    for i in $(seq ${N_REPS})
    do
        cat ${FOLDER}/${f} >> ${FOLDER}/repeated_${f}
    done
    mv ${FOLDER}/repeated_${f} ${FOLDER}/${f} 
done

echo "training inputs and targets have been repeated ${N_REPS} times" >> ${FOLDER}/repeat_train_set_info.txt


