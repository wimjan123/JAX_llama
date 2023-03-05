SOURCE_FOLDER=""             # where all llama files should come from
TARGET_FOLDER=""             # where all jax files should end up
MODEL_SIZE="7B,13B,30B,65B"  # edit this list with the model sizes you wish to download

cp ${SOURCE_FOLDER}"/tokenizer.model" ${TARGET_FOLDER}
cp ${SOURCE_FOLDER}"/tokenizer_checklist.chk" ${TARGET_FOLDER}

for i in ${MODEL_SIZE//,/ }
do
    python convert_weights.py --ckpt_dir "${SOURCE_FOLDER}/${i}/" --out_dir "${TARGET_FOLDER}/${i}/" --verbose
done
