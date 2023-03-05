SOURCE_FOLDER="/shared/csnell/llama_weights/"             # where all llama files should come from
TARGET_FOLDER="/shared/csnell/llama_weights_jax/"             # where all jax files should end up
MODEL_SIZE="65B"  # edit this list with the model sizes you wish to download

cp ${SOURCE_FOLDER}"/tokenizer.model" ${TARGET_FOLDER}
cp ${SOURCE_FOLDER}"/tokenizer_checklist.chk" ${TARGET_FOLDER}

for i in ${MODEL_SIZE//,/ }
do
    python convert_weights.py --ckpt_dir "${SOURCE_FOLDER}/${i}/" --out_dir "${TARGET_FOLDER}/${i}/" --verbose
done
