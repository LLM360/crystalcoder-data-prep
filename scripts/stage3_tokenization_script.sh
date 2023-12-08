python -B modelzoo/transformers/data_processing/scripts/hdf5_preprocessing/create_hdf5_dataset.py LMData  \
	--params ./configs/tokenizer_config_python.yaml \
	--seed 45 --processes 4 --split_text_to_tokenize True \
	--ignore_bos_in_split_text True \
	--encoder_file ./tokenizer.json \
	--input_dir data/starcoderdata_jsonl_split/train/python \
	--output_dir data/starcoderdata_tokenized_stage3_split/train/python

# ====================================================================================================


python -B modelzoo/transformers/data_processing/scripts/hdf5_preprocessing/create_hdf5_dataset.py LMData  \
	--params ./configs/tokenizer_config_html.yaml \
	--seed 45 --processes 4 --split_text_to_tokenize True \
	--ignore_bos_in_split_text True \
	--encoder_file ./tokenizer.json \
	--input_dir data/starcoderdata_jsonl_split/train/html \
	--output_dir data/starcoderdata_tokenized_stage3_split/train/html


# ====================================================================================================


python -B modelzoo/transformers/data_processing/scripts/hdf5_preprocessing/create_hdf5_dataset.py LMData  \
	--params ./configs/tokenizer_config_css.yaml \
	--seed 45 --processes 4 --split_text_to_tokenize True \
	--ignore_bos_in_split_text True \
	--encoder_file ./tokenizer.json \
	--input_dir data/starcoderdata_jsonl_split/train/css \
	--output_dir data/starcoderdata_tokenized_stage3_split/train/css


# ====================================================================================================

python -B modelzoo/transformers/data_processing/scripts/hdf5_preprocessing/create_hdf5_dataset.py LMData  \
	--params ./configs/tokenizer_config_javascript.yaml \
	--seed 45 --processes 4 --split_text_to_tokenize True \
	--ignore_bos_in_split_text True \
	--encoder_file ./tokenizer.json \
	--input_dir data/starcoderdata_jsonl_split/train/javascript \
	--output_dir data/starcoderdata_tokenized_stage3_split/train/javascript
