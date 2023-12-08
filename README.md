# CrystalCoder Training Dataset Preparation

This repository contains the code for preparing the training dataset for [CrystalCoder](https://huggingface.co/LLM360/CrystalCoder), a 7B-parameter language model pre-trained on code and natural language.

The processed dataset for each phase is available at [CrystalCoderDatasets](https://huggingface.co/datasets/LLM360/CrystalCoderDatasets). This repository contains the code for processing the dataset from scratch. 
Basically, we adhere to the procedure outlined in [Cerebra's Model Zoo](https://github.com/Cerebras/modelzoo/tree/main/modelzoo/transformers/data_processing/scripts). Specifically, the data is prepared in the following steps:

1. Download the untokenized SlimPajama and StarCoder data from the sources.
2. Tokenize the data and concatenate documents to reach the maximum length limit. For the SlimPajama dataset, we evenly divided the tokenized files into two sections, categorizing them by the evenness or oddness of their file numbers for use in Stage 1 and Stage 2, respectively.
3. Apply Fill-In-the-Middle (FIM) augmentation on the tokenized StarCoder data.
4. Shuffle data within each domain and across epochs if there are multiple epochs.

## Step 1: Data and code downloading
```
mkdir data
cd data

# SlimPajama data

# StarCoder data
git lfs install
git clone https://huggingface.co/datasets/bigcode/starcoderdata

cd ../

# Code
git clone https://github.com/Cerebras/modelzoo.git
```


## Step 2: Tokenization and Sequences Concatenation

We tokenize the [SlimPajama dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B) (in `jsonl` format) and [StarCoder dataset](https://huggingface.co/datasets/bigcode/starcoderdata) (in `parquet` format) to `hdf5` format. This is done using the [`create_hdf5_dataset.py`](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/scripts/hdf5_preprocessing/create_hdf5_dataset.py) script.

### SlimPajama data

#### Tokenization

#### Data split

The script below is employed to divide the SlimPajama data into two equal portions based on the even or odd nature of their file numbers. The subset with even-numbered files is utilized for Stage 1, while the odd-numbered subset is designated for Stage 2.

```
for i in `ls | grep train_packed | grep -v "_part[01]of2"`
do
	echo $i
	for part in {0..1}
	do
		echo "  Part $part"
		dirname="${i}_part${part}of2"
		mkdir -p $dirname
		pushd . >&/dev/null
		cd $dirname
		for h5chunk in `ls ../$i/data-*.h5 | sort`
		do
			chunkid=`echo $h5chunk | sed 's/.*data-[0]*//' | sed 's/\.h5//' | sed 's/^$/0/'`
			if [ $(($chunkid % 2)) == $part ]
			then
				ln -s $h5chunk
			fi
		done
		popd >&/dev/null
	done
done
```


### StarCoder data

First, we convert the original `parquet` format to `jsonl` format.

```
python parquet2jsonl.py
```

Next, we proceed to tokenize the data related to StarCoder for Stage 2 and Stage 3, respectively.

#### Stage 2

We tokenize the `jsonl` files from all programming languages together:

```
python -B modelzoo/transformers/data_processing/scripts/hdf5_preprocessing/create_hdf5_dataset.py LMData \ 
    --params configs/star_tokenizer_config.yaml \ 
    --input_dir ./data/starcoderdata_jsonl --eos_id 2 --pad_id 2 \ 
    --max_seq_length 2048 --output_dir ./data/starcoderdata_tokenized \ 
    --seed 45 --processes 4 --split_text_to_tokenize True \ 
    --ignore_bos_in_split_text True \ 
    --encoder_file ./tokenizer.json
```

#### Stage 3

Here we tokenize the subfolders: `Python`, `HTML`, `JaveScript`, `CSS` independently using similar scripts.
```
bash scripts/stage3_tokenization_script.sh
```


## Step 3: FIM Augmentation for StarCoder data

In the tokenized StarCoder dataset, we implement **token-level** FIM augmentation while maintaining a constant SPM rate of 0.5, utilizing the `fim_hdf5.py` script from this repository. For stage 2, the FIM rate is set at 0.9, whereas in stage 3, it is lowered to 0.3. Across both stages, we train on the corresponding StarCoder data over several epochs. FIM is applied independently to each epoch. Consequently, we prepare and store all the data for each epoch on disk prior to beginning the training process.

#### Stage 2

```
python fim_hdf5_stage2.py
```

#### Stage 3
```
python fim_hdf5_stage3.py
```

## Step 3: Shuffling

We shuffle and mix data from different sources and epochs for each stage as per the guidelines in [`h5_dataset_shuffle.py`](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/scripts/hdf5_shuffling/h5_dataset_shuffle.py).

```
bash scripts/shuffle.sh
```