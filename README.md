# CrystalCoder Training Dataset Preparation

This repository contains the code for preparing the training dataset for [CrystalCoder](https://huggingface.co/LLM360/CrystalCoder), a 7B-parameter language model pre-trained on code and natural language.

The processed dataset for each phase is available at [CrystalCoderDatasets](https://huggingface.co/datasets/LLM360/CrystalCoderDatasets). This repository contains the code for processing the dataset from scratch. 
Basically, we adhere to the procedure outlined in [Cerebra's Model Zoo](https://github.com/Cerebras/modelzoo/tree/main/modelzoo/transformers/data_processing/scripts).

#### Step 1: Tokenization

We tokenize the [SlimPajama dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B) (in `jsonl` format) and [StarCoder dataset](https://huggingface.co/datasets/bigcode/starcoderdata) (in `parquet` format) to `hdf5` format. This is done using the [`create_hdf5_dataset.py`](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/scripts/hdf5_preprocessing/create_hdf5_dataset.py) script.

#### Step 2: FIM Augmentation

In the tokenized StarCoder dataset, we implement token-level FIM augmentation while maintaining a constant SPM rate of 0.5, utilizing the `fim_hdf5.py` script from this repository. For stage 2, the FIM rate is set at 0.9, whereas in stage 3, it is lowered to 0.3. Across both stages, we train on the corresponding StarCoder data over several epochs. FIM is applied independently to each epoch. Consequently, we prepare and store all the data for each epoch on disk prior to beginning the training process.

#### Step 3: Shuffling and Mixing

We shuffle and mix data from different sources and epochs for each stage as per the guidelines in [`h5_dataset_shuffle.py`](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/scripts/hdf5_shuffling/h5_dataset_shuffle.py).
