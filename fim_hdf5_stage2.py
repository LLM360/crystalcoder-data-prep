import os
import glob
from tqdm.auto import tqdm
import json
import numpy as np
import random
from functools import partial
from itertools import repeat
from transformers import AutoTokenizer
from transformers.utils import logging as hf_tx_logging
from tokenizers import Tokenizer
from math import ceil
from multiprocessing import Pool, RLock
import copy
import h5py

import logging
logging.basicConfig(level=logging.INFO)
# ignore transformers warning for seq len > model seq len, for pbars to update inplace
hf_tx_logging.set_verbosity_error()


TOKENIZED_DATASETS_DIR = './data/starcoderdata_tokenized/'
OUTPUT_DIR = './data/StarCoder_fim/epoch'
SEED = 0
NUM_PROCESSES = 96


FIM_RATE = 0.9
NUM_EPOCHS = 2
OUTPUT_DIRS = [OUTPUT_DIR + '_' + str(i) for i in range(NUM_EPOCHS)]
for output_dir in OUTPUT_DIRS:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



TOKENIZER_NAME = 'huggyllama/llama-7b'
TOKENIZER_PATH = './tokenizer.json'
CONTEXT_LENGTH = 2048
MULTIPROCESSING_BUFFERSIZE = 12800
MULTIPROCESSING_CHUNKSIZE = 100

ADDITIONAL_SPECIAL_TOKENS = [
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<gh_stars>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>"
]


def tokenize_text(text, tokenizer):
    return tokenizer(text, add_special_tokens=False)['input_ids']

tokenizer_global = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
tokenizer_global.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})

suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
    tokenizer_global.vocab[tok]
    for tok in ['<fim_suffix>', '<fim_prefix>', '<fim_middle>', '<fim_pad>']
)
eos_id = tokenizer_global.eos_token_id


# From https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L339
def permute(sample, tokenizer, fim_rate, spm_rate, truncate_or_pad):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it. 
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    # suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
    #     tokenizer.vocab[tok]
    #     for tok in ['<fim_suffix>', '<fim_prefix>', '<fim_middle>', '<fim_pad>']
    # )


    if np.random.binomial(1, fim_rate):  # sample bernoulli dist

        contents = tokenizer.decode(sample, skip_special_tokens=False)

        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(
                np.random.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            logging.info(len(contents))
            logging.info(contents)
            logging.info(e)
            raise e

        prefix = contents[:boundaries[0]]
        middle = contents[boundaries[0]:boundaries[1]]
        suffix = contents[boundaries[1]:]

        prefix = np.array(
            tokenize_text(prefix, tokenizer=tokenizer), dtype=np.int64)
        middle = np.array(
            tokenize_text(middle, tokenizer=tokenizer), dtype=np.int64)
        suffix = np.array(
            tokenize_text(suffix, tokenizer=tokenizer), dtype=np.int64)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - sample.shape[0]
            if diff > 0:  # too long
                if suffix.shape[0] <= diff:  # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    return sample
                suffix = suffix[:suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = np.concatenate(
                    [suffix, np.full((-1 * diff), pad_tok_id)])

        if np.random.binomial(1, spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate([
                [prefix_tok_id, suffix_tok_id], suffix,
                [middle_tok_id], prefix, middle
            ])
        else:
            # PSM
            new_sample = np.concatenate([
                [prefix_tok_id], prefix,
                [suffix_tok_id], suffix,
                [middle_tok_id], middle
            ])
    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample


def fim(sample_array, tokenizer, fim_rate, spm_rate):
    # sample_array is (3, 2048). [0, :] is token id, [1, :] is loss mask, [2, :] is label
    # sample_array[n, 2, -1] == sample_array[n+1, 0, 0]
    sample = sample_array[0, :]
    sample_len = sample.shape[0]

    if fim_rate != 0:
        assert (fim_rate <= 1 and fim_rate >= 0), \
            "FIM rate must be a probability 0 <= rate <= 1"

        segment_breaks = np.argwhere(sample == eos_id)  # split sample by document

        if segment_breaks.shape != (0, 1):
            # then there is an EOD token in this example
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(segment_breaks):
                # Only permute non-empty segments.
                if loc - curr_start_position > 0:
                    permuted = permute(
                        sample=sample[curr_start_position:loc],
                        tokenizer=tokenizer,
                        fim_rate=fim_rate,
                        spm_rate=spm_rate,
                        truncate_or_pad=False,
                    )
                    new_samples += [permuted, [eos_id]]

                curr_start_position = loc + 1  # jump over the EOD token
            permuted = permute(
                sample=sample[curr_start_position:],
                tokenizer=tokenizer,
                fim_rate=fim_rate,
                spm_rate=spm_rate,
                truncate_or_pad=False,
            )
            new_samples.append(permuted)

            sample = np.concatenate(new_samples)
        else:
            sample = permute(
                sample=sample,
                tokenizer=tokenizer,
                fim_rate=fim_rate,
                spm_rate=spm_rate,
                truncate_or_pad=False,
            )

        new_labels = np.concatenate([sample[1:], [eos_id]])

        assert sample.shape[0] == new_labels.shape[0]

    # Truncate or pad sequence to max-length
    diff = sample.shape[0] - sample_len
    new_masks = sample_array[1, :]
    if diff > 0:  # too long
        sample = sample[:sample_len]
        new_labels = new_labels[:sample_len]
    elif diff < 0:  # too short
        sample = np.concatenate([sample, np.full((-1 * diff), pad_tok_id)])
        new_labels = np.concatenate([new_labels, np.full((-1 * diff), pad_tok_id)])
        new_masks[diff:] = 0
    try:
        assert sample.shape[0] == sample_len
        assert new_masks.shape[0] == sample_len
        assert new_labels.shape[0] == sample_len
    except:
        logging.info(sample.shape, sample_len)
        logging.info(new_masks.shape, sample_len)
        logging.info(new_labels.shape, sample_len)
        raise AssertionError
    # end FIM-specific code
    return np.stack([sample, new_masks, new_labels], axis=0)


def process_example(example, tokenizer, fim_rate, spm_rate):
    examples = [fim(
        sample_array=copy.deepcopy(example),
        tokenizer=tokenizer,
        fim_rate=fim_rate,
        spm_rate=spm_rate
    ) for _ in range(NUM_EPOCHS)]
    return examples


def process_file(file, rank, tokenizer, fim_rate, spm_rate):
    try:
        # file, rng = file
        np.random.seed(SEED + rank)
        random.seed(SEED + rank)
        output_files = [output_dir + '/' + file.split('/')[-1] for output_dir in OUTPUT_DIRS]
        with h5py.File(file, 'r') as in_h5:
            data_array = np.array(in_h5['data'])
            n_examples = in_h5.attrs["n_examples"]
            msl = data_array.shape[-1]
        data_fim_arrays = [[] for _ in range(NUM_EPOCHS)]
        logging.debug(f"Started processing samples in {file}.")
        with tqdm(desc=f"File {file} progress", position=((rank+1)*2), total=n_examples, leave=False) as pbar:
            for j, line in enumerate(data_array):
                if j % 1000 == 0:
                    pbar.update(1000)
                    logging.debug('finished processing {} lines for file {}'.format(j, file))
                line_fim_epochs = process_example(line, tokenizer, fim_rate, spm_rate)
                assert len(line_fim_epochs) == NUM_EPOCHS
                for i in range(NUM_EPOCHS):
                    data_fim_arrays[i].append(line_fim_epochs[i])
            pbar.update(n_examples - (j * 1000))
            logging.debug(f"Finished processing: {file}. Writing to HDF5.")
        for i in range(NUM_EPOCHS):
            data_fim_arrays[i] = np.stack(data_fim_arrays[i], axis=0)
        for i, output_file in enumerate(output_files):
            with h5py.File(output_file, 'w') as f:
                f.attrs["n_examples"] = n_examples
                f.create_dataset(
                    'data', 
                    data=data_fim_arrays[i],
                    dtype="i4",
                    chunks=(1, 3, msl),
                    compression="gzip",
                )
            logging.debug(f"Done writing output file: {output_file}.")
    except Exception as e:
        logging.info("error in processing ", file)
        logging.info(e)


def process_files(process_args):
    files, rank, tokenizer, fim_rate, spm_rate = process_args
    for file in tqdm(files, desc=f"Processed files in process {rank}", position=((rank+1)*2)-1, leave=False):
        process_file(file, rank, tokenizer, fim_rate, spm_rate)


def main(spm_rate=0.):

    # ============== parallel split files per process ================
    logging.info('start parallel processing')
    dataset_files = glob.glob(str(TOKENIZED_DATASETS_DIR)+'/*.h5')

    n_chunks = ceil(len(dataset_files) / NUM_PROCESSES)
    files_per_process = [
        dataset_files[i : i + n_chunks] for i in range(0, len(dataset_files), n_chunks)
    ]

    # add assertion to check if the split files per process is equal to 
    # total files ensuding there's no duplication
    assert len(dataset_files) == sum(map(lambda l: len(l), files_per_process))
    # https://github.com/tqdm/tqdm/blob/master/examples/parallel_bars.py#L46
    tqdm.set_lock(RLock())
    with Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), processes=NUM_PROCESSES) as pool:
        pbar = tqdm(
                pool.imap(
                    process_files, 
                    zip(
                        files_per_process, 
                        range(len(files_per_process)),
                        repeat(tokenizer_global), 
                        repeat(FIM_RATE),
                        repeat(spm_rate),
                    )
                ),
                desc="Total progress",
                total=len(files_per_process),
            )
        for _ in pbar:
            pbar.update()


if __name__ == '__main__':
    main(0.5)