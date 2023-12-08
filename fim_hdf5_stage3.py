import os
import glob
import tqdm
import json
import numpy as np
import random
from functools import partial
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import multiprocessing
import copy

import logging
logging.basicConfig(level=logging.INFO)

import h5py

# ==================== Modify these ====================
FIM_RATE = 0.3
SPLITS = ['train']
LANGUAGES = ['python', 'css', 'html', 'javascript']
TOKENIZED_DATASETS_DIRS = ['data/starcoderdata_tokenized_stage3_split/' + x + '/' + y for x in SPLITS for y in LANGUAGES]
OUTPUT_DIRS0 = [x.replace('data/starcoderdata_tokenized_stage3_split', 'data/starcoderdata_tokenized_stage3_split_fim_' + str(FIM_RATE)) for x in TOKENIZED_DATASETS_DIRS]
SEED = 0
NUM_PROCESSES = 32
MULTIPROCESSING_CHUNKSIZE = 100
NUM_SELECTED_FILES = None # set to None if using all files

# =======================================================


NUM_EPOCHS = 3
OUTPUT_DIRS = [[x + '_' + str(i) for i in range(NUM_EPOCHS)] for x in OUTPUT_DIRS0]
for output_dir1 in OUTPUT_DIRS:
    for output_dir in output_dir1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)



TOKENIZER_NAME = 'huggyllama/llama-7b'
TOKENIZER_PATH = './tokenizer.json'
CONTEXT_LENGTH = 2048
MULTIPROCESSING_BUFFERSIZE = 12800

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
    # sample = np.array(token_ids, dtype=np.int64)
    sample = sample_array[0, :]
    sample_len = sample.shape[0]

    permute_fn = partial(
        permute,
        tokenizer=tokenizer,
        fim_rate=fim_rate,
        spm_rate=spm_rate,
        truncate_or_pad=False)

    if fim_rate != 0:
        assert (fim_rate <= 1 and fim_rate >= 0), \
            "FIM rate must be a probability 0 <= rate <= 1"

        eod = tokenizer.eos_token_id
        pad = tokenizer.vocab['<fim_pad>']

        segment_breaks = np.argwhere(sample == eod)  # split sample by document

        if segment_breaks.shape != (0, 1):
            # then there is an EOD token in this example
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(segment_breaks):
                # Only permute non-empty segments.
                if loc - curr_start_position > 0:
                    # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                    permuted = permute_fn(
                        sample=sample[curr_start_position:loc])
                    new_samples += [permuted, [eod]]

                curr_start_position = loc + 1  # jump over the EOD token
            # Permute the segment after the last EOD
            permuted = permute_fn(sample=sample[curr_start_position:])
            new_samples.append(permuted)

            sample = np.concatenate(new_samples)
        else:
            sample = permute_fn(sample=sample)

        new_labels = np.concatenate([sample[1:], [eod]])

        assert sample.shape[0] == new_labels.shape[0]

    # Truncate or pad sequence to max-length
    diff = sample.shape[0] - sample_len
    new_masks = sample_array[1, :]
    if diff > 0:  # too long
        sample = sample[:sample_len]
        new_labels = new_labels[:sample_len]
    elif diff < 0:  # too short
        sample = np.concatenate([sample, np.full((-1 * diff), pad)])
        new_labels = np.concatenate([new_labels, np.full((-1 * diff), pad)])
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


def process_file(file, process_fn, id):
    try:
        file, rng = file
        np.random.seed(SEED + rng)
        random.seed(SEED + rng)
        output_files = [output_dir + '/' + file.split('/')[-1] for output_dir in OUTPUT_DIRS[id]]
        data_array = np.array(h5py.File(file, 'r')['data'])
        data_fim_arrays = [[] for _ in range(NUM_EPOCHS)]
        for j, line in enumerate(data_array):
            if j % 1000 == 0:
                logging.info('finished processing {} lines for file {}'.format(j, file))
            line_fim_epochs = process_fn(line)
            assert len(line_fim_epochs) == NUM_EPOCHS
            for i in range(NUM_EPOCHS):
                data_fim_arrays[i].append(line_fim_epochs[i])
        for i in range(NUM_EPOCHS):
            data_fim_arrays[i] = np.stack(data_fim_arrays[i], axis=0)
        for i, output_file in enumerate(output_files):
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('data', data=data_fim_arrays[i], dtype='i4', compression='gzip')
    except Exception as e:
        logging.info("error in processing ", file)
        logging.info(e)




def main(spm_rate=0.):
    process_fn = partial(
        process_example,
        tokenizer=tokenizer_global,
        fim_rate=FIM_RATE,
        spm_rate=spm_rate)

    # ============== parallel ================
    logging.info('start parallel processing')
    
    total_files = 0
    for id, TOKENIZED_DATASETS_DIR in enumerate(TOKENIZED_DATASETS_DIRS):
        logging.info('processing {}'.format(TOKENIZED_DATASETS_DIR))
        process_file_fn = partial(process_file, process_fn=process_fn, id=id)

        pool = multiprocessing.Pool(processes=NUM_PROCESSES)
        files = glob.glob(str(TOKENIZED_DATASETS_DIR)+'/*.h5')
        if NUM_SELECTED_FILES is not None:
            files = files[:NUM_SELECTED_FILES]
        files_sublists = [files[i:i+NUM_PROCESSES] for i in range(0, len(files), NUM_PROCESSES)]
        for inter_id, files_sublist in enumerate(files_sublists):
            pool.map(process_file_fn, zip(files_sublist, range(total_files + inter_id*NUM_PROCESSES, total_files + (inter_id+1)*NUM_PROCESSES)))
        
        total_files += len(files)

if __name__ == '__main__':
    main(0.5)