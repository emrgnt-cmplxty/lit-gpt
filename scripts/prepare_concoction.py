# TODO - Implement GPT
# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import urllib.parse

import numpy as np
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import LlamaTokenizer
import pyarrow as pa
import pyarrow.parquet as pq

load_dotenv()

# Input args

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

tokenizer = "gpt2"
seed = 2357
test_size = 0.01

# Configurations for the dataset and processing

config = {
    "partial-flan": {
        "dataset_name": "MBZUAI/LaMini-instruction",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### Instruction:\n{x}",
            "response": lambda x: f"\n### Response:\n{x}",
        },
    },
    "open-hermes": {
        "dataset_name": "teknium/openhermes",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### Instruction:\n{x}",
            "output": lambda x: f"\n### Response:\n{x}",
        },
    },
    "open-orca": {
        "dataset_name": "Open-Orca/OpenOrca",
        "shuffle": True,
        "format": {
            "system_prompt": lambda x: f"### System Prompt:\n{x}",
            "question": lambda x: f"\n### Instruction:\n{x}",
            "response": lambda x: f"\n### Response:\n{x}",
        },
    },
    "tiny-codes": {
        "dataset_name": "nampdn-ai/tiny-codes",
        "shuffle": True,
        "format": {
            "prompt": lambda x: f"### System:\nYou are a helpful agent.\n\n### Instruction:\n{x}",
            "response": lambda x: f"\n### Response:\n{x}",
        },
    },
    "tiny-cot": {
        "dataset_name": "nampdn-ai/tiny-cot",
        "shuffle": True,
        "format": {
            "source": lambda x: f"You are a helpful assistant that thinks step-by-step to solve instructions.\n### Instruction:\n{x}",
            "rationale": lambda x: f"\n### Response:\nI will begin by writing down my thoughts. {x}",
            "target": lambda x: f"\nGiven my reasoning above, the answer must be {x}.",
        },
    },
    "wizardlm-orca": {
        "dataset_name": "psmathur/WizardLM_Orca",
        "shuffle": True,
        "format": {
            "system": lambda x: f"### System:\n{x}",
            "instruction": lambda x: f"### Instruction:\n{x}",
            "output": lambda x: f"\n### Response:\n{x}",
        },
    },
    "cot-submix": {
        "dataset_name": "conceptofmind/cot_submix_original",
        "shuffle": True,
        "format": {
            "inputs": lambda x: x,
            "targets": lambda x: x,
        },
    },
    "gpt-teacher": {
        "dataset_name": "teknium/GPTeacher-General-Instruct",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### Instruction:\n{x}",
            "input": lambda x: f"\n### Input:\n{x}\n" if x != "" else "",
            "response": lambda x: f"\n### Response:\n{x}",
        },
    },
    "alpaca-cleaned": {
        "dataset_name": "yahma/alpaca-cleaned",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### Instruction:\n{x}",
            "input": lambda x: f"\n### Input:\n{x}\n" if x != "" else "",
            "output": lambda x: f"\n### Response:\n{x}",
        },
    },
    "codealpaca-20k": {
        "dataset_name": "sahil2801/CodeAlpaca-20k",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### Instruction:\n{x}",
            # "input": lambda x: f"", #
            "output": lambda x: f"### Response:\n{x}",
        },
    },
    "unnatural": {
        "dataset_name": "TokenBender/unnatural_code_instructions_20M",
        "shuffle": True,
        "format": {
            "text": lambda x: x,
        },
    },
    "evol-instruct-code-v1": {
        "dataset_name": "nickrosh/Evol-Instruct-Code-80k-v1",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### System:\nYou are a helpful assistant.\n\n### Instruction:\n{x}\n",
            "output": lambda x: f"### Response:\n{x}",
        },
    },
    "open-platypus": {
        "dataset_name": "garage-bAInd/Open-Platypus",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### System:\nYou are a helpful assistant.\n\n### Instruction:\n{x}\n",
            "output": lambda x: f"### Response:\n{x}",
        },
    },
    "evol-codealpaca-v1": {
        "dataset_name": "theblackcat102/evol-codealpaca-v1",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### System:\nYou are a helpful assistant.\n\n### Instruction:\n{x}\n",
            "output": lambda x: f"### Response:\n{x}",
        },
    },
    "link_soul_merge": {
        "dataset_name": "LinkSoul/instruction_merge_set",
        "shuffle": True,
        "format": {
            "conversations": lambda x: f"Below is a conversation between where a human instructs a helpful assistant named gpt, in JSON format.\n{x}",
        },
    },
    "vicuna_70k": {
        "dataset_name": "ehartford/wizard_vicuna_70k_unfiltered",
        "shuffle": True,
        "format": {
            "conversations": lambda x: f"Below is a conversation between where a human instructs a helpful assistant named gpt, in JSON format.\n{x}",
        },
    },
    "chat_arena": {
        "dataset_name": "lmsys/chatbot_arena_conversations",
        "shuffle": True,
        "format": {
            "conversation_a": lambda x: f"Below is a conversation between a human and a helpful agent, in JSON format.\n{x}",
            "conversation_b": lambda x: f"The following is a similar conversation with a different agent responding, in JSON format.\n{x}",
        },
    },
    "kaist_ai_cot": {
        "dataset_name": "kaist-ai/CoT-Collection",
        "shuffle": True,
        "format": {
            "source": lambda x: "### System:\nYou are a helpful assistant that thinks step-by-step to solve instructions. Please encase your final solution in \\boxed{SOLUTION}\n\n### Instruction:\n"
            + x,
            "rationale": lambda x: f"\n### Response:\nI will begin by writing down my thoughts.\n\n{x}",
            "target": lambda x: "\nFrom the logic above, I can see that the answer is \\boxed{"
            + x
            + "}",
        },
    },
    "leetcode_with_solutions": {
        "dataset_name": "mhhmm/leetcode-solutions-python",
        "shuffle": True,
        "format": {
            "code_with_data": lambda x: "Below is a coding problem with the correct solution.\n\n### Problem:\n"
            + x.replace("```python", """\n### Solution:\n```python"""),
            "explanation_only": lambda x: f"And here is an explanation for the solution:\n{x}",
        },
    },
    "tiny_stories": {
        "dataset_name": "skeskinen/TinyStories-GPT4",
        "shuffle": True,
        "format": {
            "prompt": lambda x: f"{x}\n",
            "story": lambda x: f"{x}\n",
            "summary": lambda x: f"### Summary:\n{x}\n",
        },
    },
    "claude_instruct": {
        "dataset_name": "Norquinal/claude_evol_instruct_210k",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### System:\nYou are a helpful assistant.\n\n### Instruction:\n{x}\n",
            "output": lambda x: f"### Response:\n{x}",
        },
    },
    "airoboros": {
        "dataset_name": "jondurbin/airoboros-2.1",
        "shuffle": True,
        "format": {
            "system": lambda x: f"### System Prompt:\n{x.replace('A chat.', 'You are a helpful assistant.')}\n",
            "instruction": lambda x: f"\n### Instruction:\n{x}\n",
            "response": lambda x: f"### Response:\n{x}",
        },
    },
    "databricks-dolly": {
        "dataset_name": "databricks/databricks-dolly-15k",
        "shuffle": True,
        "format": {
            "instruction": lambda x: f"### System:\nYou are a helpful assistant that quickly and accurately answers user questions.\n\n### User:\n{x}\n",
            "response": lambda x: f"### Assistant:\n{x}",
        },
    },
    "claude_chats": {
        "dataset_name": "Norquinal/claude_multiround_chat_30k",
        "shuffle": True,
        "format": {
            "conversations": lambda x: f"\nBelow is a conversation between a human and a helpful assistant, in JSON format.\n{x}",
        },
    },
}
# Short List For Addition

## Has a lot of crufty tokens
# https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations

## Requires extra work -
# https://huggingface.co/datasets/Muennighoff/natural-instructions/viewer/default/train?row=1

## Seems low-ish quality
# https://huggingface.co/datasets/laion/OIG

## Is fucking huge....
# https://huggingface.co/datasets/Open-Orca/FLAN

# Needs some TLC to remove 'as a large language model ...', but otherwise looks good.
# "ultra-chat" : {
#     "dataset_name": "stingning/ultrachat",
#     "shuffle": True,
# }
# https://huggingface.co/datasets/stingning/ultrachat

# Format is not ready
# https://huggingface.co/datasets/camel-ai/math

# Too abstract / high level and requires nuanced reasoning
# https://huggingface.co/datasets/eli5_category/viewer/default/train?row=31

# Probably duplicated by OpenOrca
# https://huggingface.co/datasets/ehartford/dolphin/viewer/default/train?row=10

# A mixture of other datasets..
# https://huggingface.co/datasets/StudentLLM/Open-Wyvern-74k

# Crufty
# https://huggingface.co/datasets/ChristophSchuhmann/basic-math-problems-with-step-by-step-solutions/viewer/default/train?p=1

# QingyiSi/Alpaca-CoT
# Comprehensive, but crufty


file_names = [
    "cot_fs_noopt_train.jsonl.gz",
    "cot_fs_opt_train.jsonl.gz",
    "cot_zs_noopt_train.jsonl.gz",
    "cot_zs_opt_train.jsonl.gz",
    "dialog_fs_noopt_train.jsonl.gz",
    "dialog_fs_opt_train.jsonl.gz",
    "dialog_zs_noopt_train.jsonl.gz",
    "dialog_zs_opt_train.jsonl.gz",
    "flan_fs_noopt_train.jsonl.gz",
    "flan_fs_opt_train_part1.jsonl.gz",
    "flan_fs_opt_train_part2.jsonl.gz",
    "flan_fs_opt_train_part3.jsonl.gz",
    "flan_zs_noopt_train.jsonl.gz",
    "flan_zs_opt_train.jsonl.gz",
    "niv2_fs_noopt_train.jsonl.gz",
    "niv2_fs_opt_train.jsonl.gz",
    "niv2_zs_noopt_train.jsonl.gz",
    "niv2_zs_opt_train.jsonl.gz",
    "t0_fs_noopt_train.jsonl.gz",
    "t0_zs_noopt_train.jsonl.gz",
    "t0_zs_opt_train.jsonl.gz",
]

base_config = {
    "dataset_name": "",
    "shuffle": True,
    "format": {
        "inputs": lambda x: x,
        "targets": lambda x: x,
    },
}
base_path = "/home/owen/test_run/flan_data/"

for file_name in file_names:
    key = file_name.rsplit(".", 2)[0]  # Remove ".jsonl.gz" to get the key
    new_config = base_config.copy()
    new_config["dataset_name"] = base_path + file_name
    config[f"flan_data-{key}"] = new_config


def load_and_split_dataset(config):
    """Load the dataset and split into train and validation sets."""
    if "flan_data" in config["dataset_name"]:
        dataset = load_dataset(
            path="json",
            data_files=config["dataset_name"],
            # split="train",
            num_proc=num_proc_load_dataset,
        )
    else:
        dataset = load_dataset(
            config["dataset_name"],
            num_proc=num_proc_load_dataset,
            token=os.getenv("HF_TOKEN", None),
            # cache_dir="/Volumes/Elements/HuggingFaceDatasets",
        )

    # Only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=config["shuffle"],
    )
    split_dataset["val"] = split_dataset.pop("test")

    return split_dataset


def tokenize_dataset(split_dataset, config):
    """Tokenize the dataset."""
    columns = config["format"].keys()
    format_dict = config["format"]

    hf_token = os.getenv("HF_TOKEN", None)

    # enc = tiktoken.get_encoding(tokenizer)
    enc = LlamaTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b", token=hf_token, add_eos_token=True
    )

    def process(example):
        # Use the formatter to format each column
        text = urllib.parse.unquote(
            "\n".join(
                [f"{format_dict.get(col, '{}')} {example[col]}" for col in columns]
            )
        )
        ids = enc.encode(text)
        ids.append(enc.eos_token_id)
        return {"ids": ids, "len": len(ids)}

    print("Printing example splits")
    for i in range(1):
        x = urllib.parse.unquote(
            "\n".join(
                [
                    f"{format_dict[col](split_dataset['train'][i][col])}"
                    for col in columns
                ]
            )
        )
        print("-" * 100)
        print(x)
        print("-" * 100)

    tokenized = split_dataset.map(
        process,
        remove_columns=columns,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    return tokenized


def save_to_parquet(tokenized_dataset, output_path):
    """Save the dataset in Parquet format."""
    table = pa.Table.from_pandas(tokenized_dataset.to_pandas())
    with pa.OSFile(output_path, "wb") as f:
        with pa.RecordBatchFileWriter(f, table.schema) as writer:
            writer.write_table(table)


if __name__ == "__main__":
    # Load and split dataset based on the configuration
    print("Splitting now...")
    results_train = None
    results_val = None
    total_tokens = 0
    for key in config.keys():
        print(f"Handling config key = {key}")

        print("Loading dataset now...")
        selected_config = config[key]
        split_data = load_and_split_dataset(selected_config)
        # Get the tokenizer encoding based on the configuration
        print("Tokenizing now...")
        # Tokenize the data based on the configuration
        tokenized_data = tokenize_dataset(split_data, selected_config)
        tokens = np.sum(tokenized_data["train"]["len"]) / 1e6
        print(f"Tokens = {tokens}M")
        total_tokens += tokens
        print(f"Total Tokens = {total_tokens}M")
        if not results_train:
            results_train = tokenized_data["train"]
            results_val = tokenized_data["val"]
        else:
            results_train = concatenate_datasets(
                [results_train, tokenized_data["train"]]
            ).shuffle(seed=seed)
            results_val = concatenate_datasets(
                [results_val, tokenized_data["val"]]
            ).shuffle(seed=seed)

        # Save the Arrow table as a Parquet file
        if not os.path.exists(key):
            os.mkdir(key)
        print("Saving results...")
        table = results_train.with_format("arrow").data
        # Save the Arrow table as a Parquet file
        df = results_train.to_pandas()
        # Convert the DataFrame to an Arrow table
        table = pa.Table.from_pandas(df[["ids"]])
        pq.write_table(table, f"{key}/{key}_train.parquet")

        table = results_val.with_format("arrow").data
        # Save the Arrow table as a Parquet file
        df = results_val.to_pandas()
        # Convert the DataFrame to an Arrow table
        table = pa.Table.from_pandas(df[["ids"]])
        pq.write_table(table, f"{key}/{key}_val.parquet")
