import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from core.common.data_handler import select_subset
from core.common.utils import (
    create_id_x_column,
    load_dataframe,
    read_json,
)
from core.llm_encoder.formatter import apply_template
import os


# `original_df` structure:
# - `original_df` contains rows where each example follows a SINGLE rule (e.g., "create a sentence containing the word `apple`").
# - SINGLE rules can be grouped into broader `macro-rules`, which represent a common goal (e.g., "create a sentence containing the words `apple` and `banana`").
# - The column `hash_rule_sample` is used to identify a unique tuple (macro-rule, sample): all SINGLE rules that belong to the same `macro-rule` will have the same `hash_rule_sample`.
# - Note that a given `macro-rule` can have multiple sets of examples. To distinguish these, we:
#     - Assign a unique `hash_rule_sample` (`mr``) even if the `macro-rule` is conceptually the same across rows.
#     - Use the `positive_sample_id` (`ps`) to indicate that these entries correspond to different sets of positive examples for the same `macro-rule`.
# Example of `original_df` (only meaningful columns):
#    | single_rule_id | macro_rule_id | single_rule     | sentence                                           | pos_sample_column |
#    |----------------|---------------|-----------------|----------------------------------------------------|-------------------|
#    | 1              | 1001          | Include "apple" | "I ate an apple and a banana for breakfast."       | 1                 |
#    | 2              | 1001          | Include "banana"| "I ate an apple and a banana for breakfast."       | 1                 |
#    | 3              | 1002          | Include "apple" | "She bought an apple at the store."                | 2                 |
#    | 4              | 1003          | Include "banana"| "He enjoys a banana smoothie."                     | 3                 |
#    | 5              | 1004          | Include "cat"   | "The cat and the dog are sleeping on the mat."     | 4                 |
#    | 6              | 1004          | Include "dog"   | "The cat and the dog are sleeping on the mat."     | 4                 |
#    | 7              | 1005          | Include "cat"   | "A cat walked across the street, chased by a dog." | 4                 |
#    | 8              | 1005          | Include "dog"   | "A cat walked across the street, chased by a dog." | 4                 |
# At the same time, we want to identify the `real-negatives`.
# Those are sentences on our training set that respect multiple rules at once (but are only labelled for one):
# E.g.,
# 1) RULE: ["sun", "clouds"] > "Today the sun comes out from the clouds"
# 2) RULE: ["sun", "clouds", "rain"] > "As the sun disappeared, clouds covered the sky threatening heavy rain."
# Clearly, sentence 2 respects rule 1. We don't want sentence 2 to be used as a negative example for rule 1.


class RuleConstrainerDataset(Dataset):
    def __init__(self, paths_data, macro_rule_id, subsample=False, seed=29):

        self.df = load_dataframe(file_paths=paths_data)
        self.macro_rule_id = macro_rule_id
        # 0. Subsample the data if necessary
        if subsample:
            self.df = select_subset(df=self.df, n_subsample=subsample, seed=seed)
        # 1. Extract unique macro rules
        self.unique_macro_rules = self.df[macro_rule_id].unique()

    def __repr__(self):
        schema_df = pd.DataFrame(
            {
                "Column": self.df.columns,
                "Type": [self.df[col].dtype for col in self.df.columns],
                "Sample Value": [
                    self.df[col].iloc[0] for col in self.df.columns
                ],  # Optional: first sample value
            }
        )
        return schema_df.to_string(index=False)

    def __len__(self):
        """The number of usable training samples correspond to the number of the unique macro rules ids.
        Returns:
            int: Number of unique macro rules in the dataset
        """
        return len(self.unique_macro_rules)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def preprocess_dataset(
        self, df, rule_column, input_column, path_prompt_template, cache_dir
    ):
        cache_file = (
            os.path.join(cache_dir, "preprocessed_df.parquet") if cache_dir else None
        )
        tuples_column_template = [
            (rule_column, "rules"),
            (input_column, "sentences"),
        ]
        if cache_file and os.path.isfile(cache_file):
            copy_df = pd.read_parquet(cache_file)
        else:
            # 0. Create copy of original dataframe
            copy_df = df.copy()
            # 1. Read the prompt template file
            prompt_template = read_json(path_json=path_prompt_template)
            # 2. Format columns with the prompt template
            for column, template_for in tuples_column_template:
                copy_df = apply_template(
                    df=copy_df,
                    column=column,
                    template_for=template_for,
                    prompt_template=prompt_template,
                )
            # 3. Remove the original non-template columns
            copy_df.drop([rule_column, input_column], axis=1, inplace=True)
            for _, column in tuples_column_template:
                copy_df[f"hash_templated_{column}"] = create_id_x_column(
                    df=copy_df, column=column
                )
            if cache_file:
                copy_df.to_parquet(cache_file, index=False)
        return copy_df, tuples_column_template

    def extract_tokens(
        self, tuples_column_template, pretrained_model, huggingface_cache, cache_dir
    ):
        cache_file = (
            os.path.join(cache_dir, "tokenized_input.pt") if cache_dir else None
        )
        if cache_file and os.path.isfile(cache_file):
            embeddings = torch.load(cache_file, weights_only=True)
        else:
            # 1. Load tokenizer for the chosen pretrained model
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=pretrained_model,
                cache_dir=huggingface_cache,
            )
            embeddings = {}
            # 2. Tokenize the dataset
            for _, column in tuples_column_template:
                # Hashes will be the keys of our dictionaries of embeddings
                ds = Dataset.from_pandas(
                    self.df[[column, f"hash_templated_{column}"]], preserve_index=False
                )
                hashes_x_column = self.df[f"hash_templated_{column}"].tolist()

                def tok_fn(batch):
                    return tokenizer(
                        batch[column],
                        truncation=True,
                        padding=False,
                    )

                ds_tok = ds.map(
                    tok_fn,
                    batched=True,
                    batch_size=2000,  # tune: 512–4000
                    num_proc=10,  # tune: #cores
                    remove_columns=[column],  # optional
                )

                def concat_tensors(ds_tok):

                    input_ids_list = ds_tok["input_ids"]
                    attn_list = ds_tok["attention_mask"]
                    # Choose target length: model max, or clamp if you want smaller
                    # tokenizer.model_max_length can be huge for some tokenizers; protect yourself:
                    target_len = min(
                        max(len(x) for x in input_ids_list),  # longest sequence in data
                        (
                            getattr(tokenizer, "model_max_length", 512)
                            if getattr(tokenizer, "model_max_length", 512)
                            and getattr(tokenizer, "model_max_length", 512) < 10**6
                            else 512
                        ),
                    )

                    pad_id = (
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else 0
                    )

                    n = len(input_ids_list)
                    input_ids = torch.full((n, target_len), pad_id, dtype=torch.long)
                    attention_mask = torch.zeros((n, target_len), dtype=torch.long)

                    for i, (ids, mask) in enumerate(zip(input_ids_list, attn_list)):
                        L = min(len(ids), target_len)
                        input_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
                        attention_mask[i, :L] = torch.tensor(mask[:L], dtype=torch.long)

                    # 3) Build the tensor you require: [N, 2, L]
                    concat_tensors = torch.stack((input_ids, attention_mask), dim=1)
                    return concat_tensors

                concat_tensors = concat_tensors(ds_tok)

                index_2_embedding = {
                    index: embedding
                    for index, embedding in zip(hashes_x_column, concat_tensors)
                }
                embeddings[column] = index_2_embedding

            if cache_file:  # Make sure this value wasn't None
                torch.save(embeddings, cache_file)
        return embeddings
