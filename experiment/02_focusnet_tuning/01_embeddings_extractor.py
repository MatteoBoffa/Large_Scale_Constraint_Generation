import torch
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from core.llm_encoder.encoder_func import encoding_loop, tokenizing_function
from core.common.utils import (
    get_logger,
    load_dataframe,
    convert_dataframes,
    format_with_hashes,
    read_json,
    concatenate_paths,
    create_id_x_column,
    save_json,
)
from core.llm_encoder.option_parser import get_encoding_options
from core.llm_encoder.formatter import apply_template


def encoding_process(
    logger_verbosity,
    num_proc,
    use_cpu,
    file_ids,
    file_paths,
    path_prompt_template,
    rule_column,
    input_column,
    output_path,
    sentence_encoder,
    batch_size,
    huggingface_cache,
    embeddings_aggr,
    normalize_embeddings,
):
    # 1.1. Create an experiment logger
    logger = get_logger(log_level=logger_verbosity)
    logger.info(format_with_hashes(text="Beginning of the encoding process"))
    ######################################################################## PRE-PROCESSING
    logger.info("PHASE I - PRE-PROCESSING")
    # 1.2. Load the datasets
    logger.debug("Loading the datasets...")
    dfs = {
        file_id: load_dataframe(file_paths=path)
        for file_id, path in zip(file_ids, file_paths)
    }
    # 1.3. Read the prompt template file
    prompt_template = read_json(path_json=path_prompt_template)
    # 1.4. Format columns with the prompt template
    logger.debug("Formatting the input columns with the chosen prompt template...")
    tuples_column_template = [
        (rule_column, "rules"),
        (input_column, "sentences"),
    ]
    # 1.5 Apply preprocessing for all datasets
    for key in dfs:
        for column, template_for in tuples_column_template:
            dfs[key] = apply_template(
                dfs[key],
                column=column,
                template_for=template_for,
                prompt_template=prompt_template,
            )
    logger.debug("Obtaining ids for unique elements in ['rules', 'sentences']...")
    # 1.6. Now we want to:
    # - Save the processed dataframe to later train of the rule constrainer
    # - Minimize the number of calls of the encoder. Since the process is deterministic, we will encode only unique elements.
    #   a) Remove the original non-template columns
    dfs = {key: dfs[key].drop([rule_column, input_column], axis=1) for key in dfs}
    #   b) Create new columns, identifying the unique rules and the unique sentences
    #   N.b. Ratio here is to obtain the embeddings of unique elements only once
    #        (for later, we will use a dictionary unique_value-embedding)
    for key in dfs:
        for _, column in tuples_column_template:
            dfs[key][f"hash_templated_{column}"] = create_id_x_column(
                df=dfs[key], column=column
            )
    # 1.7. Save the preprocessed dataframes for later
    logger.debug("Saving pre-processed dataframes for later...")
    output_dir = concatenate_paths(
        list_subpaths=[output_path, "intermidiate_datasets", "preprocessed_data"]
    )
    for key in dfs:
        dfs[key].to_parquet(f"{output_dir}/{key}.parquet", index=False)

    ######################################################################## EMBEDDING EXTRACTION
    logger.info("PHASE II - EMBEDDING EXTRACTION")
    logger.debug("Loading the tokenizer and model...")
    # 2.1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=sentence_encoder, cache_dir=huggingface_cache
    )
    # 2.2. Load the encoder
    encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path=sentence_encoder, cache_dir=huggingface_cache
    )
    # 2.3. Move the model to the GPU if available
    device = torch.device("cuda:0" if not use_cpu else "cpu")
    encoder.to(device)
    logger.debug("Starting the encoding loop over the meaningful columns")
    # 2.4 Create the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # 2.5. For-loop cicle iterating over the columns to encode
    for _, column in tuples_column_template:
        logger.info("Encoding the column:\t%s", column)
        # 2.5.1. Isolate the unique elements for that column + their indices
        columns_2_keep = [column, f"hash_templated_{column}"]
        sub_dfs = {key: dfs[key][columns_2_keep].drop_duplicates() for key in dfs}
        # 2.5.2. Convert the dataframes into huggingface datasets
        ds = convert_dataframes(
            pandas_dfs=list(sub_dfs.values()), keys=list(sub_dfs.keys())
        )
        logger.debug("Tokenizing the dataset...")
        # 2.5.3 Tokenize the data
        tokenized_ds = ds.map(
            function=tokenizing_function,
            batched=True,
            remove_columns=columns_2_keep,
            fn_kwargs={"tokenizer": tokenizer, "column": column},
            num_proc=num_proc,
        )
        # 2.5.4 Create the DataLoader
        dataloaders = {
            key: DataLoader(
                tokenized_ds[key], collate_fn=data_collator, batch_size=batch_size
            )
            for key in tokenized_ds.keys()
        }
        logger.debug("Start the encoding loop...")
        for partition, dataloader in dataloaders.items():
            # 2.5.4.1 Retrieve the indexes
            indexes = sub_dfs[partition][f"hash_templated_{column}"].tolist()
            # 2.5.4.2 Generate the embeddings
            embeddings = encoding_loop(
                dataloader=dataloader,
                encoder=encoder,
                device=device,
                index=f"hash_templated_{column}",
                sent_emb=embeddings_aggr,
                normalize=normalize_embeddings,
                progress_bar=True,
            )
            # 2.5.4.2 Create a mapping from indexes to embeddings
            index_2_embedding = {
                index: embedding for index, embedding in zip(indexes, embeddings)
            }
            # 2.5.4.2 Saving the generated dictionaries
            logger.debug(
                "Saving the generated embeddings for the partition %s...", partition
            )
            output_dir = concatenate_paths(
                list_subpaths=[
                    output_path,
                    "intermidiate_datasets",
                    "embeddings",
                    partition,
                ]
            )
            save_json(
                output_path=f"{output_dir}/{column}.json",
                dict_2_save=index_2_embedding,
            )

    logger.info(format_with_hashes(text="End of the encoding process!"))


if __name__ == "__main__":
    opts = get_encoding_options()
    encoding_process(
        logger_verbosity=opts["logger_verbosity"],
        num_proc=opts["num_proc"],
        use_cpu=opts["use_cpu"],
        file_ids=opts["file_ids"],
        file_paths=opts["file_paths"],
        path_prompt_template=opts["prompt_template"],
        rule_column=opts["rule_column"],
        input_column=opts["input_column"],
        output_path=opts["output_path"],
        sentence_encoder=opts["sentence_encoder"],
        batch_size=opts["batch_size"],
        huggingface_cache=opts["huggingface_cache"],
        embeddings_aggr=opts["embeddings_aggr"],
        normalize_embeddings=opts["normalize_embeddings"],
    )
