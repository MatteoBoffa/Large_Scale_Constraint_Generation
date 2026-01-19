import math
import os
from core.contrastive_encoder.classes.dataset import RuleConstrainerDataset
from core.contrastive_encoder.functions.dataset_creation import (
    OptimizedBatchProcessor,
    get_embeddings,
)
from core.contrastive_encoder.functions.prepare_data import (
    extract_positive_samples_2_macro_rules,
)


class TrainingContrastiveConstrainerDataset(RuleConstrainerDataset):
    def __init__(
        self,
        logger,
        paths_data,
        macro_rule_id,
        pretrained_model=None,
        rule_column=None,
        input_column=None,
        path_prompt_template=None,
        huggingface_cache=None,
        paths_embeddings=None,
        pos_sample_column=None,
        seed=29,
        n_negative_samplings=128,
        partition="train",
        cache_dir=None,
        subsample=False,
        oversample_factor=4,
    ):
        assert partition in [
            "train",
            "validation",
            "control",
            "test",
        ], "Error: partition must be in [`train`, `validation`, `control`, `test`]"
        valid_condition = not (
            paths_embeddings is None and (rule_column is None or input_column is None)
        )
        assert (
            valid_condition
        ), "Error: if `path_embeddings` is None, you shall specify `rule_column` AND `input_column`."
        # Initialize the parent class
        super(TrainingContrastiveConstrainerDataset, self).__init__(
            paths_data=paths_data,
            macro_rule_id=macro_rule_id,
            seed=seed,
            subsample=subsample,
        )
        self.base_seed = seed
        self.n_negative_samplings = n_negative_samplings

        self.partition = partition
        self.cache_dir = os.path.join(cache_dir, partition) if cache_dir else None
        if self.cache_dir:
            # Make sure that the new path exists
            os.makedirs(self.cache_dir, exist_ok=True)
        # 1. Now check if we also have some cached embeddings to load
        if paths_embeddings:
            logger.info("\t\tLoading the cached embeddings...")
            embeddings = get_embeddings(paths_embeddings=paths_embeddings)
            is_token = False
        # 2. If it's not the case, we need to:
        else:
            logger.info(
                "\t\tPreprocessing the input dataset (concatenating the prompts at `%s`)",
                path_prompt_template,
            )
            # 2.a Preprocess the dataset, adding the prompts and creating hashes
            self.df, tuples_column_template = self.preprocess_dataset(
                df=self.df,
                rule_column=rule_column,
                input_column=input_column,
                path_prompt_template=path_prompt_template,
                cache_dir=self.cache_dir,
            )
            logger.info(
                "\t\tTokenizing the columns `%s` and `%s`...", rule_column, input_column
            )
            # 2.b Tokenize rules and sentences (creating input_ids and attention_masks)
            embeddings = self.extract_tokens(
                tuples_column_template=tuples_column_template,
                pretrained_model=pretrained_model,
                huggingface_cache=huggingface_cache,
                cache_dir=self.cache_dir,
            )
            is_token = True
        self.embeddings = embeddings
        # 3. Create dictionaries macro-rule 2 something for efficient indexing
        #   Remember: a macro-rule is a set of micro-rules; they share the same positive sample and sentence
        logger.info(
            "\t\tGrouping by macro rule ID to cache some dictionaries we will use later..."
        )
        grouped_by_macro_rule = self.df.groupby(macro_rule_id)
        #   3a. Dictionary macro_rule_id > pos_sample_column (1 to 1)
        self.macro_rule_2_positive_samples = (
            grouped_by_macro_rule[pos_sample_column].first().to_dict()
        )
        #   3b. Dictionary macro_rule_id > hash_sentence (1 to 1)
        self.macro_rule_2_hash_sentence = (
            grouped_by_macro_rule["hash_templated_sentences"].first().to_dict()
        )
        #   3c. Each macro rule is assigned one set of micro rules
        #       Specifically, those are hashes, mapping to the corresponding single rules
        self.macro_rule_2_hash_rules = (
            grouped_by_macro_rule["hash_templated_rules"].apply(set).to_dict()
        )
        logger.info("\t\tGet all samples associated to each rule...")
        # 3d. Also, extract all the macro-rules x pos_sample_column (1 to many)
        (
            self.pos_sample_to_macro_rules,
            self.pos_id_2_other_positives,
        ) = extract_positive_samples_2_macro_rules(
            self.df,
            pos_col=pos_sample_column,
            macro_rule_col=macro_rule_id,
            cache_dir=self.cache_dir,
            logger=logger,
        )
        # 4. Initialize an OptimizerBatchProcessor > we will use it while creating the batches
        self.batch_processor = OptimizedBatchProcessor(
            unique_rules_sample=self.unique_macro_rules,
        )
        # 5. Preprocess embeddings once, to speed up the operations
        self.batch_processor.preprocess_embeddings(self.embeddings, is_token=is_token)
        # 6. Pre-compute negative sampling matrix
        logger.info(
            "\t\tInitialize a random set of %s negative samples per each rule...",
            self.n_negative_samplings * oversample_factor,
        )
        self.batch_processor.create_negative_sampling_matrix(
            mr_2_ps=self.macro_rule_2_positive_samples,
            ps_2_mrs=self.pos_sample_to_macro_rules,
            n_negative_samplings=self.n_negative_samplings,
            seed=self.base_seed,
            cache_dir=self.cache_dir,
            logger=logger,
            oversample_factor=oversample_factor,
        )
        logger.info(
            "\t\tThe dataset for the partition {} contains {:,} unique tuples rule-sample!".format(
                partition, len(self.unique_macro_rules)
            )
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        return self.batch_processor.batch_extract_training_samples(
            idx,
            self.macro_rule_2_hash_sentence,
            self.macro_rule_2_hash_rules,
            self.macro_rule_2_positive_samples,
            self.pos_id_2_other_positives,
            self.n_negative_samplings,
            partition=self.partition,
            base_seed=self.base_seed,
        )

    def get_rules_and_positive_samples(self):
        """Extracts and organizes embeddings for rules and their corresponding positive samples.
        This method processes cached embeddings to create two mappings:
            1. Rules: Maps positive sample IDs to embeddings of their associated micro-rules
            2. Positive Samples: Maps positive sample IDs to embeddings of their associated sentences
        The method uses several class-level mappings:
        - pos_sample_to_macro_rules: Maps positive sample IDs to macro rule IDs
        - macro_rule_2_hash_rules: Maps macro rule IDs to hash rules
        - macro_rule_2_hash_sentence: Maps macro rule IDs to hash sentences
        - batch_processor.embedding_cache: Contains pre-computed embeddings for rules and sentences
        Notes:
            - A macro rule ID is a hash of (rule, sentence) pair
            - All rules with the same positive sample ID share the same macro rule ID
            - The embeddings are retrieved from a cache using string representations of hash values
        Returns:
            tuple: A pair of dictionaries (rules, positive_samples) where:
                - rules: Dict mapping positive sample IDs to lists of rule embeddings
                - positive_samples: Dict mapping positive sample IDs to lists of sentence embeddings
        """
        rules, positive_samples = {}, {}
        embeddings = self.batch_processor.embedding_cache
        # Extract the unique rules > iterating over the positive samples index
        for (
            positive_samples_id,
            macro_rule_ids,
        ) in self.pos_sample_to_macro_rules.items():
            # Remember: the macro_rule_id is just an hash of (rule, sentence)
            #   Hence, all the rules having the same positive_sample_id also have the same macro_rule_id
            #       (what changes are the associated samples)
            sample_macro_rule_id = list(macro_rule_ids)[0]
            # For a given macro rule, get all associated micro-rules
            hash_rules = self.macro_rule_2_hash_rules[sample_macro_rule_id]
            rule_embeddings = [
                embeddings["rules"][str(hash_rule)] for hash_rule in hash_rules
            ]
            rules[positive_samples_id] = rule_embeddings
            # Now, given the positive_sample_id, extract all the associated positive
            hash_positive_sentences = [
                self.macro_rule_2_hash_sentence[macro_rule_id]
                for macro_rule_id in macro_rule_ids
            ]
            sentence_embeddings = [
                embeddings["sentences"][str(hash_sentence)]
                for hash_sentence in hash_positive_sentences
            ]
            positive_samples[positive_samples_id] = sentence_embeddings
        return rules, positive_samples

    def get_avg_samples_per_rule(self):
        # Compute the total number of elements
        total_elements = sum(len(v) for v in self.pos_sample_to_macro_rules.values())
        # Compute the number of keys
        num_keys = len(self.pos_sample_to_macro_rules)
        # Compute the average
        return math.ceil(total_elements / num_keys) if num_keys > 0 else 0
