import os
import random
from core.contrastive_encoder.classes.dataset import RuleConstrainerDataset
from core.contrastive_encoder.functions.dataset_creation import (
    OptimizedBatchProcessor,
    get_embeddings,
)
from core.contrastive_encoder.functions.prepare_data import (
    extract_positive_samples_2_macro_rules,
)
from core.data_extraction.utils import deterministic_hash


class InferenceConstrainerDataset(RuleConstrainerDataset):
    def __init__(
        self,
        logger,
        paths_data,
        macro_rule_id,
        pos_sample_column,
        pretrained_model=None,
        rule_column=None,
        input_column=None,
        path_prompt_template=None,
        huggingface_cache=None,
        paths_embeddings=None,
        partition="inference",
        cache_dir=None,
        subsample=False,
        seed=29,
        prob_neg_sample=0.3,
    ):
        valid_condition = not (
            paths_embeddings is None and (rule_column is None or input_column is None)
        )
        assert (
            valid_condition
        ), "Error: if `path_embeddings` is None, you shall specify `rule_column` AND `input_column`."
        self.cache_dir = os.path.join(cache_dir, partition) if cache_dir else None
        if self.cache_dir:
            # Make sure that the new path exists
            os.makedirs(self.cache_dir, exist_ok=True)
        super(InferenceConstrainerDataset, self).__init__(
            paths_data=paths_data,
            macro_rule_id=macro_rule_id,
            subsample=subsample,
            seed=seed,
        )
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
        # 3. Initialize an OptimizerBatchProcessor > we will use it while creating the batches
        self.batch_processor = OptimizedBatchProcessor(
            unique_rules_sample=self.unique_macro_rules
        )
        # 4. Preprocess embeddings once, to speed up the operations
        self.batch_processor.preprocess_embeddings(self.embeddings, is_token=is_token)
        # 5. Create dictionaries macro-rule 2 something for efficient indexing
        grouped_by_macro_rule = self.df.groupby(macro_rule_id)
        # 6. Each macro rule is assigned one sets of hashes, mapping maps to the corresponding single rules
        self.macro_rule_2_hash_rules = (
            grouped_by_macro_rule["hash_templated_rules"].apply(set).to_dict()
        )
        # 7. Dictionary macro_rule_id > hash_sentence (1 to 1)
        self.macro_rule_2_hash_sentence = (
            grouped_by_macro_rule["hash_templated_sentences"].first().to_dict()
        )
        # 8. Remember: as we can solve the problem both 1 rule at the time or with a single shot,
        #   we must extract the list of single rules hashes given a macro rule
        self.macro_rule_to_single_hashes = grouped_by_macro_rule[pos_sample_column].agg(
            lambda x: set(id for sample in x for id in sample.split("-"))
        )
        # 9. Also save the probability of extracting a negative sample while retrieving the rules
        self.prob_neg_sample = prob_neg_sample
        self.seed = seed
        # 10. Eventually, extract the mapping from positive samples to associated macro rules
        self.pos_sample_to_macro_rules, _ = extract_positive_samples_2_macro_rules(
            self.df,
            pos_col=pos_sample_column,
            macro_rule_col=macro_rule_id,
            cache_dir=self.cache_dir,
            logger=logger,
        )
        self.macro_rule_2_positive_samples = (
            grouped_by_macro_rule[pos_sample_column].first().to_dict()
        )

    def get_rules_and_sentence(self):
        """Generates testing samples by pairing macro rules with sentences, including both positive and negative examples.
        This method creates testing data by:
        1. Randomly selecting macro rules from the unique set
        2. For each macro rule, deciding whether to create a positive or negative sample based on prob_neg_sample
        3. For negative samples, selecting an unrelated rule hash from the available set
        4. Collecting corresponding embeddings for both rules and sentences
        Returns:
            tuple: Contains six elements:
                - macro_rule_hashes (list): Hashes of the selected macro rules
                - rules_embeddings (list): Embeddings for each selected rule
                - sentences_embeddings (list): Embeddings for the paired sentences
                - labels (list): Boolean indicators for whether each pair is compliant
                - micro_rules_hashes (list): Associated micro rule hashes for each macro rule
                - sentence_hashes (list): Hashes of the selected sentences
        The method ensures at least one positive prediction per macro rule to maintain balanced training data.
        Note:
            Uses a random number generator seeded with self.seed for reproducible sampling.
            The probability of generating negative samples is controlled by self.prob_neg_sample.
            Performance optimization: Uses pre-cached lookups and reduced dictionary access for better performance.
        """
        # Initialize random number generator for reproducible sampling
        rng = random.Random(self.seed)
        # Cache frequently accessed embeddings
        embeddings_cache = self.batch_processor.embedding_cache
        rules_embed = embeddings_cache["rules"]
        sentences_embed = embeddings_cache["sentences"]
        # Initialize output lists for collecting samples
        min_positive_predictions = len(self.unique_macro_rules)
        rule_hashes = []
        micro_rules_hashes = []
        sentence_hashes = []
        rules_embeddings = []
        sentences_embeddings = []
        labels = []
        # Prepare the list of unique macro rules and create a set for negative sampling
        unique_macro_rules_list = list(self.unique_macro_rules)

        rng.shuffle(unique_macro_rules_list)
        set_macro_rules = set(self.unique_macro_rules)
        # Pre-cache rule data for faster access during sampling
        macro_rule_lookup = {
            rule_id: {
                "pos_hash": self.macro_rule_2_positive_samples[rule_id],
                "hash_rules": self.macro_rule_2_hash_rules[rule_id],
                "micro_rules": list(self.macro_rule_to_single_hashes[rule_id]),
                "sentence_hash": self.macro_rule_2_hash_sentence[rule_id],
            }
            for rule_id in unique_macro_rules_list
        }
        # Pre-calculate positive sample mappings for each rule
        pos_sample_mappings = {
            rule_id: self.pos_sample_to_macro_rules[
                macro_rule_lookup[rule_id]["pos_hash"]
            ]
            for rule_id in unique_macro_rules_list
        }
        # Pre-calculate available negative samples for each rule
        negative_samples = {
            rule_id: sorted(set_macro_rules - pos_sample_mappings[rule_id])
            for rule_id in unique_macro_rules_list
        }
        # Generate samples ensuring minimum positive predictions requirement is met
        positives_found = 0
        idx = 0
        while positives_found < min_positive_predictions:
            # Select current macro rule, cycling through the list multiple times if needed
            macro_rule_id = unique_macro_rules_list[idx % len(unique_macro_rules_list)]
            # Determine if this should be a negative sample
            is_negative = rng.random() <= self.prob_neg_sample
            if is_negative:
                # Select a rule from pre-calculated negative samples
                current_macro_rule_id = rng.choice(negative_samples[macro_rule_id])
                label_is_compliant = False
            else:
                # Use current rule for positive sample
                current_macro_rule_id = macro_rule_id
                label_is_compliant = True
                positives_found += 1
            # Get cached rule data for current selection
            rule_data = macro_rule_lookup[current_macro_rule_id]
            # Collect all required data for the current sample
            rule_hashes.append(rule_data["pos_hash"])
            rules_embeddings.append(
                [rules_embed[str(hash_rule)] for hash_rule in rule_data["hash_rules"]]
            )
            micro_rules_hashes.append(rule_data["micro_rules"])
            sentence_hashes.append(
                str(macro_rule_lookup[macro_rule_id]["sentence_hash"])
            )
            sentences_embeddings.append(
                sentences_embed[str(macro_rule_lookup[macro_rule_id]["sentence_hash"])]
            )
            labels.append(label_is_compliant)
            idx += 1
        return (
            rule_hashes,
            rules_embeddings,
            sentences_embeddings,
            labels,
            micro_rules_hashes,
            sentence_hashes,
        )

    def gather_real_rule(self, hash_rules):
        real_rules = list(
            self.df[self.df["positive_sample_id"] == hash_rules].rules.unique()
        )
        template = real_rules[0].split(": ")[0] + ": "
        words = [real_rule.split(": ")[1] for real_rule in real_rules]
        rules = template + f"[{', '.join(words)}]"
        return rules

    def gather_real_sentences(self, hash_sentence):
        real_sentence = self.df[
            self.df["hash_templated_sentences"] == hash_sentence
        ].sentences.tolist()[0]
        return real_sentence
