import warnings
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm, trange
from transformers import AutoModel
from core.contrastive_encoder.classes.embedding_cacher import EmbeddingCache
from core.contrastive_encoder.functions.metrics import (
    compute_batch_gini_sparsity,
    compute_mri,
)
from core.llm_encoder.encoder_func import batch_embedding_extraction, mean_pooling

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RulesIntersection(nn.Module):

    def __init__(self, h_in, h_out):
        super(RulesIntersection, self).__init__()
        self.value_transform = nn.Linear(h_in, h_out)
        # Attention layers
        self.attention_transform = nn.Linear(h_in, h_out)
        # Learning the attention mechanism receiving as input the output of layer1
        self.layer2 = nn.Linear(h_out, h_out)
        nn.init.xavier_uniform_(self.value_transform.weight)
        nn.init.xavier_uniform_(self.attention_transform.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.value_transform.bias)
        nn.init.zeros_(self.attention_transform.bias)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, embeddings, padding):
        # embeddings: (batch_size, rules, hidden)
        # padding: (batch_size, rules) - reshape padding to match rules dimension
        # Get the updated embeddings in the new space:
        value_features = F.relu(self.value_transform(embeddings))
        # Now, let's take care about the attention scores
        attention_features = F.relu(
            self.attention_transform(embeddings)
        )  # (batch, rules, h_out)
        # Apply layer 2 to get attention logits
        attention_logits = self.layer2(
            attention_features
        )  # (batch_size, rules, hidden)
        # Mask padded positions in attention_logits AFTER layer2 but BEFORE softmax
        padding_mask = padding.unsqueeze(-1)  # (batch_size, rules, 1)
        attention_logits = attention_logits.masked_fill(
            padding_mask == 0, float("-inf")
        )
        # Now apply softmax
        attention = F.softmax(attention_logits, dim=1)  # (batch_size, rules, hidden)
        # Weighted sum of embeddings using the attention scores + residual connections
        updated_embedding = torch.sum(attention * value_features, dim=1)
        return updated_embedding


class SentenceProjector(nn.Module):
    def __init__(self, h_in, h_out):
        super(SentenceProjector, self).__init__()
        self.sentece_projector = nn.Linear(h_in, h_out)
        nn.init.xavier_uniform_(self.sentece_projector.weight)

    def forward(self, embeddings):
        return F.relu(self.sentece_projector(embeddings))


class NaiveConstrainerModel(nn.Module):
    def __init__(
        self,
        pretrained_model="sentence-transformers/all-mpnet-base-v2",
        huggingface_cache=None,
        aggregation="mean_pooling",
        normalize=False,
        cache_dir=None,
    ):
        super(NaiveConstrainerModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            cache_dir=huggingface_cache,
        )
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False  # Ensure encoder is frozen
        self.embedding_cache = EmbeddingCache(cache_dir=cache_dir)
        self.aggregation = aggregation
        self.normalize = normalize

    def encode_rules(self, rule_embeddings, padding_mask_rule):
        with torch.no_grad():  # Ensure no gradients are computed for encoder
            # Process the rules (return tensor BxRxH)
            rules_embeddings = batch_embedding_extraction(
                encoder=self.encoder,
                input_ids=rule_embeddings[:, :, 0, :],
                attention_mask=rule_embeddings[:, :, 1, :],
                cache=self.embedding_cache,
                normalize=self.normalize,
                aggregation=self.aggregation,
                batch_size=rule_embeddings.shape[0],
                progress_bar=False,
            )
        return mean_pooling((rules_embeddings,), padding_mask_rule)

    def encode_sentences(self, input_embedding):
        with torch.no_grad():  # Ensure no gradients are computed for encoder
            # Process the input embeddings (return tensor BxNxH)
            input_embedding = batch_embedding_extraction(
                encoder=self.encoder,
                input_ids=input_embedding[:, :, 0, :],
                attention_mask=input_embedding[:, :, 1, :],
                cache=self.embedding_cache,
                normalize=self.normalize,
                aggregation=self.aggregation,
                batch_size=input_embedding.shape[0],
                progress_bar=False,
            )
        return input_embedding

    def save_safetensors(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        save_file(self.state_dict(), os.path.join(out_dir, "model.safetensors"))

    def load_best(self, path_best_model):
        state_dict = load_file(f"{path_best_model}/model.safetensors")
        # strict=False ignores missing keys for the encoder
        self.load_state_dict(state_dict, strict=False)


class ContrastiveConstrainerModel(nn.Module):
    def __init__(
        self,
        h_out=768,
        h_in=768,
        require_encoder=True,
        pretrained_model=None,
        huggingface_cache=None,
        aggregation="mean_pooling",
        normalize=False,
        temperature=0.1,
        cache_dir=None,
        loss_rules=False,
    ):
        super(ContrastiveConstrainerModel, self).__init__()
        self.sentence_net = SentenceProjector(h_in=h_in, h_out=h_out)
        self.rules_net = RulesIntersection(h_in=h_in, h_out=h_out)
        self.logsigmoid = nn.LogSigmoid()
        if require_encoder:
            self.encoder = AutoModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model,
                cache_dir=huggingface_cache,
            )
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False  # Ensure encoder is frozen
            self.embedding_cache = EmbeddingCache(cache_dir=cache_dir)
        self.require_encoder = require_encoder
        self.aggregation = aggregation
        self.normalize = normalize
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.loss_rules = loss_rules

    def load_best(self, path_best_model):
        state_dict = load_file(f"{path_best_model}/model.safetensors")
        # strict=False ignores missing keys for the encoder
        self.load_state_dict(state_dict, strict=False)

    def encode_rules(self, rule_embeddings, padding_mask_rule):
        if self.require_encoder:
            with torch.no_grad():  # Ensure no gradients are computed for encoder
                # Process the rules (return tensor BxRxH)
                rules_embeddings = batch_embedding_extraction(
                    encoder=self.encoder,
                    input_ids=rule_embeddings[:, :, 0, :],
                    attention_mask=rule_embeddings[:, :, 1, :],
                    cache=self.embedding_cache,
                    normalize=self.normalize,
                    aggregation=self.aggregation,
                    batch_size=rule_embeddings.shape[0],
                    progress_bar=False,
                )
        else:
            rules_embeddings = rule_embeddings[:, :, 0, :]
        # 2. Obtain the intersection of rules (batch_size, hidden_dimension)
        embedding_rules_inter = self.rules_net(
            embeddings=rules_embeddings, padding=padding_mask_rule
        )
        return embedding_rules_inter

    def encode_sentences(self, input_embedding):
        if self.require_encoder:
            with torch.no_grad():  # Ensure no gradients are computed for encoder
                # Process the input embeddings (return tensor BxNxH)
                input_embedding = batch_embedding_extraction(
                    encoder=self.encoder,
                    input_ids=input_embedding[:, :, 0, :],
                    attention_mask=input_embedding[:, :, 1, :],
                    cache=self.embedding_cache,
                    normalize=self.normalize,
                    aggregation=self.aggregation,
                    batch_size=input_embedding.shape[0],
                    progress_bar=False,
                )
        else:
            input_embedding = input_embedding[:, :, 0, :]
        embeddings_sentences = self.sentence_net(embeddings=input_embedding)
        return embeddings_sentences

    def encoder_rules_and_sentences(
        self,
        input_embedding,
        rule_embeddings,
        padding_mask_rule,
    ):
        embeddings_sentences = self.encode_sentences(input_embedding=input_embedding)
        embedding_rules_inter = self.encode_rules(
            rule_embeddings=rule_embeddings, padding_mask_rule=padding_mask_rule
        )
        return embeddings_sentences, embedding_rules_inter

    def forward(
        self,
        input_embedding,
        padding_input_embedding,
        rule_embeddings,
        padding_mask_rule,
        labels=None,
        rules_vs_rules_labels=None,
    ):
        embeddings_sentences, embedding_rules_inter = self.encoder_rules_and_sentences(
            input_embedding, rule_embeddings, padding_mask_rule
        )
        if labels is not None:
            # Compute InfoNCE loss
            infonce_loss = self.compute_infonce_loss(
                sentence_embeddings=embeddings_sentences,
                rule_embeddings=embedding_rules_inter,
                labels=labels,
                padding_mask=padding_input_embedding,
            )
            output = {"sentence_loss": infonce_loss}
            if (
                infonce_loss is None
                or torch.isnan(infonce_loss)
                or torch.isinf(infonce_loss)
            ):
                warnings.warn(f"Invalid loss value detected: {infonce_loss}")
        else:
            output = {"sentence_loss": torch.tensor([0], device=input_embedding.device)}
        if rules_vs_rules_labels is not None and self.loss_rules:
            # Also compute loss for rules vs rules
            rule_loss = self.compute_infonce_loss(
                sentence_embeddings=embedding_rules_inter,
                rule_embeddings=embedding_rules_inter,
                labels=rules_vs_rules_labels,
            )
            output["rule_loss"] = rule_loss
        else:
            output["rule_loss"] = torch.tensor([0], device=input_embedding.device)
        output["loss"] = output["sentence_loss"] + output["rule_loss"]
        if not self.training:
            # Add debugging info
            with torch.no_grad():
                # Compute sparsity metrics for the positive samples and rules for the sake of debugging
                sentence_sparsity = compute_batch_gini_sparsity(
                    embeddings_sentences[:, 0]
                )
                rules_sparsity = compute_batch_gini_sparsity(embedding_rules_inter)
                # Compute similarities between all sentences and rules
                sims = self.compute_similarity(
                    embeddings_sentences, embedding_rules_inter, padding_input_embedding
                )  # Shape: [batch_size, batch_size * num_sentences]
                # Now these will correctly identify positive and negative pairs
                positive_sentence_to_rule_sims = sims[labels.bool()]
                negative_sentence_to_rule_sims = sims[
                    ~labels.bool() & ~torch.isinf(sims)
                ]  # Exclude padded value
                # Now, also compute the distances between rules
                if rules_vs_rules_labels is not None:
                    sims = self.compute_similarity(
                        embedding_rules_inter, embedding_rules_inter, padding_mask=None
                    )
                    positive_rule_to_rule_sims = sims[rules_vs_rules_labels.bool()]
                    negative_rule_to_rule_sims = sims[
                        ~rules_vs_rules_labels.bool() & ~torch.isinf(sims)
                    ]  # Exclude padded value
                else:
                    positive_rule_to_rule_sims = torch.tensor([])
                    negative_rule_to_rule_sims = torch.tensor([])
                metrics = {
                    # Norm metrics
                    "sentence_sparsity": sentence_sparsity,
                    "rules_sparsity": rules_sparsity,
                    # Positive/negative metrics
                    "pos_sent_2_rule_sim_mean": positive_sentence_to_rule_sims.mean(),
                    "neg_sent_2_rule_sim_mean": negative_sentence_to_rule_sims.mean(),
                    "positive_rule_to_rule_sims": positive_rule_to_rule_sims.mean(),
                    "negative_rule_to_rule_sims": negative_rule_to_rule_sims.mean(),
                }
                output["predictions"] = metrics
        return output

    @staticmethod
    def compute_tensor_similarity(tensor1, tensor2, eps=1e-12):
        """
        Compute cosine similarity between two tensors with improved numerical stability.

        Args:
            tensor1 (torch.Tensor): First tensor of shape [..., hidden_dim]
            tensor2 (torch.Tensor): Second tensor of shape [..., hidden_dim]
            eps (float): Small value for numerical stability in normalization

        Returns:
            torch.Tensor: Cosine similarity matrix
        """
        # Reshape tensor1 if needed to be 2D [N, hidden_dim]
        original_shape = tensor1.shape
        if len(original_shape) > 2:
            tensor1 = tensor1.reshape(-1, original_shape[-1])

        # Normalize embeddings (using stable norm)
        normalized_tensor1 = F.normalize(tensor1, p=2, dim=-1)
        normalized_tensor2 = F.normalize(tensor2, p=2, dim=-1)

        # Compute cosine similarity
        cosine_sim = torch.matmul(normalized_tensor1, normalized_tensor2.T)

        return cosine_sim

    def compute_similarity(
        self, sentence_embeddings, rule_embeddings, padding_mask=None, mask_value=-1e9
    ):
        if sentence_embeddings.dim() == 3:
            _, _, hidden_dim = sentence_embeddings.shape
            # Reshape sentence embeddings to [B*N, H]
            flat_sentence_embeddings = sentence_embeddings.reshape(-1, hidden_dim)
        elif sentence_embeddings.dim() == 2:
            flat_sentence_embeddings = sentence_embeddings
        else:
            raise ValueError(
                f"Invalid sentence embeddings shape: {sentence_embeddings.shape}"
            )
        # Compute cosine similarities using the static method
        cosine_sim = self.compute_tensor_similarity(
            flat_sentence_embeddings, rule_embeddings
        )
        masked_similarities = cosine_sim.clone()
        if padding_mask is not None:
            # Reshape padding mask and apply masking
            padding_mask = padding_mask.reshape(-1).bool()  # [B*N]
            masked_similarities[~padding_mask] = mask_value
        # Reshape to [B, B*N] for loss computation
        masked_similarities = masked_similarities.T
        return masked_similarities

    def compute_infonce_loss(
        self, sentence_embeddings, rule_embeddings, labels, padding_mask=None
    ):
        """
        Compute InfoNCE loss for contrastive learning.

        Args:
            sentence_embeddings: Tensor of shape [batch_size, num_sentences, hidden_dim]
            rule_embeddings: Tensor of shape [batch_size, hidden_dim]
            labels: Binary tensor indicating positive pairs [batch_size, batch_size * num_sentences]
            padding_mask: Mask for valid sentences [batch_size, num_sentences]

        Returns:
            Tensor: InfoNCE loss value
        """
        # Compute similarities between all sentences and rules
        similarities = self.compute_similarity(
            sentence_embeddings, rule_embeddings, padding_mask
        )  # Shape: [batch_size, batch_size * num_sentences]
        # Scale similarities by temperature
        scaled_similarities = similarities / self.temperature
        # For numerical stability, subtract max value before exponential
        max_similarities = torch.max(scaled_similarities, dim=1, keepdim=True)[0]
        exp_similarities = torch.exp(scaled_similarities - max_similarities)
        # Mask out invalid pairs (-inf similarities)
        valid_mask = ~torch.isinf(similarities)
        positive_mask = labels.bool()
        numerator = torch.sum(exp_similarities * positive_mask, dim=1)
        # Denominator includes all valid pairs (both positive and negative)
        denominator = torch.sum(exp_similarities * valid_mask, dim=1)
        # Calcolate the log ratio
        eps = 1e-12
        log_ratio = torch.log(numerator / (denominator + eps) + eps)
        return -log_ratio.mean()

    def extract_MRI_metrics(
        self, dataloader, show_progress_bar, desc, device, top_m_values
    ):
        metrics = {}
        iterator = tqdm(dataloader, desc=desc) if show_progress_bar else dataloader
        with torch.no_grad():
            for batch in iterator:
                batch = {
                    k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
                }
                # Get embeddings for entire batch at once
                embeddings_sentences, embedding_rules = (
                    self.encoder_rules_and_sentences(
                        batch["input_embedding"],
                        batch["rule_embeddings"],
                        batch["padding_mask_rule"],
                    )
                )
                batch_size = embedding_rules.shape[0]
                # Compute similarities one batch at the time
                # Reshape embeddings to handle batch dimension properly
                similarities_list = []
                for i in range(batch_size):
                    # [1, hidden_dim]
                    rule_emb = embedding_rules[i].unsqueeze(0)
                    # [1, num_sentences, hidden_dim]
                    sent_emb = embeddings_sentences[i].unsqueeze(0)
                    padding_input = batch["padding_input_embedding"][i].unsqueeze(0)
                    # [1, num_sentences]
                    sim = self.compute_similarity(
                        sent_emb,
                        rule_emb,
                        padding_input,
                    )
                    # Reshape to [num_sentences]
                    similarities_list.append(sim.squeeze(0))
                # Stack similarities for parallel processing ([batch_size, num_sentences])
                stacked_similarities = torch.stack(similarities_list)
                # Compute metrics for entire batch at once
                batch_metrics = compute_mri(
                    stacked_similarities, batch["labels"], top_m_values
                )
                # Update metric sums
                for metric_name, value in batch_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(value)

        metrics = {
            f"control/{metric_name}": np.mean(metric)
            for metric_name, metric in metrics.items()
        }
        return metrics
