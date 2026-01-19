from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import Optional
from core.common.utils import move_2_cpu
from core.contrastive_encoder.classes.embedding_cacher import EmbeddingCache


def mean_pooling(model_output, attention_mask):
    """Performs mean pooling on the token embeddings output by a model, averaging
    over tokens for each input while considering the attention mask.
    From https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    Args:
        model_output (torch.Tensor): The output from a transformer model, where the
            first element contains the embeddings of each token in the sequence.
        attention_mask (torch.Tensor): A binary attention mask tensor indicating
            which tokens are valid (1) and which are padding (0) in each sequence.
    Returns:
        torch.Tensor: A tensor representing the mean-pooled embeddings for each
            sequence in the batch. Each embedding is an average of the non-padded
            token embeddings, weighted by the attention mask.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def tokenizing_function(sample, tokenizer, column):
    """
    Tokenizes a specified column from a sample dictionary using a pre-defined tokenizer.
    Args:
        sample (dict): A dictionary representing a data sample, where `sample[column]`
            contains the text data to be tokenized.
        column (str): The key in the `sample` dictionary for the text data to tokenize.
    Returns:
        dict: A dictionary with tokenized outputs, including input IDs, attention mask,
            and any other tokenizer outputs based on the tokenizer configuration.
    """
    return tokenizer(sample[column], truncation=True)


def extract_embedding(
    encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Optional[EmbeddingCache] = None,
    aggregation: str = "cls",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Extracts embeddings from a transformer encoder with optional caching.

    Args:
        encoder (torch.nn.Module): Pre-trained transformer model
        input_ids (torch.Tensor): Input token IDs tensor
        attention_mask (torch.Tensor): Attention mask tensor
        cache (EmbeddingCache, optional): Cache instance for storing embeddings
        aggregation (str, optional): Aggregation method ("cls" or "mean_pooling")
        normalize (bool, optional): Whether to L2-normalize embeddings

    Returns:
        torch.Tensor: Extracted embeddings
    """
    if cache is not None:
        # Compute hash of input tensors
        cache_key = cache.compute_hash(input_ids, attention_mask)

        # Check if embedding exists in cache
        cached_embedding = cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding.to(input_ids.device)

    with torch.no_grad():
        output = encoder(input_ids, attention_mask)
    if aggregation == "mean_pooling":
        embeddings = mean_pooling(output, attention_mask)
    elif aggregation == "cls":
        embeddings = output[0][:, 0, :]
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    # Store in cache if cache is provided
    if cache is not None:
        cache.put(cache_key, embeddings.cpu())  # Store on CPU to save GPU memory
    return embeddings


def batch_embedding_extraction(
    encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Optional[EmbeddingCache] = None,
    aggregation: str = "cls",
    normalize: bool = True,
    batch_size: int = 128,
    progress_bar: bool = True,
) -> torch.Tensor:
    """
    Extracts embeddings in batches with optional caching.
    Args:
        encoder (torch.nn.Module): Pre-trained transformer model
        input_ids (torch.Tensor): Input token IDs tensor
        attention_mask (torch.Tensor): Attention mask tensor
        cache (EmbeddingCache, optional): Cache instance for storing embeddings
        aggregation (str, optional): Aggregation method
        normalize (bool, optional): Whether to L2-normalize embeddings
        batch_size (int, optional): Batch size for processing
        progress_bar (bool, optional): Whether to show progress bar

    Returns:
        torch.Tensor: Extracted embeddings
    """
    n_batches, n_examples = input_ids.shape[:2]
    flatten_input_ids = torch.flatten(input_ids, start_dim=0, end_dim=1)
    flatten_attention_mask = torch.flatten(attention_mask, start_dim=0, end_dim=1)
    flatten_dim = flatten_input_ids.shape[0]

    internal_batch_size = min(32, batch_size)

    flatten_embeddings = []
    range_to_iterate = range(0, flatten_dim, internal_batch_size)
    pbar = (
        tqdm(range_to_iterate, desc="Embedding with encoder...")
        if progress_bar
        else range_to_iterate
    )

    for i in pbar:
        with torch.amp.autocast("cuda"):
            input_ids_batch = flatten_input_ids[i : i + internal_batch_size].long()
            attention_mask_batch = flatten_attention_mask[
                i : i + internal_batch_size
            ].long()

            if i % (internal_batch_size * 8) == 0:
                torch.cuda.empty_cache()

            embeddings_batch = extract_embedding(
                encoder=encoder,
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                cache=cache,
                aggregation=aggregation,
                normalize=normalize,
            )
            flatten_embeddings.append(embeddings_batch.cpu())

    flatten_embeddings = torch.cat(flatten_embeddings)
    embeddings = flatten_embeddings.reshape(n_batches, n_examples, -1)

    return embeddings.to(input_ids.device)


def encoding_loop(
    dataloader,
    encoder,
    device,
    index=None,
    sent_emb="cls",
    normalize=True,
    progress_bar=True,
):
    """
    Processes batches of data from a dataloader to extract sentence embeddings
    using a specified encoder model. Optionally normalizes and indexes embeddings.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader providing batches of
            input data, where each batch is a dictionary with "input_ids" and
            "attention_mask" tensors.
        encoder (torch.nn.Module): The encoder model used to compute embeddings,
            typically a transformer model.
        device (torch.device): The device (CPU or GPU) on which the computation will run.
        index (str, optional): The key for the index values in each batch, used to
            keep track of the sample order. Defaults to None.
        sent_emb (str, optional): Method for sentence embedding extraction. Options are:
            "cls" for using the CLS token, or "mean_pooling" for averaging token embeddings.
            Defaults to "cls".
        normalize (bool, optional): Whether to normalize embeddings to unit norm. Defaults to True.
        progress_bar (bool, optional): Whether to display a progress bar during processing.
            Defaults to True.

    Returns:
        torch.Tensor: tensor_embeddings: The tensor of embeddings for all data points, possibly normalized.
    """
    encoder.eval()  # Put the encoder into validation mode
    list_embeddings = []
    progress_bar = tqdm(dataloader) if progress_bar else None
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        embeddings = extract_embedding(
            encoder=encoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            aggregation=sent_emb,
            normalize=normalize,
        )
        list_embeddings.append(move_2_cpu(embeddings))
        if progress_bar:
            progress_bar.update(1)
    tensor_embeddings = torch.cat(list_embeddings).tolist()
    return tensor_embeddings
