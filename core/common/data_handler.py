from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


def select_subset(df, n_subsample, seed):
    """Randomly selects a subset of rows from a DataFrame.
    Args:
        df (pandas.DataFrame): Input DataFrame
        n_subsample (int): Number of rows to select
        seed (int): Random seed for reproducible sampling
    Returns:
        pandas.DataFrame: A new DataFrame containing the randomly selected subset of rows
    Example:
        >>> df = pd.DataFrame({
        ...     'id': range(100),
        ...     'value': range(100)
        ... })
        >>> subset_df = select_subset(df, n_subsample=10, seed=42)
    Note:
        The function first shuffles the entire DataFrame and then selects the first
        n_subsample rows. It returns a copy of the selected rows to avoid modifying
        the original DataFrame.
    """
    df = df.sample(frac=1, random_state=seed)
    return df.iloc[:n_subsample].copy()


def pad_ragged_tensor_lists(
    x: List[List[torch.Tensor]],
    pad_value: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: List[List[Tensor]] where all tensors share the same shape (and dtype).
       Only the inner-list length varies.

    Returns:
      out: Tensor [B, Lmax, *Tshape]
      padding_mask: Bool [B, Lmax], True for padded positions, False for real positions
    """
    B = len(x)
    Lmax = max((len(row) for row in x), default=0)

    # Handle empty cases
    if B == 0:
        out = torch.empty((0, 0), device=device)
        padding_mask = torch.empty((0, 0), dtype=torch.bool, device=device)
        return out, padding_mask

    # Find a reference tensor to infer shape/dtype/device
    ref = None
    for row in x:
        if len(row) > 0:
            ref = row[0]
            break

    if ref is None:
        # All rows empty
        out = torch.empty((B, Lmax), device=device)
        padding_mask = torch.zeros((B, Lmax), dtype=torch.bool, device=device)
        return out, padding_mask

    if device is None:
        device = ref.device
    dtype = ref.dtype
    tshape = tuple(ref.shape)
    with torch.no_grad():
        out = torch.full((B, Lmax, *tshape), pad_value, dtype=dtype, device=device)
        padding_mask = torch.zeros(
            (B, Lmax), dtype=torch.bool, device=device
        )  # False = padded

        for b, row in enumerate(x):
            n = len(row)
            if n == 0:
                continue
            stacked = torch.stack(
                [t.to(device=device) for t in row], dim=0
            )  # [n, *Tshape]
            out[b, :n].copy_(stacked)
            padding_mask[b, :n] = True
    # Make absolutely explicit:
    out.requires_grad_(False)
    return out, padding_mask
