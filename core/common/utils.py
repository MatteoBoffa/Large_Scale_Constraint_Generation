import ast
import json
import pickle
import random
import os
import numpy as np
import torch
import logging
import pandas as pd
from datasets import Dataset, DatasetDict

from core.data_extraction.utils import deterministic_hash


def load_openai_key():

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return api_key


def get_logger(log_level):
    """Get logger.
    This function initializes and configures a logger.
    Args:
        log_level (str): The log level, either 'debug', 'info', or 'warning'.

    Returns:
        Logger: The configured logger object.
    """
    if log_level == "debug":
        log_level = logging.DEBUG
    elif log_level == "info":
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # Add a StreamHandler if there isn't one
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def save_json(output_path, dict_2_save):
    """Function that saves a dictionary into a local json file.
    Args:
        output_path (str): File path toward the output json file.
        dict_2_save (dict): Dictionary to save.
    """
    with open(output_path, encoding="utf-8", mode="w+") as f:
        json.dump(dict_2_save, fp=f)


def set_seeds(seed):
    """Set seeds.
    This function sets seeds for random number generators.
    Args:
        seed (int): The seed value.
    """
    # Equivalent of https://huggingface.co/docs/accelerate/v0.1.0/_modules/accelerate/utils.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataframe(file_paths):
    """Loads and concatenates dataframes from multiple file paths.
    Admitted extensions are [`csv`, `parquet`]

    Args:
        file_paths (str or list): Single file path or list of file paths to load.

    Returns:
        DataFrame: Concatenated pandas dataframe
    """
    # Handle single file path case
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    dataframes = []
    for file_path in file_paths:
        extension = os.path.splitext(file_path)[1]
        assert extension in [
            ".csv",
            ".parquet",
        ], f"Error: extension {extension} is still not supported (use `.parquet` or `.csv`)"

        if extension == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)
        dataframes.append(df)
    # Concatenate all dataframes
    return pd.concat(dataframes, ignore_index=True)


def concatenate_paths(list_subpaths):
    """Simple function that returns a complete path, given a list of subpaths.
    Notice: The list must be ordered!
    Args:
        list_subpaths (list): List of subpaths, must incremental lead to the final directory.
    Returns:
        str: The complete path.
    """
    complete_path = os.path.join(*list_subpaths)
    os.makedirs(complete_path, exist_ok=True)
    return complete_path


def create_id_x_column(df, column):
    """Assign a unique group ID for each unique value in the specified column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name based on which to group the DataFrame.
    Returns:
        pd.Series: A Series containing the unique group IDs for each row in the DataFrame.
    """

    return df[column].apply(deterministic_hash)


def save_tensor_to_pickle(tensor, path_pickle):
    """
    Saves a PyTorch tensor to a specified pickle file.

    Args:
    -----
    tensor : torch.Tensor
        The PyTorch tensor to be saved.

    path_pickle : str
        The file path where the tensor should be saved in pickle format.

    Returns:
    --------
    None

    Example:
    --------
    >>> tensor = torch.tensor([1, 2, 3])
    >>> save_tensor_to_pickle(tensor, 'tensor.pkl')

    Notes:
    ------
    - This function serializes the tensor object directly to a pickle file.
    - Ensure the file path provided has write permissions.
    """
    with open(path_pickle, "wb") as file:
        pickle.dump(tensor, file)


def read_pickle(path_pickle):
    """
    Reads and deserializes a Python object from a specified pickle file.
    Args:
    -----
    path_pickle : str
        The file path to the pickle file to be read.
    Returns:
    --------
    object
        The Python object that was serialized in the pickle file.
    Example:
    --------
    >>> data = read_pickle('example.pkl')
    >>> print(data)
    Notes:
    ------
    - This function uses `pickle.load()` to deserialize the object from the file.
    - Only use with pickle files from trusted sources, as unpickling can execute arbitrary code.
    """
    with open(path_pickle, "rb") as file:
        data = pickle.load(file)
    return data


def read_json(path_json):
    """Reads a JSON file from a specified path and returns its contents as a dictionary.
    This function verifies that the provided path points to an existing JSON file
    and then loads its contents.
    Args:
        path_json (str): The path to the JSON file.
    Returns:
        dict: The contents of the JSON file.
    Raises:
        AssertionError: If the file does not exist or the extension is not `.json`.
    Example:
        >>> data = read_json("config.json")
        >>> print(data)
    Notes:
        - This function expects the JSON file to be encoded in UTF-8.
        - An assertion error is raised if the file does not exist or if the path does not end in `.json`.
    """
    assert os.path.isfile(path=path_json), f"Error: {path_json} does not exits"
    path_extension = os.path.splitext(path_json)[1]
    assert path_extension == ".json", "Error: the path must lead to a json file."
    with open(path_json, encoding="utf-8") as f:
        json_file = json.load(f)
    return json_file


def format_with_hashes(text, n_hashes=40):
    """
    Formats a given text message with a specified number of hash characters (#) before the message.

    This function creates a string with the message preceded by a line of hashes, providing
    an easy way to emphasize or separate log messages.

    Args:
        text (str): The text message to be formatted.
        n_hashes (int, optional): The number of hash characters (#) to include
            before the message. Default is 50.

    Returns:
        str: The formatted string with hash characters and the text.

    Example:
        >>> format_with_hashes("Process started", n_hashes=30)
        '##############################\tProcess started\n'
    """
    return "#" * n_hashes + f"\t{text}"


def check_list_format(candidate_list):
    """Validates and evaluates a string representation of a Python list.

    This function attempts to convert the provided string `candidate_list`
    into an actual Python list. It ensures that the input is a valid
    representation of a list and raises an error if it is not.

    Parameters:
    candidate_list (str): A string representation of a Python list
                            (e.g., "[1, 2, 3]").

    Returns:
    list: The evaluated list if the input is valid.

    Raises:
    ValueError: If the input cannot be evaluated as a list, or if the
                evaluated object is not a list. The error message will
                indicate the nature of the invalid input.
    """
    try:
        # Evaluate the string input to check if it's a valid list
        evaluated_list = ast.literal_eval(candidate_list)
        if not isinstance(evaluated_list, list):
            raise ValueError(
                f"Input must be a list, got {type(evaluated_list).__name__}."
            )
        return evaluated_list  # Return the evaluated list
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid input for --mylist: {e}")


def convert_dataframes(pandas_dfs, keys):
    """
    Converts a list of pandas DataFrames to a Hugging Face `DatasetDict`.
    Args:
        pandas_dfs (list of pd.DataFrame): A list of pandas DataFrames to be converted.
        keys (list of str): A list of string keys to label each dataset in the DatasetDict.
            The length of `keys` must match the length of `pandas_dfs`.
    Returns:
        DatasetDict: A Hugging Face DatasetDict where each key from `keys` maps to a dataset
            created from the corresponding DataFrame in `pandas_dfs`.
    """
    ds = DatasetDict()
    for key, dataframe in zip(keys, pandas_dfs):
        ds[key] = Dataset.from_pandas(dataframe)
    return ds


def move_2_cpu(tensor):
    """Postprocess tensor.
    This function detaches tensor and moves it to CPU
    Args:
        tensor (Tensor): The input tensor.
    Returns:
        tensor: The postprocessed tensor.
    """
    return tensor.detach().cpu().clone()
