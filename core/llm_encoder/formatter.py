import os
import json


def concatenate_to_template(template, dynamic_content, sep):
    """
    Concatenates a template string with dynamic content, using a specified separator.
    This function combines a template and dynamic content, separated by a given
    separator, and removes any leading or trailing whitespace.
    Args:
        template (str): The template string to prepend to the dynamic content.
        dynamic_content (str): The content to append to the template.
        sep (str): The separator to place between the template and the dynamic content.
    Returns:
        str: The concatenated string with the template, separator, and dynamic content.
    Example:
        >>> concatenate_to_template("Template for", "Example1", sep=" ")
        'Template for Example1'
    """
    return (template + sep + dynamic_content).strip()


def apply_template(df, column, prompt_template, template_for="rules", sep=" "):
    """
    Applies a template to each element in a specified column of a DataFrame or dictionary.

    This function verifies that the specified template type exists in the provided
    template dictionary (`prompt_template`). It then formats each element in the specified
    column using the given template and adds the formatted text as a new list to `sample`
    under the key specified by `template_for`.

    Args:
        sample (pd.DataFrame or dict): The sample containing the column to which the template
            will be applied. Can be a pandas DataFrame or dictionary with lists.
        column (str): The name of the column in `sample` containing elements to be templated.
        prompt_template (dict): A dictionary containing different template strings, where keys
            represent template types (e.g., "rules").
        template_for (str, optional): The key in `prompt_template` for the template to use.
            Default is "rules".
        sep (str, optional): The separator to use between concatenated template components.
            Default is a single space `" "`.
    Returns:
        pd.DataFrame or dict: The modified sample with the new templated list added under
        the key `template_for`.
    Raises:
        AssertionError: If `template_for` is not found in `prompt_template`.
    Example:
        >>> sample = {'text': ["Example1", "Example2"]}
        >>> prompt_template = {"rules": "Template for {}"}
        >>> apply_template(sample, "text", prompt_template, template_for="rules")
        {'text': ["Example1", "Example2"], 'rules': ["Template for Example1", "Template for Example2"]}
    Notes:
        - This function uses a helper function `concatenate_to_template` to format each
          element with the specified template.
        - The function assumes `sample` is mutable and will modify it in place.
    """
    assert (
        template_for in prompt_template
    ), f"Error: template for {template_for} is not present in the template file!"
    df[template_for] = [
        concatenate_to_template(prompt_template[template_for], el, sep)
        for el in df[column]
    ]
    return df
