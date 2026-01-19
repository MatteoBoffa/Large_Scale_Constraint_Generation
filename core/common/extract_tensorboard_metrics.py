from tensorboard.backend.event_processing import event_accumulator


def extract_metric(input_dir, tag):
    """Extracts metric values from TensorBoard logs stored in a specified directory.

    Parameters:
    -----------
    input_dir : str
        The directory containing TensorBoard log files (event files) from which metrics are extracted.

    tag : str
        The specific metric tag to extract (e.g., 'loss', 'accuracy'). This tag corresponds to the scalar
        data recorded in the TensorBoard logs.

    Returns:
    --------
    list of tensorflow.python.summary.summary_iterator.SummaryIterator.Event
        A list of `Event` objects containing the scalar metric values for the specified tag. Each event
        represents a data point recorded at a specific step in the TensorBoard logs.

    Example:
    --------
    >>> events = extract_metric("/path/to/logs", "accuracy")
    >>> for event in events:
    >>>     print(f"Step: {event.step}, Value: {event.value}")
    Step: 100, Value: 0.85
    Step: 200, Value: 0.90

    Logic:
    ------
    - Initializes a TensorBoard `EventAccumulator` to load the logs from the specified directory.
    - Reloads the data to ensure all metrics are available for extraction.
    - Retrieves the scalar events corresponding to the specified tag.
    - Returns the list of scalar `Event` objects.
    """
    # Load TensorBoard logs
    ea = event_accumulator.EventAccumulator(input_dir)
    # This will load all the available data
    ea.Reload()
    # Accessing scalar data corresponding to the specified tag
    events = ea.Scalars(tag)
    return events
