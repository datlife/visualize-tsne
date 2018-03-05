

def input_fn(is_training, dataset, preprocess_fn,
             batch_size, shuffle_buffer, num_epochs,
             num_parallel_calls=1, multi_gpu=False):
  """

  Args:
  Returns:

  """
  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  dataset = dataset.map(
      lambda img, label: preprocess_fn(img, label, is_training),
      num_parallel_calls=num_parallel_calls)

  dataset = dataset.batch(batch_size)
  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path.
  dataset = dataset.prefetch(1)

  return dataset