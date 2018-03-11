"""Input Function"""

import tensorflow as tf

def input_fn(mode, dataset, num_epochs, batch_size,
             preprocess_fn, shuffle_buffer=None,
             num_parallel_calls=4, multi_gpu=False):  # pylint: disable=unused-argument
  """

  Args:
    mode:
    dataset:
    num_epochs:
    batch_size:
    preprocess_fn:
    shuffle_buffer:
    num_parallel_calls:
    multi_gpu:

  Returns:


  Raises:


  """
  dataset = tf.data.Dataset.from_tensor_slices(dataset)
  dataset = dataset.prefetch(buffer_size=batch_size)

  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  dataset = dataset.repeat(num_epochs)
  if mode != tf.estimator.ModeKeys.PREDICT:
    dataset = dataset.map(
        lambda image, label: preprocess_fn(image, label, mode),
        num_parallel_calls=num_parallel_calls)
  else:
    dataset = dataset.map(
        lambda image: preprocess_fn(image, None, mode), num_parallel_calls)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(1)

  return dataset
