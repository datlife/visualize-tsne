r"""An example of how to use Tensorboard .

In this case, this script will:
  * Fine-tune a pre-trained MobileNet on Cifar-10 dataset
  * Summary the training process on Tensorboard
  * Visualize t-SNE
"""
import tensorflow as tf
from tensorboard.plugins import projector

tf.logging.set_verbosity(tf.logging.INFO)
_CIFAR10_CLASSES = 10
_HEIGHT, _WIDTH, _DEPTH = 128, 128, 3


def main():
  # Load CIFAR-10 dataset
  cifar10 = tf.keras.datasets.cifar10.load_data()

  # Create an Estimator for training/evaluation
  classifier = tf.estimator.Estimator(
      model_fn=cifar10_model_fn, model_dir='model',
      config=tf.estimator.RunConfig(),
      params={
        'learning_rate': 0.001,
        'weight_decay': 2e-4,
        'multi_gpu': False})

  # Training
  training_epochs = 5
  epochs_per_eval = 1

  for _ in range(training_epochs // epochs_per_eval):
    print("Starting a training cycle")
    classifier.train(input_fn=lambda: input_fn(
        tf.estimator.ModeKeys.TRAIN, cifar10[0], epochs_per_eval, 128,
        cifar10_preprocess, 128, 8, True))

    print("Starting an evaluation cycle")
    eval_results = classifier.evaluate(input_fn=lambda: input_fn(
        tf.estimator.ModeKeys.EVAL, cifar10[1], 1, 128,
        cifar10_preprocess, None, 8, True))

    print(eval_results)


# #############################################################################
# CIFAR-10 PROCESSING
# #############################################################################
def cifar10_preprocess(image, label, mode):
  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT, _WIDTH)

  if mode == tf.estimator.ModeKeys.EVAL or \
      mode == tf.estimator.ModeKeys.TRAIN:

    label = tf.one_hot(tf.cast(label[0], tf.int32), _CIFAR10_CLASSES)
    if tf.estimator.ModeKeys.TRAIN:  # perform augmentation
      image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image, label

  else:  # predict mode
    return image
# #############################################################################
# INPUT_FN
# #############################################################################
def input_fn(mode, dataset, num_epochs, batch_size, preprocess_fn,
             shuffle_buffer=None, num_parallel_calls=4, multi_gpu=False):
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

  """
  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = tf.data.Dataset.from_tensor_slices(dataset)
  dataset = dataset.prefetch(buffer_size=batch_size)

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  dataset = dataset.map(
      lambda image, label: preprocess_fn(image, label, mode),
      num_parallel_calls=num_parallel_calls)

  dataset = dataset.batch(batch_size)
  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path.
  dataset = dataset.prefetch(1)

  return dataset

# #############################################################################
# MODEL_FN CONSTRUCTION
# #############################################################################
def cifar10_model_fn(features, labels, mode, params):
  """

  Args:
    features:
    labels:
    mode:
    params:

  Returns:

  """

  model = mobilenet_fn(input_tensor=features, num_classes=10)
  logits = model.outputs[0]
  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  loss = tf.losses.softmax_cross_entropy(labels, logits) + \
         params['weight_decay'] * \
         tf.add_n([tf.nn.l2_loss(t) for t in tf.trainable_variables()
                   if 'bn' not in t.name])

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])

    if params['multi_gpu']:
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    # update ops for Batch Norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_or_create_global_step())
  else:
    train_op = None

  if mode == tf.estimator.ModeKeys.EVAL:
    # create projector
    
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={'accuracy': tf.metrics.accuracy(labels,
                                                       tf.nn.softmax(logits))})

def mobilenet_fn(input_tensor, num_classes):
  """Mobilenet

  Args:
    input_tensor: tf.Tensor
    num_classes:  int

  Returns:
    model: tf.models.Model
  """
  mobile_net = tf.keras.applications.MobileNet(
      input_tensor=input_tensor,
      input_shape=(128, 128, 3),
      include_top=False,
      weights='imagenet' if tf.train.get_global_step() == 0 else None)

  feature_map = tf.keras.layers.GlobalAvgPool2D()(mobile_net.layers[-1].output)
  logits = tf.keras.layers.Dense(num_classes, name='output')(feature_map)

  mobile_net = tf.keras.models.Model(
      inputs=mobile_net.inputs,
      outputs=logits)

  return mobile_net


if __name__ == '__main__':
    main()