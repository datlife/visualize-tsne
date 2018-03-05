r"""An example of how to use Tensorboard .

In this case, this script will:
  * Fine-tune a pre-trained MobileNet on Cifar-10 dataset
  * Summary the training process on Tensorboard
  * Visualize t-SNE
"""
import tensorflow as tf

_CIFAR10_CLASSES = 10
_HEIGHT, _WIDTH, _DEPTH = 128, 128, 3

_BATCH_SIZE = 128
_NUM_EPOCHS = 3
_SHUFFLE_BUFFER = 1000

tf.logging.set_verbosity(tf.logging.DEBUG)

############################################
# Data processing
############################################
def preprocess(image, label, is_training):
  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT, _WIDTH)
  label = tf.one_hot(tf.cast(label, tf.int32), _CIFAR10_CLASSES)
  if is_training:  # perform augmentation
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
    image = tf.image.random_flip_left_right(image)
  return image, label

def main():
  cifar10 = tf.keras.datasets.cifar10.load_data()

  # Create an Estimator for training/evaluation
  classifier = get_estimator(
    model_function=cifar10_mobilenet_fn,
    model_dir='model')

  print('Starting a training cycle.')
  images, labels = cifar10[0]

  classifier.train(
      input_fn=lambda: input_fn(
          is_training=True,
          num_epochs=_NUM_EPOCHS,
          batch_size=_BATCH_SIZE,
          preprocess_fn=preprocess,
          shuffle_buffer=_SHUFFLE_BUFFER,
          num_parallel_calls=16,
          dataset=tf.data.Dataset.from_tensor_slices((images, labels))),)

  print('Starting to evaluate.')
  test_images, test_labels = cifar10[1]

  eval_results = classifier.evaluate(
      input_fn=lambda: input_fn(
        is_training=False,
        num_epochs=1,
        batch_size=_BATCH_SIZE,
        preprocess_fn=preprocess,
        dataset=tf.data.Dataset.from_tensor_slices((test_images, test_labels)),
        shuffle_buffer=_SHUFFLE_BUFFER,
        num_parallel_calls=16))

  print(eval_results)


def get_estimator(model_function, model_dir):
  session_config = tf.ConfigProto(
      inter_op_parallelism_threads=16,
      intra_op_parallelism_threads=16)

  run_config = tf.estimator.RunConfig().replace(
      save_summary_steps=100,
      log_step_count_steps=100,
      save_checkpoints_steps=1000,
      session_config=session_config)

  classifier = tf.estimator.Estimator(
      model_fn=model_function,
      model_dir=model_dir,
      config=run_config,
      params={'multi_gpu': False})
  return classifier


def cifar10_mobilenet_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  weight_decay = 2e-4
  learning_rate = 0.001
  optimizer = tf.train.AdamOptimizer(learning_rate)
  return model_fn(features, labels, mode, construct_model,
                  optimizer=optimizer,
                  weight_decay=weight_decay)


def model_fn(features, labels, mode, construct_model_fn, weight_decay,
             optimizer, multi_gpu=False):
  """Construct `model_fn` for estimator

  Returns:
    EstimatorSpec
  """
  tf.summary.image('images', features, max_outputs=6)

  model = construct_model_fn(
      input_tensor=features,
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  logits = model.outputs
  predictions = {
      'classes': tf.argmax(logits),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  # Compute soft-max cross-entropy loss
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # tf.identity(cross_entropy, name='cross_entropy')
  # tf.summary.scalar('cross_entropy', cross_entropy)

  # L2 regularization
  loss = cross_entropy + weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    # Batch norm requires update ops to be added
    # as a dependency to the train op.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  # Create a tensor named `train_accuracy` for logging
  # tf.identity(accuracy[1], name='train_accuracy')
  # tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={'accuracy': accuracy})


def construct_model(input_tensor, is_training):
  with tf.variable_scope(name_or_scope='mobilenet', reuse=tf.AUTO_REUSE):

    tf.keras.backend.set_learning_phase(is_training)

    model = tf.keras.applications.MobileNet(
        input_tensor=input_tensor,
        include_top=True,
        weights=None)
    # Remove the last two layer (Conv2D, Reshape)
    # for fine-tuning on CIFAR-10.
    logits = tf.keras.layers.Conv2D(
        filters=_CIFAR10_CLASSES, activation='softmax',
        kernel_size=(1, 1),
        padding='same')(model.layers[-4].output)
    # Create a new output layer for CIFAR-10.
    logits = tf.keras.layers.Reshape(
        target_shape=(_CIFAR10_CLASSES,),
        name='output')(logits)
    model = tf.keras.Model(
        inputs=model.inputs,
        outputs=logits)
  return model


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


if __name__ == '__main__':
    main()
