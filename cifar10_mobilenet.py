r"""An example of how to use Tensorboard .

In this case, this script will:
  * Fine-tune a pre-trained MobileNet on Cifar-10 dataset
  * Summary the training process on Tensorboard
  * Visualize t-SNE
"""
import dataset
import tensorflow as tf

_CIFAR10_CLASSES = 10
_HEIGHT, _WIDTH, _DEPTH = 128, 128, 3

_EPOCHS = 5
_BATCH = 64
_SHUFFLE_BUFFER = 100

tf.logging.set_verbosity(tf.logging.DEBUG)

def main():
  # Load dataset
  cifar10 = tf.keras.datasets.cifar10.load_data()

  model = mobilenet_fn(_CIFAR10_CLASSES)
  model.summary()

  def cross_entropy_with_weight_decay(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = loss + 0.00002 * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'conv' in v.name])
    return loss

  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss=cross_entropy_with_weight_decay,
      metrics=[tf.keras.metrics.categorical_accuracy])

  print('Starting a training cycle')
  model.fit_generator(
      epochs=_EPOCHS,
      generator=cifar10_generator(cifar10[0], True, _BATCH, _EPOCHS, 100),
      validation_data=cifar10_generator(cifar10[1], False, _BATCH, None, None),
      validation_steps=int(len(cifar10[1][0])/_BATCH),
      steps_per_epoch=400,
      workers=0,
      verbose=1,
      shuffle=False)
  model.save_weights('model.weights')


def cifar10_generator(data, is_training, batch_size, epochs,
                      shuffle_buffer, num_parallel_calls=8, multi_gpu=False):
  images, labels = data
  data = tf.data.Dataset.from_tensor_slices((images, labels))
  data = dataset.input_fn(
      is_training=is_training,
      dataset=data,
      preprocess_fn=cifar10_preprocess,
      batch_size=batch_size,
      shuffle_buffer=shuffle_buffer,
      num_epochs=epochs,
      num_parallel_calls=num_parallel_calls,
      multi_gpu=multi_gpu)

  iterator = data.make_one_shot_iterator()
  next_batch = iterator.get_next()
  session = tf.keras.backend.get_session()
  while True:
     yield session.run(next_batch)


def cifar10_preprocess(image, label, is_training):
  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT, _WIDTH)
  label = tf.one_hot(tf.cast(label[0], tf.int32), _CIFAR10_CLASSES)

  if is_training:  # perform augmentation
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
    image = tf.image.random_flip_left_right(image)

  return image, label


def mobilenet_fn(num_classes):
  """

  Args:
    input_tensor:
    num_classes:
    is_training:

  Returns:

  """
  # tf.keras.backend.set_learning_phase(is_training)

  mobile_net = tf.keras.applications.MobileNet(
      input_shape=(_HEIGHT, _WIDTH, _DEPTH),
      include_top=True,
      weights='imagenet')

  # Remove the last two layer (Conv2D, Reshape)
  # for fine-tuning on CIFAR-10.
  logits = tf.keras.layers.Conv2D(
      filters=num_classes, activation='softmax',
      kernel_size=(1, 1),
      padding='same')(mobile_net.layers[-4].output)
  # Create a new output layer for CIFAR-10.
  logits = tf.keras.layers.Reshape(
      target_shape=(num_classes,),
      name='output')(logits)

  mobile_net = tf.keras.Model(
      inputs=mobile_net.inputs,
      outputs=logits)

  return mobile_net


if __name__ == '__main__':
    main()

  # # Create an Estimator for training/evaluation
  # classifier = model.get_estimator(
  #   model_function=cifar10_mobilenet_fn,
  #   model_dir='model')
  #
  # print('Starting a training cycle.')
  # classifier.train(
  #     input_fn=lambda: dataset.input_fn(
  #         is_training=True,
  #         dataset=training_data,
  #         preprocess_fn=cifar10_preprocess,
  #         num_epochs=_NUM_EPOCHS,
  #         batch_size=_BATCH_SIZE,
  #         shuffle_buffer=_SHUFFLE_BUFFER,
  #         num_parallel_calls=16),)
  #
  # print('Starting to evaluate.')
  # test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
  # eval_results = classifier.evaluate(
  #     input_fn=lambda: dataset.input_fn(
  #       is_training=False,
  #       dataset=test_data,
  #       preprocess_fn=cifar10_preprocess,
  #       num_epochs=1,
  #       batch_size=_BATCH_SIZE,
  #       shuffle_buffer=_SHUFFLE_BUFFER,
  #       num_parallel_calls=16))
  #
  # print(eval_results)
