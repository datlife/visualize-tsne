r"""An example of how to use Tensorboard .

In this case, this script will:
  * Fine-tune a pre-trained MobileNet on Cifar-10 dataset
  * Summary the training process on Tensorboard
  * Visualize t-SNE
"""
import tensorflow as tf
import dataset
import model

_CIFAR10_CLASSES = 10
_HEIGHT, _WIDTH, _DEPTH = 128, 128, 3

_NUM_EPOCHS = 3
_BATCH_SIZE = 128
_SHUFFLE_BUFFER = 1000

tf.logging.set_verbosity(tf.logging.DEBUG)


def main():
  # Load dataset
  cifar10 = tf.keras.datasets.cifar10.load_data()

  # Create an Estimator for training/evaluation
  classifier = model.get_estimator(
    model_function=cifar10_mobilenet_fn,
    model_dir='model')

  print('Starting a training cycle.')
  images, labels = cifar10[0]
  training_data = tf.data.Dataset.from_tensor_slices((images, labels))

  classifier.train(
      input_fn=lambda: dataset.input_fn(
          is_training=True,
          dataset=training_data,
          preprocess_fn=cifar10_preprocess,
          num_epochs=_NUM_EPOCHS,
          batch_size=_BATCH_SIZE,
          shuffle_buffer=_SHUFFLE_BUFFER,
          num_parallel_calls=16),)

  print('Starting to evaluate.')
  test_images, test_labels = cifar10[1]  # testing
  test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
  eval_results = classifier.evaluate(
      input_fn=lambda: dataset.input_fn(
        is_training=False,
        dataset=test_data,
        preprocess_fn=cifar10_preprocess,
        num_epochs=1,
        batch_size=_BATCH_SIZE,
        shuffle_buffer=_SHUFFLE_BUFFER,
        num_parallel_calls=16))

  print(eval_results)


def cifar10_mobilenet_fn(features, labels, mode, params):
  params['weight_decay'] = 2e-4
  params['num_classes'] = _CIFAR10_CLASSES

  learning_rate = 0.001
  optimizer = tf.train.AdamOptimizer(learning_rate)

  return model.model_fn(
      features, labels, mode,
      construct_model_fn=mobilenet_fn,
      optimizer=optimizer,
      params=params)


def cifar10_preprocess(image, label, is_training):
  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT, _WIDTH)
  label = tf.one_hot(tf.cast(label, tf.int32), _CIFAR10_CLASSES)

  if is_training:  # perform augmentation
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
    image = tf.image.random_flip_left_right(image)
  return image, label


def mobilenet_fn(input_tensor, num_classes, is_training):
  """

  Args:
    input_tensor:
    num_classes:
    is_training:

  Returns:

  """
  with tf.variable_scope('mobilenet', reuse=tf.AUTO_REUSE):
    tf.keras.backend.set_learning_phase(is_training)

    mobile_net = tf.keras.applications.MobileNet(
        input_tensor=input_tensor,
        include_top=True,
        weights=None)

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
