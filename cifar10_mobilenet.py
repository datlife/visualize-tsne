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

_BATCH_SIZE = 128
_NUM_EPOCHS = 3
_SHUFFLE_BUFFER = 1000

tf.logging.set_verbosity(tf.logging.DEBUG)


def cifar10_mobilenet_fn(features, labels, mode, params):
  params['weight_decay'] = 2e-4
  params['num_classes'] = _CIFAR10_CLASSES

  learning_rate = 0.001
  optimizer = tf.train.AdamOptimizer(learning_rate)

  return model.model_fn(
      features, labels, mode,
      construct_model_fn=model.mobilenet_fn,
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


def main():
  cifar10 = tf.keras.datasets.cifar10.load_data()

  # Create an Estimator for training/evaluation
  classifier = model.get_estimator(
    model_function=cifar10_mobilenet_fn,
    model_dir='model')

  print('Starting a training cycle.')
  images, labels = cifar10[0]  # training

  classifier.train(
      input_fn=lambda: dataset.input_fn(
          is_training=True,
          num_epochs=_NUM_EPOCHS,
          batch_size=_BATCH_SIZE,
          preprocess_fn=cifar10_preprocess,
          shuffle_buffer=_SHUFFLE_BUFFER,
          num_parallel_calls=16,
          dataset=tf.data.Dataset.from_tensor_slices((images, labels))),)

  print('Starting to evaluate.')
  test_images, test_labels = cifar10[1]  # testing

  eval_results = classifier.evaluate(
      input_fn=lambda: dataset.input_fn(
        is_training=False,
        num_epochs=1,
        batch_size=_BATCH_SIZE,
        preprocess_fn=cifar10_preprocess,
        dataset=tf.data.Dataset.from_tensor_slices((test_images, test_labels)),
        shuffle_buffer=_SHUFFLE_BUFFER,
        num_parallel_calls=16))

  print(eval_results)


if __name__ == '__main__':
    main()
