"""Fine=tune MobileNet on CIFAR-10 and Visualize result on Tensorboard ."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ProgressBar import ProgressBar

from input_fn import input_fn
from model_fn import cifar10_mobilenet
from visualize import visualize_embeddings, get_samples

_CIFAR10_CLASSES = 10
_HEIGHT, _WIDTH, _DEPTH = 32, 32, 3
_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

def cifar10_preprocess(image, label, mode):
  """Preprocess Inputs function

  Args:
    image:
    label:
    mode:

  Returns:

  """
  image = tf.image.per_image_standardization(image)
  image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT, _WIDTH)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return image
  else:  # train/eval mode
    label = tf.one_hot(tf.cast(label[0], tf.int32), _CIFAR10_CLASSES)

    if mode == tf.estimator.ModeKeys.TRAIN:  # perform augmentation
      image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image, label


def main():

  # ##########################
  # Configure hyper-parameters
  # ##########################
  batch_size = 1024

  epochs_per_eval = 1
  training_steps = 5000
  steps_per_epoch = 1000
  training_epochs = int(training_steps // steps_per_epoch)

  cpu_cores = 8
  multi_gpu = True
  shuffle_buffer = 2048

  model_dir = 'model'

  # ########################
  # Load CIFAR-10 dataset
  # ########################
  cifar10 = tf.keras.datasets.cifar10.load_data()

  # ########################
  # Define a Classifier
  # ########################
  model_fn = cifar10_mobilenet if not multi_gpu else \
      tf.contrib.estimator.replicate_model_fn(
          cifar10_mobilenet, tf.losses.Reduction.MEAN)

  classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=model_dir,
      config=tf.estimator.RunConfig().replace(
          save_checkpoints_steps=steps_per_epoch * epochs_per_eval,
          save_summary_steps=200),
      params={
          'learning_rate': 0.001,
          'optimizer': tf.train.AdamOptimizer,
          'weight_decay': 2e-4,
          'multi_gpu': multi_gpu})

  # #########################
  # Training/Eval
  # #########################
  tensors_to_log = ['train_accuracy', 'train_loss']
  for _ in range(training_epochs // epochs_per_eval):
    classifier.train(
        input_fn=lambda: input_fn(
            tf.estimator.ModeKeys.TRAIN, cifar10[0], None, batch_size,
            cifar10_preprocess, shuffle_buffer, cpu_cores, multi_gpu),
        steps=epochs_per_eval * steps_per_epoch,
        hooks=[ProgressBar(training_epochs, steps_per_epoch, tensors_to_log)])

    print("\nStart evaluating...")
    eval_results = classifier.evaluate(
        input_fn=lambda: input_fn(
            tf.estimator.ModeKeys.EVAL, cifar10[1], 1, batch_size,
            cifar10_preprocess, None, cpu_cores, multi_gpu))
    print(eval_results)

  # ##################################
  # Visualize t-SNE
  # ###################################
  output_dir = 'model/projector'

  # Randomly pick 50 samples from each class
  images, _ = get_samples(cifar10[0],
                          logdir=output_dir,
                          samples_per_class=100)

  # Extract embeddings (output before output layer) from
  # above samples.
  predictions = classifier.predict(
      input_fn=lambda: input_fn(
          tf.estimator.ModeKeys.PREDICT, images, 1, batch_size,
          cifar10_preprocess, None, 8, True))

  embeddings = tf.stack(
      values=[pred['embeddings'] for pred in predictions],
      axis=0)

  visualize_embeddings(images, embeddings, output_dir)


if __name__ == '__main__':
  main()
