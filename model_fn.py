"""Model Function"""
import tensorflow as tf

def cifar10_mobilenet(features, labels, mode, params):
  """CIFAR-10 Model Function"""

  num_classes = 10

  # This is important!
  tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)

  model = tf.keras.applications.MobileNet(
      input_tensor=features,
      input_shape=(32, 32, 3),
      include_top=False,
      pooling='avg',
      weights=None)
  return model_fn(num_classes, model, features, labels, mode, params)

def model_fn(num_classes, keras_model, features, labels, mode, params):
  """An abstract method of model_fn

  Args:
    num_classes:
    keras_model:
    features:
    labels:
    mode:
    params:

  Returns:


  Raise:


  """
  feature_map = keras_model(features)
  logits = tf.keras.layers.Dense(units=num_classes)(feature_map)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
      'embeddings': feature_map}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  cross_entropy = tf.losses.softmax_cross_entropy(
      onehot_labels=labels, logits=logits)
  loss = cross_entropy + params['weight_decay'] * \
      tf.add_n([tf.nn.l2_loss(t) for t in tf.trainable_variables()])

  # Setting up metrics
  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    optimizer = params['optimizer'](params['learning_rate'])

    if params['multi_gpu']:
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    # update ops for Batch Norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  tf.identity(loss, name='train_loss')
  tf.summary.scalar('train_loss', loss)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)
