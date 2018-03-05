import tensorflow as tf


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


def mobilenet_fn(input_tensor, num_classes, is_training):
  """

  Args:
    input_tensor:
    num_classes:
    is_training:

  Returns:

  """
  with tf.variable_scope(name_or_scope='mobilenet', reuse=tf.AUTO_REUSE):
    tf.keras.backend.set_learning_phase(is_training)

    model = tf.keras.applications.MobileNet(
        input_tensor=input_tensor,
        include_top=True,
        weights=None)

    # Remove the last two layer (Conv2D, Reshape)
    # for fine-tuning on CIFAR-10.
    logits = tf.keras.layers.Conv2D(
        filters=num_classes, activation='softmax',
        kernel_size=(1, 1),
        padding='same')(model.layers[-4].output)
    # Create a new output layer for CIFAR-10.
    logits = tf.keras.layers.Reshape(
        target_shape=(num_classes,),
        name='output')(logits)

    model = tf.keras.Model(
        inputs=model.inputs,
        outputs=logits)

  return model


def model_fn(features, labels, mode, construct_model_fn, optimizer, params):
  """Construct `model_fn` for estimator

  Returns:
    EstimatorSpec
  """
  tf.summary.image('images', features, max_outputs=6)

  model = construct_model_fn(
      input_tensor=features,
      num_classes=params['num_classes'],
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  logits = model.outputs
  predictions = {
      'classes': tf.argmax(logits),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # L2 regularization
  loss = cross_entropy + params['weight_decay'] * tf.add_n(
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

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={'accuracy': accuracy})