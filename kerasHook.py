import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs


class KerasLogger(tf.train.SessionRunHook):
  """Write metrics to std out

  """
  def __init__(self, epochs, steps_per_epoch, curr_epoch=0):
    self._epochs = epochs
    self._step_per_epoch = steps_per_epoch
    self._curr_epoch = curr_epoch
    self._curr_step = 0

  def begin(self):
    self._global_step_tensor = training_util._get_global_step_read()
    if self._global_step_tensor is None:
      raise RuntimeError(
        "Global step should be created to use KerasLogger")

    loss = tf.
  def after_create_session(self, session, coord):
    pass

  def before_run(self, run_context):
    if self._curr_step % self._step_per_epoch == 0:
      self._curr_epoch += 1
      self._curr_step = 0
      print('Epoch %s/%s:' % (self._curr_epoch, self._epochs))
      self.progbar = tf.keras.utils.Progbar(target=self._step_per_epoch)

    return SessionRunArgs(self._global_step_tensor)

  def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
    _ = run_context

    global_step = run_context.session.run(self._global_step_tensor)
    self._curr_step += 1
    self.progbar.update(self._curr_step, values=[('steps', global_step)])
