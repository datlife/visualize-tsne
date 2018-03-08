import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

class KerasLogger(tf.train.SessionRunHook):
  """Write metrics to std out

  """
  def __init__(self, tensors, epochs, steps_per_epoch, curr_epoch=0):
    self._epochs = epochs
    self._step_per_epoch = steps_per_epoch
    self._curr_epoch = curr_epoch
    self._curr_step = 0

    if not isinstance(tensors, dict):
      self._tag_order = tensors
      tensors = {item: item for item in tensors}
    else:
      self._tag_order = tensors.keys()
    self._tensors = tensors

  def begin(self):
    self._global_step_tensor = training_util._get_global_step_read()
    if self._global_step_tensor is None:
      raise RuntimeError(
        "Global step should be created to use KerasLogger")

    # Convert names to tensors if given
    self._current_tensors = {tag: _as_graph_element(tensor)
                             for (tag, tensor) in self._tensors.items()}

  def before_run(self, run_context):
    if self._curr_step % self._step_per_epoch == 0:
      self._curr_epoch += 1
      self._curr_step = 0
      print('Epoch %s/%s:' % (self._curr_epoch, self._epochs))
      self.progbar = tf.keras.utils.Progbar(target=self._step_per_epoch)

    return SessionRunArgs(self._current_tensors)

  def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
    values = self._log_tensors(run_values.results)
    self._curr_step += 1
    self.progbar.update(self._curr_step, values=values)

  def _log_tensors(self, tensor_values):
    stats = []
    for tag in self._tag_order:
      stats.append((tag, tensor_values[tag]))
    return stats
