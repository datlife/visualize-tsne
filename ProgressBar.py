import tensorflow as tf
import numpy as np

from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

class ProgressBar(tf.train.SessionRunHook):
  """Monitors training progress, similar to Keras training.

  This Logger uses `tf.keras.utils.Progbar` to write messages
  to stdout.

  Note: avoid to call the following command, it will cause messy looking
  stdout.
  ```python
    tf.logging.set_verbosity(tf.logging.INFO)
  ```

  """
  def __init__(self,
               epochs,
               steps_per_epoch,
               tensors_to_log=None):
    """Initializes a `ProgressBar`

    Args:
      epochs: `int`, Total number of expected epochs, None if unknown.
      steps_per_epoch: `int`, numbers of expected iterations per epoch
      tensors_to_log: - optional - can be:
          `dict` maps string-valued tags to tensors/tensor names,
          or `iterable` of tensors/tensor names.

    Raise:
      ValueError: `tensors_to_log` is not a list or a dictionary.
    """
    self._epochs = epochs
    self._step_per_epoch = steps_per_epoch

    if tensors_to_log is not None:
      if not isinstance(tensors_to_log, dict):
        self._tag_order = tensors_to_log
        tensors_to_log = {item: item for item in tensors_to_log}
      else:
        self._tag_order = tensors_to_log.keys()
      self._tensors = tensors_to_log
    else:
      self._tensors = None

  def begin(self):
    self._global_step_tensor = training_util._get_global_step_read()
    if self._global_step_tensor is None:
      raise RuntimeError(
        "Global step should be created to use KerasLogger")

    # Convert names to tensors if given
    if self._tensors:
      self._current_tensors = {tag: _as_graph_element(tensor)
                               for (tag, tensor) in self._tensors.items()}

  def after_create_session(self, session, coord):
    # Init current_epoch and current_step
    self._curr_step = session.run(self._global_step_tensor)
    if self._curr_step != 0:
      print("Resuming training from global step(s): %s...\n" % self._curr_step)

    self._curr_epoch = int(np.floor(self._curr_step / self._step_per_epoch))
    self._curr_step -= self._curr_epoch * self._step_per_epoch
    self._first_run = True


  def before_run(self, run_context):
    if self._first_run is  True:
      self._curr_epoch += 1
      print('Epoch %s/%s:' % (self._curr_epoch, self._epochs))
      self.progbar = tf.keras.utils.Progbar(target=self._step_per_epoch)
      self._first_run = False

    elif self._curr_step % self._step_per_epoch == 0:
      self._curr_epoch += 1
      self._curr_step = 0
      print('Epoch %s/%s:' % (self._curr_epoch, self._epochs))
      self.progbar = tf.keras.utils.Progbar(target=self._step_per_epoch)

    if self._tensors:
      return SessionRunArgs(self._current_tensors)


  def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
    if self._tensors:
      values = self._extract_tensors_info(run_values.results)
    else:
      values = None

    self._curr_step += 1
    self.progbar.update(self._curr_step, values=values)

  def _extract_tensors_info(self, tensor_values):
    stats = []
    for tag in self._tag_order:
      stats.append((tag, tensor_values[tag]))
    return stats
