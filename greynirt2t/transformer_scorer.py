"""
 This is a copy of tensor2tensor.utils.transformer.TransformerScorer (t2t v1.14)
"""
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import TransformerScorer
from tensor2tensor.layers import common_layers

import tensorflow as tf


@registry.register_model
class TransformerLogProbScorer(TransformerScorer):
  """Transformer model, but only scores in PREDICT mode.

  Checkpoints between Transformer and TransformerScorer are interchangeable.
  """

  def __init__(self, *args, **kwargs):
    super(TransformerLogProbScorer, self).__init__(*args, **kwargs)
    self._name = "transformer"
    self._base_name = "transformer"

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Returns the targets and their log probabilities."""
    del decode_length, beam_size, top_beams, alpha, use_tpu
    assert features is not None

    self._fill_problem_hparams_features(features)

    # Run the model
    self.hparams.force_full_predict = True
    with tf.variable_scope(self.name):
      logits, _ = self.model_fn(features)
    assert len(logits.shape) == 5  # [batch, time, 1, 1, vocab]
    logits = tf.squeeze(logits, [2, 3])
    #import pdb; pdb.set_trace()

    # Compute the log probabilities
    log_probs = common_layers.log_prob_from_logits(logits)

    targets = features["targets"]
    assert len(targets.shape) == 4  # [batch, time, 1, 1]
    targets = tf.squeeze(targets, [2, 3])

    # Slice out the log_probs of the targets
    log_probs = common_layers.index_last_dim_with_indices(log_probs, targets)

    # return log-probs instead of beam-score
    return {"outputs": targets, "scores": log_probs}


