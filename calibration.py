r"""
Temperature scaling and Best-of-N candidate scoring for the Qwen chat model.

Key equations:
- Temperature scaling: $\text{softmax}(z_i / T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$
  - Higher T (>1) softens probabilities; lower T (<1) sharpens them.

- Length-normalized log-likelihood score (Best-of-N):
  $s(x, y) = \frac{1}{|M|} \sum_{t \in M} \log p(y_t | x, y_{<t}; T_{\text{calib}})$
  where $M$ is the set of generated positions that are not padding. This is the log of
  the geometric mean token probability, so it does not favour short responses.

The per-token log-probabilities are recorded during generation by
``TokenLogProbRecorder`` (one vocabulary row per step) instead of keeping the full
``[N, steps, vocab]`` score tensor in GPU memory.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from config import ChatConfig


class SimpleTemperatureScaler:
    """Temperature scaling for candidate scoring."""

    def __init__(self, config: ChatConfig):
        self.temperature = config.calibration_temperature

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        r"""Return $\text{logits} / T_{\text{calib}}$."""
        return logits / self.temperature

    def log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Log-softmax over the vocabulary after calibration scaling."""
        return F.log_softmax(self.scale_logits(logits.float()), dim=-1)

    @staticmethod
    def score_token_log_probs(
            token_log_probs: torch.Tensor,
            generated_tokens: torch.Tensor,
            pad_token_id: Optional[int],
    ) -> Dict[str, float]:
        """
        Length-normalized score from per-token log-probabilities.

        Args:
            token_log_probs: log p(y_t) for each generated position [seq_len]
            generated_tokens: generated token IDs [seq_len]
            pad_token_id: positions holding this ID are excluded from the score
        """
        if pad_token_id is None:
            valid_mask = torch.ones_like(generated_tokens, dtype=torch.bool)
        else:
            valid_mask = generated_tokens != pad_token_id
        valid_len = int(valid_mask.sum().item())
        if valid_len == 0:
            return {'score': -float('inf'), 'geom_mean_prob': 0.0, 'token_count': 0}

        score = token_log_probs[valid_mask].sum().item() / valid_len
        return {
            'score': score,
            'geom_mean_prob': float(torch.exp(torch.tensor(score)).item()),
            'token_count': valid_len,
        }

    def score_candidate(
            self,
            logits: torch.Tensor,
            generated_tokens: torch.Tensor,
            pad_token_id: int
    ) -> Dict[str, float]:
        """Score a candidate from its full logits [seq_len, vocab_size] (kept for offline use)."""
        log_probs = self.log_probs(logits)
        positions = torch.arange(generated_tokens.shape[0], device=log_probs.device)
        token_log_probs = log_probs[positions, generated_tokens]
        return self.score_token_log_probs(token_log_probs, generated_tokens, pad_token_id)

    def set_temperature(self, temperature: float):
        self.temperature = temperature


class TokenLogProbRecorder(LogitsProcessor):
    """
    Records the calibrated log-probability of each sampled token during ``generate``.

    It must be the last logits processor so it sees the same scores the sampler uses
    (after repetition penalty, temperature, top-k and top-p). At step t it receives
    ``input_ids`` that end with the token sampled at step t-1, so it looks that token
    up in the previous step's log-softmax row. Only one ``[batch, vocab]`` row is kept.
    """

    def __init__(self, scaler: SimpleTemperatureScaler):
        self.scaler = scaler
        self._previous: Optional[torch.Tensor] = None
        self._steps = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._previous is not None:
            self._steps.append(self._previous.gather(1, input_ids[:, -1:]).squeeze(1))
        self._previous = self.scaler.log_probs(scores)
        return scores

    def finish(self, sequences: torch.LongTensor) -> torch.Tensor:
        """Return log-probabilities for every generated token, shape [batch, generated_len]."""
        if self._previous is not None:
            self._steps.append(self._previous.gather(1, sequences[:, -1:]).squeeze(1))
            self._previous = None
        if not self._steps:
            return torch.empty(sequences.shape[0], 0, device=sequences.device)
        return torch.stack(self._steps, dim=1)
