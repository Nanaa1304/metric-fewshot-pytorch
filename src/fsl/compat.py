"""Compatibility patch between easyfsl 1.5.0 and recent PyTorch versions.

Recent PyTorch releases removed the ``data_source`` argument of
``torch.utils.data.Sampler.__init__``, which easyfsl's ``TaskSampler`` still
passes. The patch turns that call into a no-op without changing the behaviour
of the sampler. Import this module BEFORE importing anything from easyfsl.
"""
import torch.utils.data as tud

_PATCHED = False


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return

    def _patched_sampler_init(self, *args, **kwargs):
        return None

    tud.Sampler.__init__ = _patched_sampler_init
    _PATCHED = True


apply()
