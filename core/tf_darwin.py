"""TensorFlow defaults on macOS to avoid kernel crashes with Metal / tf.data."""

from __future__ import annotations

import platform

_applied = False


def apply_darwin_tf_compat() -> None:
    """Enable eager execution and tf.data debug mode on Darwin (idempotent).

    Call after ``import tensorflow`` and before building datasets or training.
    Safe no-op on non-macOS.
    """
    global _applied
    if _applied or platform.system() != "Darwin":
        return
    import tensorflow as tf

    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    _applied = True
