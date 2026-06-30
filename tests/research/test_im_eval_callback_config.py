from __future__ import annotations

import inspect

from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback


def test_im_eval_callback_accepts_optional_max_eval_steps() -> None:
    signature = inspect.signature(ImEvalCallback)

    assert "max_eval_steps" in signature.parameters
    callback = ImEvalCallback(eval_frequency=1, max_eval_steps=128)
    assert callback.max_eval_steps == 128
