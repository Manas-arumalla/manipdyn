"""Headless GUI smoke test: the control center constructs and configures.

Runs under Qt's 'offscreen' platform so it needs no display. We don't drive the
event loop — just verify the window builds, the modes are present, and
per-controller gain fields populate, which catches import/layout regressions.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_gui_constructs_and_populates_gains():
    from PySide6.QtWidgets import QApplication

    from manipdyn.control import CONTROLLERS
    from manipdyn.gui.app import _MODES, ManipdynGUI

    app = QApplication.instance() or QApplication([])
    win = ManipdynGUI()

    # all modes wired up
    assert win.mode_group.buttons()
    assert len(win.mode_group.buttons()) == len(_MODES)

    # every controller selectable in Reach mode, with gain fields
    assert win.cb_ctrl.count() == len(CONTROLLERS)
    win.cb_ctrl.setCurrentText("pid")
    assert set(win.gain_fields) == {"kp", "ki", "kd"}
    win.cb_ctrl.setCurrentText("tsid")
    assert "kp" in win.gain_fields

    # mode switching swaps the config panel without error
    for i in range(len(_MODES)):
        win._select_mode(i)
        assert win.config_stack.currentIndex() == i

    win.close()
    del win
    del app
