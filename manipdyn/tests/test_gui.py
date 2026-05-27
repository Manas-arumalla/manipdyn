"""Headless GUI smoke test: the control center constructs and configures.

Runs under Qt's 'offscreen' platform so it needs no display. We don't drive the
event loop — just verify the window builds and per-controller gain fields
populate, which catches import/layout regressions.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_gui_constructs_and_populates_gains():
    from PySide6.QtWidgets import QApplication

    from manipdyn.control import CONTROLLERS
    from manipdyn.gui.app import ManipdynGUI

    app = QApplication.instance() or QApplication([])
    win = ManipdynGUI()

    # Every controller is selectable and exposes gain fields.
    assert win.cb_ctrl.count() == len(CONTROLLERS)
    win.cb_ctrl.setCurrentText("pid")
    assert set(win.gain_fields) == {"kp", "ki", "kd"}
    win.cb_ctrl.setCurrentText("tsid")
    assert "kp" in win.gain_fields

    win.close()
    del win
    del app
