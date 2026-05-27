"""manipdyn control center (PySide6).

A QTimer drives the simulation on the main thread (so the MuJoCo GL context is
never touched cross-thread): each tick advances several physics steps, renders
one offscreen frame into the embedded view, and updates the live error plot.
The long-running benchmark runs in a background QThread.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from manipdyn.control import CONTROLLERS, Target
from manipdyn.kinematics import IKSolver
from manipdyn.models import AVAILABLE_SCENES
from manipdyn.planning import PLANNERS, shortcut_path
from manipdyn.sim import World
from manipdyn.trajectory import parameterize_time_optimal
from manipdyn.tuning import TUNE_SPECS, tuned_params

_STYLE = """
QWidget { font-family: 'Segoe UI'; font-size: 10pt; color: #e6e6e6; }
QMainWindow, QWidget#central { background: #1e1e1e; }
QGroupBox { border: 1px solid #3d3d3d; border-radius: 6px; margin-top: 16px;
            font-weight: bold; color: #00b4ff; background: #2b2b2b; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QLineEdit, QComboBox { background: #333; border: 1px solid #555; padding: 4px; border-radius: 3px; }
QLineEdit:focus { border: 1px solid #00b4ff; }
QPushButton { background: #444; border: 1px solid #555; padding: 7px 14px; border-radius: 4px; }
QPushButton:hover { background: #555; }
QPushButton#run { background: #007acc; border: 1px solid #007acc; font-weight: bold; }
QPushButton#run:hover { background: #008be6; }
QPushButton#stop { background: #b5392f; border: 1px solid #b5392f; font-weight: bold; }
"""


class LivePlot(QWidget):
    """Lightweight QPainter plot of end-effector error over time."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(130)
        self._data: list[float] = []

    def push(self, value: float) -> None:
        self._data.append(value)
        self._data = self._data[-600:]
        self.update()

    def clear(self) -> None:
        self._data = []
        self.update()

    def paintEvent(self, _event) -> None:
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor(30, 30, 30))
        qp.setPen(QPen(QColor(70, 70, 70), 1))
        for gy in range(1, 4):
            y = h * gy / 4
            qp.drawLine(0, int(y), w, int(y))
        if len(self._data) < 2:
            qp.setPen(QColor(150, 150, 150))
            qp.drawText(8, 18, "EE error (mm) — run a simulation")
            return
        vmax = max(self._data) or 1.0
        pen = QPen(QColor(0, 200, 120), 2)
        qp.setPen(pen)
        n = len(self._data)
        for i in range(1, n):
            x0 = (i - 1) / (n - 1) * w
            x1 = i / (n - 1) * w
            y0 = h - self._data[i - 1] / vmax * (h - 10) - 5
            y1 = h - self._data[i] / vmax * (h - 10) - 5
            qp.drawLine(int(x0), int(y0), int(x1), int(y1))
        qp.setPen(QColor(180, 180, 180))
        qp.drawText(8, 18, f"EE error (mm)  now={self._data[-1]:.1f}  max={vmax:.1f}")


class BenchmarkWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def run(self) -> None:
        try:
            from manipdyn.benchmark import benchmark_controllers
            from manipdyn.benchmark.report import markdown_table

            rows = benchmark_controllers(duration=2.0)
            self.done.emit(markdown_table(rows))
        except Exception as exc:  # pragma: no cover - GUI path
            self.failed.emit(str(exc))


class ManipdynGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("manipdyn — Control Center")
        self.resize(1180, 720)

        self.world: World | None = None
        self.controller = None
        self.renderer: mujoco.Renderer | None = None
        self.target: Target | None = None
        self.goal_x: np.ndarray | None = None
        self.timed = None
        self._t_traj = 0.0
        self.gain_fields: dict[str, QLineEdit] = {}

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._build_ui()
        self._on_controller_changed(self.cb_ctrl.currentText())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.setStyleSheet(_STYLE)
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: controls
        left = QVBoxLayout()
        root.addLayout(left, 0)

        gb_setup = QGroupBox("Setup")
        f = QFormLayout(gb_setup)
        self.cb_scene = QComboBox()
        self.cb_scene.addItems(list(AVAILABLE_SCENES))
        self.cb_ctrl = QComboBox()
        self.cb_ctrl.addItems(list(CONTROLLERS))
        self.cb_ctrl.currentTextChanged.connect(self._on_controller_changed)
        self.cb_plan = QComboBox()
        self.cb_plan.addItems(["none", *PLANNERS])
        self.chk_tuned = QCheckBox("Use tuned gains")
        self.chk_tuned.setChecked(True)
        self.chk_tuned.stateChanged.connect(
            lambda _=0: self._on_controller_changed(self.cb_ctrl.currentText())
        )
        f.addRow("Scene:", self.cb_scene)
        f.addRow("Controller:", self.cb_ctrl)
        f.addRow("Planner:", self.cb_plan)
        f.addRow(self.chk_tuned)
        left.addWidget(gb_setup)

        self.gb_gains = QGroupBox("Gains")
        self.gains_form = QFormLayout(self.gb_gains)
        left.addWidget(self.gb_gains)

        gb_target = QGroupBox("Cartesian target (m)")
        ft = QFormLayout(gb_target)
        self.ent_x = QLineEdit("0.45")
        self.ent_y = QLineEdit("0.0")
        self.ent_z = QLineEdit("0.4")
        ft.addRow("X:", self.ent_x)
        ft.addRow("Y:", self.ent_y)
        ft.addRow("Z:", self.ent_z)
        left.addWidget(gb_target)

        row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.setObjectName("run")
        self.btn_run.clicked.connect(self.start_sim)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stop")
        self.btn_stop.clicked.connect(self.stop_sim)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_stop)
        left.addLayout(row)

        self.btn_bench = QPushButton("Run benchmark (controllers)")
        self.btn_bench.clicked.connect(self.run_benchmark)
        left.addWidget(self.btn_bench)
        left.addStretch()

        # Right: live view + telemetry
        right = QVBoxLayout()
        root.addLayout(right, 1)
        header = QLabel("MANIPDYN CONTROL CENTER")
        header.setFont(QFont("Segoe UI", 15, QFont.Bold))
        header.setStyleSheet("color: #00b4ff;")
        header.setAlignment(Qt.AlignCenter)
        right.addWidget(header)

        self.view = QLabel()
        self.view.setMinimumSize(640, 460)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background: #111; border: 1px solid #3d3d3d; border-radius: 4px;")
        self.view.setText("Press Run to start the simulation")
        right.addWidget(self.view, 1)

        self.plot = LivePlot()
        right.addWidget(self.plot)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

    def _on_controller_changed(self, name: str) -> None:
        # Rebuild gain fields for the selected controller from its tuning spec.
        while self.gains_form.rowCount():
            self.gains_form.removeRow(0)
        self.gain_fields = {}
        spec = TUNE_SPECS.get(name)
        if not spec:
            return
        tuned = tuned_params(name) if self.chk_tuned.isChecked() else None
        for gain, (lo, hi) in spec.space.items():
            default = tuned[gain] if tuned and gain in tuned else round((lo + hi) / 2, 3)
            field = QLineEdit(str(default))
            self.gain_fields[gain] = field
            self.gains_form.addRow(f"{gain}:", field)

    # -------------------------------------------------------------- actions
    def _read_gains(self) -> dict[str, float]:
        out = {}
        for gain, field in self.gain_fields.items():
            try:
                out[gain] = float(field.text())
            except ValueError:
                pass
        return out

    def start_sim(self) -> None:
        try:
            self.stop_sim()
            scene = self.cb_scene.currentText()
            ctrl_name = self.cb_ctrl.currentText()
            self.world = World(scene=scene)
            self.world.reset(self.world.home_qpos_arm)
            self.renderer = mujoco.Renderer(self.world.model, height=460, width=640)

            goal = np.array(
                [float(self.ent_x.text()), float(self.ent_y.text()), float(self.ent_z.text())]
            )
            self.goal_x = goal
            self.world.set_target_marker(goal)

            ik = IKSolver(self.world)
            q_goal = ik.solve(goal, q_guess=self.world.home_qpos_arm).q

            spec = TUNE_SPECS[ctrl_name]
            self.controller = spec.factory(self.world, **self._read_gains())
            self.controller.reset()

            self.timed = None
            self._t_traj = 0.0
            plan_name = self.cb_plan.currentText()
            if plan_name != "none" and spec.target_space == "joint":
                self._plan_path(plan_name, q_goal)

            self.target = Target(q=q_goal, x=goal)
            self.plot.clear()
            self.timer.start(33)  # ~30 Hz
            self.status.showMessage(f"Running {ctrl_name} on {scene}...")
        except Exception as exc:
            QMessageBox.critical(self, "Start failed", str(exc))

    def _plan_path(self, plan_name: str, q_goal: np.ndarray) -> None:
        planner = PLANNERS[plan_name](self.world, seed=0)
        path = planner.plan(self.world.qpos_arm, q_goal)
        if path is None:
            self.status.showMessage("Planner found no path; moving directly.")
            return
        path = shortcut_path(path, planner.checker, iterations=100, seed=0)
        self.timed = parameterize_time_optimal(path, np.full(6, 1.5), np.full(6, 3.0))

    def _tick(self) -> None:
        if self.world is None or self.controller is None:
            return
        for _ in range(16):  # advance ~real-time per 33 ms tick
            if self.timed is not None:
                tt = min(self._t_traj, self.timed.duration)
                q = np.array([np.interp(tt, self.timed.t, self.timed.q[:, j]) for j in range(6)])
                v = np.array([np.interp(tt, self.timed.t, self.timed.qd[:, j]) for j in range(6)])
                self.target.q, self.target.v = q, v
                self._t_traj += self.world.timestep
            self.world.step(self.controller.compute(self.target))

        self.renderer.update_scene(self.world.data)
        frame = self.renderer.render()
        img = QImage(
            frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format_RGB888
        )
        self.view.setPixmap(QPixmap.fromImage(img).scaled(self.view.size(), Qt.KeepAspectRatio))

        err_mm = float(np.linalg.norm(self.goal_x - self.world.ee_pos) * 1e3)
        self.plot.push(err_mm)
        self.status.showMessage(f"t={self.world.time:5.2f}s   EE error={err_mm:6.1f} mm")

    def stop_sim(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.status.showMessage("Stopped")

    def run_benchmark(self) -> None:
        self.btn_bench.setEnabled(False)
        self.status.showMessage("Benchmarking controllers (background)...")
        self._bw = BenchmarkWorker()
        self._bw.done.connect(self._benchmark_done)
        self._bw.failed.connect(lambda e: QMessageBox.critical(self, "Benchmark failed", e))
        self._bw.finished.connect(lambda: self.btn_bench.setEnabled(True))
        self._bw.start()

    def _benchmark_done(self, table: str) -> None:
        self.status.showMessage("Benchmark complete")
        box = QMessageBox(self)
        box.setWindowTitle("Controller benchmark")
        box.setText("Results (tuned gains, reach scenarios):")
        box.setDetailedText(table)
        box.exec()

    def closeEvent(self, event) -> None:
        self.stop_sim()
        super().closeEvent(event)


def launch() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(30, 30, 30))
    pal.setColor(QPalette.WindowText, QColor(230, 230, 230))
    app.setPalette(pal)
    win = ManipdynGUI()
    win.show()
    app.exec()


if __name__ == "__main__":
    launch()
