"""manipdyn control center — a mode-based PySide6 desktop app.

Pick a **mode** (Reach, Obstacle Avoidance, Pick & Place, RL Reach, Benchmark),
configure it, then either:

  * **Watch Sim** — opens the interactive MuJoCo viewer *and* an embedded live
    view, stepping physics on the main thread (so the GL context is never
    touched cross-thread) while streaming live telemetry; or
  * **Run Sim** — runs the same scenario headless in a background thread and
    reports results/plots (no viewer), at faster-than-real-time.

Heavy setup (IK, planning, grasp solving, policy loading) runs in a worker so
the UI never freezes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from manipdyn.control import ComputedTorqueController, Target
from manipdyn.kinematics import IKSolver
from manipdyn.planning import PLANNERS, shortcut_path
from manipdyn.sim import World
from manipdyn.tasks import pick_place
from manipdyn.trajectory import parameterize_time_optimal
from manipdyn.tuning import TUNE_SPECS, tuned_params

# ----------------------------------------------------------------------- theme
ACCENT = "#4c8dff"
GOOD = "#22c55e"
DANGER = "#ef4444"
_STYLE = f"""
* {{ font-family: 'Segoe UI', sans-serif; }}
QWidget {{ color: #e6e8ec; font-size: 10pt; }}
QMainWindow, QWidget#root {{ background: #0f1115; }}
QLabel#h1 {{ font-size: 17pt; font-weight: 800; color: #ffffff; }}
QLabel#sub {{ color: #9aa3b2; font-size: 9pt; }}
QLabel#cardTitle {{ color: {ACCENT}; font-weight: 700; font-size: 8.5pt;
                    letter-spacing: 1px; }}
QLabel#statName {{ color: #9aa3b2; font-size: 8pt; letter-spacing: 1px; }}
QLabel#statValue {{ color: #ffffff; font-size: 16pt; font-weight: 700; }}
QFrame#card {{ background: #1a1d24; border: 1px solid #2c313c; border-radius: 10px; }}
QFrame#stat {{ background: #161922; border: 1px solid #2c313c; border-radius: 8px; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: #0f1115; border: 1px solid #2c313c; border-radius: 6px; padding: 5px 7px; }}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox QAbstractItemView {{ background: #1a1d24; selection-background-color: {ACCENT}; }}
QCheckBox {{ spacing: 8px; }}
QPushButton {{ background: #232733; border: 1px solid #333a48; padding: 9px 14px;
               border-radius: 8px; font-weight: 600; }}
QPushButton:hover {{ background: #2b3140; }}
QPushButton:disabled {{ color: #5b6270; background: #171a21; }}
QPushButton#mode {{ background: transparent; border: 1px solid #2c313c; padding: 8px 6px;
                    color: #9aa3b2; }}
QPushButton#mode:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: #ffffff; }}
QPushButton#watch {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: #fff; }}
QPushButton#watch:hover {{ background: #5d9bff; }}
QPushButton#run {{ background: {GOOD}; border: 1px solid {GOOD}; color: #05210f; }}
QPushButton#run:hover {{ background: #2fd36b; }}
QPushButton#stop {{ background: #2a1416; border: 1px solid {DANGER}; color: #ff9b9b; }}
QTextEdit {{ background: #0c0e12; border: 1px solid #2c313c; border-radius: 8px;
             font-family: 'Consolas', monospace; font-size: 9pt; color: #cfd6e4; }}
QSlider::groove:horizontal {{ height: 5px; background: #2c313c; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: #ffffff; width: 14px; margin: -6px 0;
                              border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QStatusBar {{ color: #9aa3b2; }}
"""


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(14, 12, 14, 14)
    outer.setSpacing(8)
    lab = QLabel(title.upper())
    lab.setObjectName("cardTitle")
    outer.addWidget(lab)
    return frame, outer


class AxisSlider(QWidget):
    """A labeled draggable slider over a float range, in metres."""

    changed = Signal()

    def __init__(self, label: str, lo: float, hi: float, default: float):
        super().__init__()
        self._lo, self._hi = lo, hi
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        name = QLabel(label)
        name.setFixedWidth(58)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(int((default - lo) / (hi - lo) * 1000))
        self.slider.valueChanged.connect(self._on)
        self.readout = QLabel()
        self.readout.setFixedWidth(56)
        self.readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(name)
        row.addWidget(self.slider, 1)
        row.addWidget(self.readout)
        self._update()

    def _on(self) -> None:
        self._update()
        self.changed.emit()

    def _update(self) -> None:
        self.readout.setText(f"{self.value():.2f} m")

    def value(self) -> float:
        return self._lo + self.slider.value() / 1000 * (self._hi - self._lo)


# ------------------------------------------------------------------- live plot
class LivePlot(QWidget):
    """Lightweight error-vs-time plot drawn with QPainter."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self._data: list[float] = []
        self._label = "error"

    def configure(self, label: str) -> None:
        self._label = label

    def push(self, value: float) -> None:
        self._data.append(value)
        self._data = self._data[-1200:]
        self.update()

    def set_series(self, values: list[float]) -> None:
        self._data = list(values)[-1200:]
        self.update()

    def clear(self) -> None:
        self._data = []
        self.update()

    def paintEvent(self, _event) -> None:
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor(12, 14, 18))
        qp.setPen(QPen(QColor(40, 45, 56), 1))
        for gy in range(1, 4):
            qp.drawLine(0, int(h * gy / 4), w, int(h * gy / 4))
        if len(self._data) < 2:
            qp.setPen(QColor(120, 128, 140))
            qp.drawText(10, 20, f"{self._label} — run a simulation")
            return
        vmax = max(self._data) or 1.0
        qp.setPen(QPen(QColor(76, 141, 255), 2))
        n = len(self._data)
        for i in range(1, n):
            x0, x1 = (i - 1) / (n - 1) * w, i / (n - 1) * w
            y0 = h - self._data[i - 1] / vmax * (h - 16) - 8
            y1 = h - self._data[i] / vmax * (h - 16) - 8
            qp.drawLine(int(x0), int(y0), int(x1), int(y1))
        qp.setPen(QColor(154, 163, 178))
        qp.drawText(10, 20, f"{self._label}   now={self._data[-1]:.2f}   peak={vmax:.2f}")


# ----------------------------------------------------------------------- engines
# Each engine builds its scenario in prepare() (heavy work) and advances one
# control step per step(), returning telemetry with at least {t, err_mm, phase}.

_OBS_START = np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0])
_OBS_GOAL = np.array([-1.4, -1.2, 1.4, -1.7, -1.57, 0.0])
_PLANNER_KW = {
    "rrt": {"max_iter": 6000, "goal_bias": 0.2},
    "rrt_connect": {"max_iter": 5000},
    "rrt_star": {"max_iter": 3000, "goal_bias": 0.2},
    "informed_rrt_star": {"max_iter": 3000, "goal_bias": 0.2},
    "prm": {"n_samples": 500, "k_neighbors": 15},
}


class ReachEngine:
    metric = "EE error (mm)"
    steps_per_tick = 16

    def __init__(self, scene, ctrl_name, gains, target):
        self.scene, self.ctrl_name, self.gains = scene, ctrl_name, gains
        self.target_xyz = np.asarray(target, float)

    def prepare(self):
        self.world = World(scene=self.scene)
        self.world.reset(self.world.home_qpos_arm)
        self.world.set_target_marker(self.target_xyz)
        q_goal = IKSolver(self.world).solve(self.target_xyz, q_guess=self.world.home_qpos_arm).q
        self.ctrl = TUNE_SPECS[self.ctrl_name].factory(self.world, **self.gains)
        self.ctrl.reset()
        self.target = Target(q=q_goal, x=self.target_xyz)
        self._k, self._n, self._errs = 0, int(4.0 / self.world.timestep), []

    def step(self):
        self.world.step(self.ctrl.compute(self.target))
        self._k += 1
        err = float(np.linalg.norm(self.target_xyz - self.world.ee_pos) * 1e3)
        self._errs.append(err)
        return {"t": float(self.world.time), "err_mm": err, "phase": self.ctrl_name}

    def retarget(self, xyz) -> None:
        """Move the goal live (used when the target slider is dragged mid-run)."""
        self.target_xyz = np.asarray(xyz, float)
        self.world.set_target_marker(self.target_xyz)
        q = IKSolver(self.world).solve(self.target_xyz, q_guess=self.world.qpos_arm).q
        self.target.q, self.target.x = q, self.target_xyz

    def done(self):
        return self._k >= self._n

    def summary(self):
        e = np.array(self._errs)
        return (
            f"controller: {self.ctrl_name}\n"
            f"final EE error: {e[-1]:.3f} mm\n"
            f"RMS error:      {np.sqrt(np.mean(e**2)):.3f} mm\n"
            f"min error:      {e.min():.3f} mm"
        )

    @property
    def series(self):
        return self._errs


class ObstacleEngine:
    metric = "EE error to goal (mm)"
    steps_per_tick = 16

    def __init__(self, planner_name, obstacle_xy=None):
        self.planner_name = planner_name
        self.obstacle_xy = obstacle_xy

    def prepare(self):
        self.world = World(scene="scene_obstacle")
        if self.obstacle_xy is not None:
            gid = mujoco.mj_name2id(self.world.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
            self.world.model.geom_pos[gid][:2] = self.obstacle_xy
        self.world.reset(_OBS_START)
        planner = PLANNERS[self.planner_name](
            self.world, seed=0, **_PLANNER_KW.get(self.planner_name, {})
        )
        path = planner.plan(_OBS_START, _OBS_GOAL)
        if path is None:
            raise RuntimeError(f"{self.planner_name} found no path for this query")
        path = shortcut_path(path, planner.checker, iterations=150, seed=0)
        self.timed = parameterize_time_optimal(
            path, np.full(6, 1.2), np.full(6, 2.5), n_samples=200
        )
        self.ctrl = ComputedTorqueController(self.world, kp=600, kd=50)
        self.ctrl.reset()
        self.world.set_arm_qpos(_OBS_GOAL)
        self.world.forward()
        self.goal_x = self.world.ee_pos.copy()
        self.world.reset(_OBS_START)
        self.target = Target()
        self._errs = []

    def step(self):
        tt = min(self.world.time, self.timed.duration)
        q = np.array([np.interp(tt, self.timed.t, self.timed.q[:, j]) for j in range(6)])
        v = np.array([np.interp(tt, self.timed.t, self.timed.qd[:, j]) for j in range(6)])
        self.target.q, self.target.v = q, v
        self.world.step(self.ctrl.compute(self.target))
        err = float(np.linalg.norm(self.goal_x - self.world.ee_pos) * 1e3)
        self._errs.append(err)
        return {"t": float(self.world.time), "err_mm": err, "phase": self.planner_name}

    def done(self):
        return self.world.time > self.timed.duration + 0.6

    def summary(self):
        return (
            f"planner: {self.planner_name}\n"
            f"path duration: {self.timed.duration:.2f} s\n"
            f"final EE error: {self._errs[-1]:.2f} mm\n"
            "collision-free detour over the pillar."
        )

    @property
    def series(self):
        return self._errs


class PickPlaceEngine:
    metric = "cube-to-target (mm)"
    steps_per_tick = 18

    def prepare(self):
        self.world = World(scene=pick_place.SCENE, ee_site=pick_place.EE_SITE)
        self.plan = pick_place.solve(self.world)
        self.gen = pick_place.run(self.world, self.plan)
        self._fin = False
        self._last = {"t": 0.0, "err_mm": 0.0, "phase": "ready", "tilt": 0.0}
        self._errs = []

    def step(self):
        try:
            info = next(self.gen)
            self._last = {
                "t": info["t"],
                "err_mm": info["place_err_mm"],
                "phase": info["phase"],
                "tilt": info["cube_tilt_deg"],
            }
            self._errs.append(info["place_err_mm"])
        except StopIteration:
            self._fin = True
        return self._last

    def done(self):
        return self._fin

    def summary(self):
        return (
            f"phase: {self._last['phase']}\n"
            f"cube-to-target: {self._last['err_mm']:.1f} mm\n"
            f"cube tilt: {self._last['tilt']:.1f} deg\n"
            "top-down grasp, base-rotation carry, stable place."
        )

    @property
    def series(self):
        return self._errs


class RLEngine:
    metric = "EE-to-goal (mm)"
    steps_per_tick = 2

    def __init__(self, seed):
        self.seed = seed

    def prepare(self):
        from stable_baselines3 import SAC

        from manipdyn import rl
        from manipdyn.rl import ReachEnv

        self.env = ReachEnv(seed=self.seed)
        self.model = SAC.load(str(Path(rl.__file__).parent / "sac_reach.zip"))
        self.obs, _ = self.env.reset(seed=self.seed)
        self.world = self.env.world
        self._done, self._errs = False, []

    def step(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, _, term, trunc, info = self.env.step(action)
        self._done = bool(term or trunc)
        err = float(info["distance"] * 1e3)
        self._errs.append(err)
        return {"t": float(self.world.time), "err_mm": err, "phase": "SAC policy"}

    def done(self):
        return self._done

    def summary(self):
        e = np.array(self._errs)
        ok = "reached" if e[-1] < 30 else "did not reach"
        return f"SAC policy\nfinal distance: {e[-1]:.1f} mm ({ok}, 30 mm tol)\nmin distance: {e.min():.1f} mm"

    @property
    def series(self):
        return self._errs


# ------------------------------------------------------------------- workers
class PrepareWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.engine.prepare()
            self.ready.emit(self.engine)
        except Exception as exc:  # pragma: no cover - GUI path
            self.failed.emit(str(exc))


class RunWorker(QThread):
    progress = Signal(dict)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._stop = False

    def run(self):
        try:
            i = 0
            while not self.engine.done() and not self._stop:
                info = self.engine.step()
                if i % 8 == 0:
                    self.progress.emit(info)
                i += 1
            self.done.emit(self.engine)
        except Exception as exc:  # pragma: no cover - GUI path
            self.failed.emit(str(exc))

    def stop(self):
        self._stop = True


class BenchmarkWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, which, duration, trials):
        super().__init__()
        self.which, self.duration, self.trials = which, duration, trials

    def run(self):
        try:
            from manipdyn.benchmark import benchmark_controllers, benchmark_planners
            from manipdyn.benchmark.report import markdown_table, write_report

            cr, pr = [], []
            if self.which in ("controllers", "all"):
                cr = benchmark_controllers(duration=self.duration)
            if self.which in ("planners", "all"):
                pr = benchmark_planners(n_trials=self.trials)
            out = Path("benchmarks/results")
            write_report(cr, pr, out)
            txt = ""
            if cr:
                txt += "CONTROLLERS (tuned gains, reach)\n" + markdown_table(cr) + "\n\n"
            if pr:
                txt += "PLANNERS (blocked obstacle query)\n" + markdown_table(pr) + "\n"
            self.done.emit(txt + f"\nPlots + tables written to {out}/")
        except Exception as exc:  # pragma: no cover - GUI path
            self.failed.emit(str(exc))


# --------------------------------------------------------------------- window
_MODES = ["Reach", "Obstacle Avoidance", "Pick & Place", "RL Reach", "Benchmark"]


class ManipdynGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("manipdyn — Control Center")
        self.resize(1240, 780)

        self.engine = None
        self.renderer: mujoco.Renderer | None = None
        self.viewer = None
        self.preview_world: World | None = None
        self.preview_renderer: mujoco.Renderer | None = None
        self.gain_fields: dict[str, QLineEdit] = {}
        self._pending_watch = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._build_ui()
        self._on_controller_changed(self.cb_ctrl.currentText())
        self._select_mode(0)

    # --------------------------------------------------------------- UI build
    def _build_ui(self) -> None:
        self.setStyleSheet(_STYLE)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 10)
        outer.setSpacing(12)

        # header
        head = QVBoxLayout()
        head.setSpacing(1)
        t = QLabel("manipdyn — Control Center")
        t.setObjectName("h1")
        s = QLabel("6-DOF UR5e · planning · control · learning · benchmark")
        s.setObjectName("sub")
        head.addWidget(t)
        head.addWidget(s)
        outer.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)

        # ---- left column ------------------------------------------------
        left = QVBoxLayout()
        left.setSpacing(12)
        body.addLayout(left, 0)

        mode_card, mode_box = _card("Mode")
        grid = QGridLayout()
        grid.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for i, name in enumerate(_MODES):
            b = QPushButton(name)
            b.setObjectName("mode")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, idx=i: self._select_mode(idx))
            self.mode_group.addButton(b, i)
            grid.addWidget(b, i // 2, i % 2)
        mode_box.addLayout(grid)
        mode_card.setFixedWidth(330)
        left.addWidget(mode_card)

        self.config_stack = QStackedWidget()
        self.config_stack.addWidget(self._panel_reach())
        self.config_stack.addWidget(self._panel_obstacle())
        self.config_stack.addWidget(self._panel_pick())
        self.config_stack.addWidget(self._panel_rl())
        self.config_stack.addWidget(self._panel_bench())
        left.addWidget(self.config_stack)
        left.addStretch()

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.btn_watch = QPushButton("▶  Watch Sim")
        self.btn_watch.setObjectName("watch")
        self.btn_watch.setToolTip("Open the interactive MuJoCo viewer + live telemetry")
        self.btn_watch.clicked.connect(lambda: self._start(watch=True))
        self.btn_run = QPushButton("▶  Run Sim")
        self.btn_run.setObjectName("run")
        self.btn_run.setToolTip("Headless evaluation → results & plots (no viewer)")
        self.btn_run.clicked.connect(lambda: self._start(watch=False))
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("stop")
        self.btn_stop.clicked.connect(self.stop_sim)
        actions.addWidget(self.btn_watch)
        actions.addWidget(self.btn_run)
        actions.addWidget(self.btn_stop)
        left.addLayout(actions)

        # ---- right column ----------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(12)
        body.addLayout(right, 1)

        self.view = QLabel("Configure a mode, then Watch Sim or Run Sim")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(620, 380)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setStyleSheet(
            "background: #0a0c10; border: 1px solid #2c313c; border-radius: 10px; color:#5b6270;"
        )
        right.addWidget(self.view, 1)

        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.stat_time = self._stat("TIME", "0.00 s")
        self.stat_err = self._stat("METRIC", "—")
        self.stat_phase = self._stat("PHASE", "idle")
        stats.addWidget(self.stat_time[0])
        stats.addWidget(self.stat_err[0])
        stats.addWidget(self.stat_phase[0])
        right.addLayout(stats)

        self.plot = LivePlot()
        right.addWidget(self.plot)

        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setFixedHeight(140)
        self.results.setPlainText("Results and benchmark tables appear here.")
        right.addWidget(self.results)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

    def _stat(self, name: str, value: str) -> tuple[QFrame, QLabel]:
        f = QFrame()
        f.setObjectName("stat")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        n = QLabel(name)
        n.setObjectName("statName")
        v = QLabel(value)
        v.setObjectName("statValue")
        lay.addWidget(n)
        lay.addWidget(v)
        return f, v

    # ---- per-mode config panels -----------------------------------------
    def _panel_reach(self) -> QWidget:
        card, box = _card("Reach — controller → Cartesian target")
        form = QFormLayout()
        form.setSpacing(8)
        self.cb_ctrl = QComboBox()
        self.cb_ctrl.addItems(list(TUNE_SPECS))
        self.cb_ctrl.currentTextChanged.connect(self._on_controller_changed)
        self.chk_tuned = QCheckBox("Use tuned gains")
        self.chk_tuned.setChecked(True)
        self.chk_tuned.stateChanged.connect(
            lambda _=0: self._on_controller_changed(self.cb_ctrl.currentText())
        )
        form.addRow("Controller", self.cb_ctrl)
        form.addRow(self.chk_tuned)
        box.addLayout(form)
        self.gains_form = QFormLayout()
        self.gains_form.setSpacing(6)
        box.addLayout(self.gains_form)
        drag = QLabel("Drag to place the target — the red marker moves in-scene:")
        drag.setObjectName("sub")
        box.addWidget(drag)
        self.sl_x = AxisSlider("target X", -0.7, 0.7, 0.45)
        self.sl_y = AxisSlider("target Y", -0.7, 0.7, 0.15)
        self.sl_z = AxisSlider("target Z", 0.2, 0.85, 0.5)
        for s in (self.sl_x, self.sl_y, self.sl_z):
            s.changed.connect(self._on_target_changed)
            box.addWidget(s)
        return card

    def _panel_obstacle(self) -> QWidget:
        card, box = _card("Obstacle avoidance — plan around the pillar")
        form = QFormLayout()
        form.setSpacing(8)
        self.cb_obs_planner = QComboBox()
        self.cb_obs_planner.addItems(list(PLANNERS))
        self.cb_obs_planner.setCurrentText("rrt_connect")
        form.addRow("Planner", self.cb_obs_planner)
        box.addLayout(form)
        drag = QLabel("Drag to place the pillar — it moves in-scene; the planner routes around it:")
        drag.setObjectName("sub")
        drag.setWordWrap(True)
        box.addWidget(drag)
        self.sl_ox = AxisSlider("pillar X", -0.78, -0.36, -0.57)
        self.sl_oy = AxisSlider("pillar Y", 0.10, 0.50, 0.30)
        for s in (self.sl_ox, self.sl_oy):
            s.changed.connect(self._on_obstacle_changed)
            box.addWidget(s)
        return card

    def _panel_pick(self) -> QWidget:
        card, box = _card("Pick & Place — cube from one table to another")
        note = QLabel(
            "A top-down grasp config is solved, the cube is approached along an "
            "orientation-locked vertical line, gripped (held by a weld), carried "
            "by a 90° base rotation, and placed upright on the second table.\n\n"
            "Computed-torque control throughout. Solving the grasp takes a moment."
        )
        note.setObjectName("sub")
        note.setWordWrap(True)
        box.addWidget(note)
        return card

    def _panel_rl(self) -> QWidget:
        card, box = _card("RL Reach — learned SAC policy")
        form = QFormLayout()
        form.setSpacing(8)
        self.sp_rl_seed = QSpinBox()
        self.sp_rl_seed.setRange(0, 9999)
        self.sp_rl_seed.setValue(1)
        form.addRow("goal seed", self.sp_rl_seed)
        box.addLayout(form)
        note = QLabel(
            "The shipped Soft Actor-Critic policy drives the arm to a randomly "
            "sampled goal on the same physics as the classical controllers. "
            "Requires the 'rl' extra (stable-baselines3)."
        )
        note.setObjectName("sub")
        note.setWordWrap(True)
        box.addWidget(note)
        return card

    def _panel_bench(self) -> QWidget:
        card, box = _card("Benchmark — score every method")
        form = QFormLayout()
        form.setSpacing(8)
        self.cb_bench_which = QComboBox()
        self.cb_bench_which.addItems(["controllers", "planners", "all"])
        self.sp_bench_dur = QDoubleSpinBox()
        self.sp_bench_dur.setRange(1.0, 6.0)
        self.sp_bench_dur.setValue(3.0)
        self.sp_bench_dur.setSingleStep(0.5)
        self.sp_bench_trials = QSpinBox()
        self.sp_bench_trials.setRange(1, 10)
        self.sp_bench_trials.setValue(5)
        form.addRow("which", self.cb_bench_which)
        form.addRow("controller duration (s)", self.sp_bench_dur)
        form.addRow("planner trials", self.sp_bench_trials)
        box.addLayout(form)
        note = QLabel(
            "Runs headless and writes tables + comparison plots to "
            "benchmarks/results/. 'Watch' is not used for this mode."
        )
        note.setObjectName("sub")
        note.setWordWrap(True)
        box.addWidget(note)
        return card

    # ---- mode / controller switching ------------------------------------
    def _select_mode(self, idx: int) -> None:
        self._teardown_sim()
        self.mode_group.button(idx).setChecked(True)
        self.config_stack.setCurrentIndex(idx)
        is_bench = _MODES[idx] == "Benchmark"
        self.btn_watch.setEnabled(not is_bench)
        self.btn_watch.setToolTip(
            "Benchmark runs headless only" if is_bench else "Interactive viewer + live telemetry"
        )
        self.plot.clear()
        self._build_preview()
        self.status.showMessage(f"Mode: {_MODES[idx]}")

    def _on_controller_changed(self, name: str) -> None:
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
            self.gains_form.addRow(f"{gain}", field)

    def _read_gains(self) -> dict[str, float]:
        out = {}
        for gain, field in self.gain_fields.items():
            try:
                out[gain] = float(field.text())
            except ValueError:
                pass
        return out

    # ---- engine construction --------------------------------------------
    def _make_engine(self):
        mode = _MODES[self.config_stack.currentIndex()]
        if mode == "Reach":
            tgt = [self.sl_x.value(), self.sl_y.value(), self.sl_z.value()]
            return ReachEngine("scene_base", self.cb_ctrl.currentText(), self._read_gains(), tgt)
        if mode == "Obstacle Avoidance":
            return ObstacleEngine(
                self.cb_obs_planner.currentText(), [self.sl_ox.value(), self.sl_oy.value()]
            )
        if mode == "Pick & Place":
            return PickPlaceEngine()
        if mode == "RL Reach":
            return RLEngine(self.sp_rl_seed.value())
        return None  # Benchmark handled separately

    # ---- start / stop ----------------------------------------------------
    def _start(self, watch: bool) -> None:
        self._teardown_sim()
        if _MODES[self.config_stack.currentIndex()] == "Benchmark":
            self._run_benchmark()
            return
        self._clear_preview()
        try:
            engine = self._make_engine()
        except Exception as exc:
            self._fail(str(exc))
            return
        self._pending_watch = watch
        self._set_busy(True, "Preparing scenario…")
        self.plot.configure(engine.metric)
        self.plot.clear()
        self.results.setPlainText("")
        self._prep = PrepareWorker(engine)
        self._prep.ready.connect(self._on_prepared)
        self._prep.failed.connect(self._fail)
        self._prep.start()

    def _on_prepared(self, engine) -> None:
        self.engine = engine
        self.stat_err[1].setText("—")
        try:
            self.renderer = mujoco.Renderer(engine.world.model, height=400, width=620)
        except Exception as exc:
            self._fail(f"renderer: {exc}")
            return
        if self._pending_watch:
            try:
                self.viewer = mujoco.viewer.launch_passive(engine.world.model, engine.world.data)
                self.status.showMessage("Watching — interactive viewer open. Close it or Stop.")
            except Exception:
                self.viewer = None
                self.status.showMessage("Watching (embedded view; interactive viewer unavailable).")
            self.timer.start(33)
        else:
            self.status.showMessage("Running headless…")
            self._rw = RunWorker(engine)
            self._rw.progress.connect(self._on_progress)
            self._rw.done.connect(self._on_run_done)
            self._rw.failed.connect(self._fail)
            self._rw.start()

    def _tick(self) -> None:
        if self.engine is None:
            return
        info = {}
        for _ in range(self.engine.steps_per_tick):
            info = self.engine.step()
            if self.engine.done():
                break
        self._render_embedded()
        if self.viewer is not None:
            if not self.viewer.is_running():
                self.stop_sim()
                return
            self.viewer.sync()
        if info:
            self._update_telemetry(info)
            self.plot.push(info["err_mm"])
        if self.engine.done():
            self.timer.stop()
            self._finish()

    def _render_embedded(self) -> None:
        self.renderer.update_scene(self.engine.world.data)
        frame = self.renderer.render()
        img = QImage(
            frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format_RGB888
        )
        self.view.setPixmap(QPixmap.fromImage(img).scaled(self.view.size(), Qt.KeepAspectRatio))

    def _on_progress(self, info: dict) -> None:
        self._update_telemetry(info)

    def _on_run_done(self, engine) -> None:
        self.plot.set_series(engine.series)
        self._render_embedded()
        self._finish()

    def _finish(self) -> None:
        summary = self.engine.summary() if self.engine is not None else ""
        self._teardown_sim()
        if summary:
            self.results.setPlainText(summary)
        self._build_preview()
        self._set_busy(False, "Done")

    def _update_telemetry(self, info: dict) -> None:
        self.stat_time[1].setText(f"{info.get('t', 0):.2f} s")
        self.stat_err[1].setText(f"{info.get('err_mm', 0):.2f}")
        self.stat_phase[1].setText(str(info.get("phase", "—")))

    def _teardown_sim(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        rw = getattr(self, "_rw", None)
        if rw is not None and rw.isRunning():
            rw.stop()
            rw.wait(2000)
        self._close_viewer()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.engine = None

    def stop_sim(self) -> None:
        self._teardown_sim()
        self._build_preview()
        self._set_busy(False, "Stopped")

    def _close_viewer(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    # ---- idle preview (draggable placement) -----------------------------
    def _build_preview(self) -> None:
        """Show a static, draggable preview of the current scene when idle."""
        self._clear_preview()
        mode = _MODES[self.config_stack.currentIndex()]
        try:
            if mode == "Reach":
                w = World(scene="scene_base")
                w.reset(w.home_qpos_arm)
                w.set_target_marker([self.sl_x.value(), self.sl_y.value(), self.sl_z.value()])
            elif mode == "Obstacle Avoidance":
                w = World(scene="scene_obstacle")
                gid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
                w.model.geom_pos[gid][:2] = [self.sl_ox.value(), self.sl_oy.value()]
                w.reset(_OBS_START)
            else:
                self.view.setPixmap(QPixmap())
                self.view.setText("Configure, then Watch Sim or Run Sim")
                return
            self.preview_world = w
            self.preview_renderer = mujoco.Renderer(w.model, height=400, width=620)
            self._render_preview()
        except Exception:
            self._clear_preview()

    def _render_preview(self) -> None:
        if self.preview_renderer is None or self.preview_world is None:
            return
        self.preview_renderer.update_scene(self.preview_world.data)
        frame = self.preview_renderer.render()
        img = QImage(
            frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format_RGB888
        )
        self.view.setPixmap(QPixmap.fromImage(img).scaled(self.view.size(), Qt.KeepAspectRatio))

    def _clear_preview(self) -> None:
        if self.preview_renderer is not None:
            self.preview_renderer.close()
            self.preview_renderer = None
        self.preview_world = None

    def _on_target_changed(self) -> None:
        xyz = [self.sl_x.value(), self.sl_y.value(), self.sl_z.value()]
        if self.timer.isActive() and isinstance(self.engine, ReachEngine):
            self.engine.retarget(xyz)  # live re-targeting during Watch
        elif self.preview_world is not None:
            self.preview_world.set_target_marker(xyz)
            self._render_preview()

    def _on_obstacle_changed(self) -> None:
        if self.preview_world is not None and not self.timer.isActive():
            gid = mujoco.mj_name2id(self.preview_world.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
            self.preview_world.model.geom_pos[gid][:2] = [self.sl_ox.value(), self.sl_oy.value()]
            self.preview_world.forward()
            self._render_preview()

    # ---- benchmark mode --------------------------------------------------
    def _run_benchmark(self) -> None:
        self._set_busy(True, "Benchmarking… (this can take a few minutes)")
        self.results.setPlainText("Benchmarking…")
        self._bw = BenchmarkWorker(
            self.cb_bench_which.currentText(),
            self.sp_bench_dur.value(),
            self.sp_bench_trials.value(),
        )
        self._bw.done.connect(self._on_bench_done)
        self._bw.failed.connect(self._fail)
        self._bw.start()

    def _on_bench_done(self, table: str) -> None:
        self.results.setPlainText(table)
        self._set_busy(False, "Benchmark complete")

    # ---- helpers ---------------------------------------------------------
    def _set_busy(self, busy: bool, msg: str) -> None:
        self.btn_run.setEnabled(not busy)
        is_bench = _MODES[self.config_stack.currentIndex()] == "Benchmark"
        self.btn_watch.setEnabled(not busy and not is_bench)
        for b in self.mode_group.buttons():
            b.setEnabled(not busy)
        self.status.showMessage(msg)

    def _fail(self, msg: str) -> None:
        self.stop_sim()
        self.results.setPlainText(f"⚠ {msg}")
        self.status.showMessage(f"Failed: {msg}")

    def closeEvent(self, event) -> None:
        self._teardown_sim()
        self._clear_preview()
        super().closeEvent(event)


def launch() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(15, 17, 21))
    pal.setColor(QPalette.WindowText, QColor(230, 232, 236))
    app.setPalette(pal)
    QApplication.setFont(QFont("Segoe UI", 10))
    win = ManipdynGUI()
    win.show()
    app.exec()


if __name__ == "__main__":
    launch()
