
import sys
import json
import subprocess
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTabWidget, QGroupBox, QLabel, QLineEdit, QPushButton, 
                               QFormLayout, QMessageBox, QCheckBox, QRadioButton, QButtonGroup, QComboBox, QListWidget, QListWidgetItem, 
                               QScrollArea, QSizePolicy, QGridLayout, QSplitter, QFrame, QStatusBar)
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush, QPainterPath

import scene_utils

# --- MAP PREVIEW WIDGET ---
class MapPreview(QWidget):
    def __init__(self, obstacles):
        super().__init__()
        self.obstacles = obstacles
        self.target_pos = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3d3d3d; border-radius: 4px;")

    def set_target(self, x, y):
        self.target_pos = (x, y)
        self.update()

    def set_waypoints(self, waypoints):
        """
        waypoints: list of (x, y) tuples
        """
        self.waypoints_list = waypoints
        self.update()

    def update_map(self):
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        scale = min(w, h) / 4.0 # View +/- 2m coverage
        cx, cy = w/2, h/2
        
        # Background Grid (Pro Look)
        qp.fillRect(0, 0, w, h, QColor(30, 30, 30))
        
        pen_grid = QPen(QColor(60, 60, 60))
        pen_grid.setStyle(Qt.DotLine)
        qp.setPen(pen_grid)
        
        # Draw Grid Lines
        grid_step = 50
        for x in range(0, w, grid_step): qp.drawLine(x, 0, x, h)
        for y in range(0, h, grid_step): qp.drawLine(0, y, w, y)
        
        # Origin Axis
        qp.setPen(QPen(QColor(100, 100, 100), 2))
        qp.drawLine(cx, 0, cx, h)
        qp.drawLine(0, cy, w, cy)
        
        # Robot Base (Center)
        qp.setBrush(QBrush(QColor(0, 229, 255))) # Cyan
        qp.setPen(Qt.NoPen)
        qp.drawEllipse(cx-6, cy-6, 12, 12)
        
        # Obstacles
        for o in self.obstacles:
            c_name = o.get('color', 'Gray')
            # Custom Palette
            if c_name == 'Red': qc = QColor(220, 50, 50, 180)
            elif c_name == 'Green': qc = QColor(50, 220, 50, 180)
            elif c_name == 'Blue': qc = QColor(50, 80, 220, 180)
            else: qc = QColor(150, 150, 150, 180)
            
            qp.setBrush(QBrush(qc))
            qp.setPen(QPen(Qt.white, 1))
            
            x, y = o.get('x', 0), o.get('y', 0)
            px = cx + x * scale
            py = cy - y * scale # Invert Y for screen coords
            
            t = o.get('type', 'box')
            if t == 'box':
                dx = o.get('dx', 0.1) * 2 * scale 
                dy = o.get('dy', 0.1) * 2 * scale
                qp.drawRect(px - dx/2, py - dy/2, dx, dy)
            elif t in ['sphere', 'cylinder']:
                r = o.get('r', 0.1) * scale
                qp.drawEllipse(px - r, py - r, r*2, r*2)

        # Target Marker
        if self.target_pos:
            tx, ty = self.target_pos
            px = cx + tx * scale
            py = cy - ty * scale
            
            # Glow Effect
            qp.setPen(Qt.NoPen)
            qp.setBrush(QBrush(QColor(255, 215, 0, 80)))
            qp.drawEllipse(px - 8, py - 8, 16, 16)
            
            qp.setBrush(QBrush(QColor(255, 215, 0)))
            qp.drawEllipse(px - 4, py - 4, 8, 8)

        # Draw Waypoints Path
        if hasattr(self, 'waypoints_list') and self.waypoints_list:
            # Draw lines
            pen_path = QPen(QColor(0, 255, 0, 150), 2, Qt.DashLine)
            qp.setPen(pen_path)
            
            wps_px = []
            for (wx, wy) in self.waypoints_list:
                px = cx + wx * scale
                py = cy - wy * scale
                wps_px.append(QPointF(px, py))
            
            # Lines
            if len(wps_px) > 0:
                path = QPainterPath(QPointF(cx, cy)) # Start from center
                if len(wps_px) > 0: path.moveTo(wps_px[0]) # Or logic to connect seq
                
                # Let's draw sequence: Start(Center?) -> WP1 -> WP2
                # Actually user sets Start Pos. I don't have start pos here easily.
                # Just connect WPs.
                path = QPainterPath()
                if len(wps_px) > 0:
                    path.moveTo(wps_px[0])
                    for pt in wps_px[1:]:
                        path.lineTo(pt)
                qp.drawPath(path)
                
            # Dots for WPs
            qp.setPen(Qt.NoPen)
            qp.setBrush(QBrush(QColor(0, 255, 0)))
            for i, pt in enumerate(wps_px):
                qp.drawEllipse(pt, 4, 4)
                # qp.drawText(pt, f"{i+1}")

# --- PRO GUI ---
class ProManipulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manipulator Control Center [PRO]")
        self.setGeometry(100, 50, 1200, 800)
        
        self.obstacles = []
        # self.obstacles.append({'type': 'box', 'x': 0.3, 'y': -0.1, 'z': 0.4, 'dx': 0.05, 'dy': 0.05, 'dz': 0.2, 'color': 'Blue'})
        
        self.setup_theme()
        self.init_ui()
        
    def setup_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(25, 25, 25))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
        palette.setColor(QPalette.Highlight, QColor(0, 180, 255))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI'; font-size: 10pt; }
            QGroupBox { 
                border: 1px solid #444; 
                border-radius: 6px; 
                margin-top: 20px; 
                font-weight: bold; 
                color: #00b4ff; 
                background-color: #2b2b2b;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; }
            QLineEdit, QComboBox { 
                background-color: #333; 
                border: 1px solid #555; 
                color: white; 
                padding: 5px; 
                border-radius: 3px;
            }
            QLineEdit:focus { border: 1px solid #00b4ff; }
            QListWidget { background-color: #333; border: 1px solid #555; border-radius: 4px; padding: 5px; }
            QPushButton { 
                background-color: #444; 
                border: 1px solid #555; 
                padding: 8px 16px; 
                color: white; 
                border-radius: 4px; 
            }
            QPushButton:hover { background-color: #555; border: 1px solid #666; }
            QPushButton:pressed { background-color: #333; }
            QPushButton#actionBtn { 
                background-color: #007acc; 
                border: 1px solid #007acc;
                font-weight: bold; 
            }
            QPushButton#actionBtn:hover { background-color: #008be6; }
            QTabWidget::pane { border: 0px; background: #252525; }
            QTabBar::tab { 
                background: #333; 
                color: #aaa; 
                padding: 10px 24px; 
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #444; color: #00b4ff; font-weight: bold; }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel("MANIPULATOR CONTROL CENTER")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setStyleSheet("color: #00b4ff; padding: 5px;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Main Content
        self.tabs = QTabWidget()
        self.tab_dashboard = QWidget()
        self.tab_control = QWidget()
        
        self.tabs.addTab(self.tab_dashboard, "DASHBOARD (ENV & MAP)")
        self.tabs.addTab(self.tab_control, "CONTROL & TUNING")
        self.tab_task = QWidget()
        self.tabs.addTab(self.tab_task, "TASK SEQUENCER")
        
        main_layout.addWidget(self.tabs)
        
        self.build_dashboard_tab()
        self.build_control_tab()
        self.build_task_tab()
        
        # Footer
        footer = QFrame()
        footer.setStyleSheet("background-color: #2b2b2b; border-top: 1px solid #444;")
        f_layout = QHBoxLayout(footer)
        
        self.status = QLabel("Ready")
        self.status.setStyleSheet("color: #888;")
        
        btn_launch = QPushButton("LAUNCH SIMULATION")
        btn_launch.setObjectName("actionBtn")
        btn_launch.clicked.connect(self.launch_sim)
        
        f_layout.addWidget(self.status)
        f_layout.addStretch()
        f_layout.addWidget(btn_launch)
        
        main_layout.addWidget(footer)

    # --- TAB 1: DASHBOARD (Environment + List + Map) ---
    def build_dashboard_tab(self):
        layout = QHBoxLayout(self.tab_dashboard)
        
        # Splitter for adjustable width
        splitter = QSplitter(Qt.Horizontal)
        
        # LEFT PANEL: Controls & List
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)
        
        # 1. Obstacle List
        gb_list = QGroupBox("Scene Objects")
        l_list = QVBoxLayout()
        self.obs_list = QListWidget()
        self.obs_list.setSelectionMode(QListWidget.SingleSelection)
        l_list.addWidget(self.obs_list)
        
        # Delete Button (Under list)
        btn_del = QPushButton("Delete Selected")
        btn_del.setStyleSheet("color: #ff6666;")
        btn_del.clicked.connect(self.del_obstacle)
        l_list.addWidget(btn_del)
        
        gb_list.setLayout(l_list)
        left_layout.addWidget(gb_list)
        
        # 2. Add New Obstacle
        gb_add = QGroupBox("Add Object")
        l_add = QFormLayout()
        
        self.cb_type = QComboBox(); self.cb_type.addItems(["Box", "Sphere", "Cylinder"])
        self.cb_color = QComboBox(); self.cb_color.addItems(["Red", "Green", "Blue", "Gray"])
        self.cb_type.currentTextChanged.connect(self.update_dims_ui)
        
        # Compact Pos
        h_pos = QHBoxLayout()
        self.ent_px = QLineEdit("0.5"); self.ent_px.setPlaceholderText("X")
        self.ent_py = QLineEdit("0.0"); self.ent_py.setPlaceholderText("Y")
        self.ent_pz = QLineEdit("0.4"); self.ent_pz.setPlaceholderText("Z")
        h_pos.addWidget(self.ent_px); h_pos.addWidget(self.ent_py); h_pos.addWidget(self.ent_pz)
        
        # Dynamic Dims
        self.dim_stack = QWidget()
        self.l_dim = QHBoxLayout(self.dim_stack); self.l_dim.setContentsMargins(0,0,0,0)
        self.ent_d1 = QLineEdit("0.1"); self.ent_d1.setPlaceholderText("L")
        self.ent_d2 = QLineEdit("0.1"); self.ent_d2.setPlaceholderText("W")
        self.ent_d3 = QLineEdit("0.1"); self.ent_d3.setPlaceholderText("H")
        self.l_dim.addWidget(self.ent_d1); self.l_dim.addWidget(self.ent_d2); self.l_dim.addWidget(self.ent_d3)
        
        l_add.addRow("Type / Color:", self.create_hbox([self.cb_type, self.cb_color]))
        l_add.addRow("Position:", h_pos)
        l_add.addRow("Dimensions:", self.dim_stack)
        
        btn_add = QPushButton("Add to Scene")
        btn_add.clicked.connect(self.add_obstacle)
        l_add.addRow(btn_add)
        
        gb_add.setLayout(l_add)
        left_layout.addWidget(gb_add)
        
        # RIGHT PANEL: Map Preview
        right_panel = QGroupBox("Top-Down Preview")
        r_layout = QVBoxLayout(right_panel)
        self.map_preview = MapPreview(self.obstacles)
        r_layout.addWidget(self.map_preview)
        
        # Add to Splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 2) # Map is bigger
        
        layout.addWidget(splitter)
        
        self.refresh_obs_list()
        self.update_dims_ui()

    # --- TAB 2: CONTROL ---
    def build_control_tab(self):
        layout = QHBoxLayout(self.tab_control)
        
        # Column 1: Planning & Target
        col1 = QWidget()
        v1 = QVBoxLayout(col1)
        
        #        # Planner
        gb_plan = QGroupBox("Path Planning")
        v_plan = QVBoxLayout()
        self.rb_plan_direct = QRadioButton("Direct (No Plan)")
        self.rb_plan_rrt = QRadioButton("RRT (Basic)")
        self.rb_plan_rrt_star = QRadioButton("RRT* (Optimal)")
        self.rb_plan_prm = QRadioButton("PRM (Roadmap)")
        self.chk_smooth = QCheckBox("Smooth Path")
        self.chk_smooth.setChecked(True)
        
        self.rb_plan_direct.setChecked(True)
        
        bg_plan = QButtonGroup()
        bg_plan.addButton(self.rb_plan_direct)
        bg_plan.addButton(self.rb_plan_rrt)
        bg_plan.addButton(self.rb_plan_rrt_star)
        bg_plan.addButton(self.rb_plan_prm)
        
        v_plan.addWidget(self.rb_plan_direct)
        v_plan.addWidget(self.rb_plan_rrt)
        v_plan.addWidget(self.rb_plan_rrt_star)
        v_plan.addWidget(self.rb_plan_prm)
        v_plan.addWidget(self.chk_smooth)
        gb_plan.setLayout(v_plan)
        v1.addWidget(gb_plan)
        
        # 2. Points Management (Start & Waypoints)
        gb_points = QGroupBox("2. Trajectory Points")
        l_points = QVBoxLayout()
        
        # Start Point
        h_start = QHBoxLayout()
        h_start.addWidget(QLabel("Start Pos:"))
        self.ent_sx = QLineEdit("0.4"); self.ent_sy = QLineEdit("-0.4"); self.ent_sz = QLineEdit("0.4")
        h_start.addWidget(self.ent_sx); h_start.addWidget(self.ent_sy); h_start.addWidget(self.ent_sz)
        l_points.addLayout(h_start)
        
        # Waypoints List
        self.wp_list = QListWidget()
        self.wp_list.setMaximumHeight(80)
        l_points.addWidget(self.wp_list)
        
        # Current Target Input (to add as waypoint)
        h_curr = QHBoxLayout()
        h_curr.addWidget(QLabel("New WP:"))
        self.ent_tx = QLineEdit("0.5"); self.ent_tx.textChanged.connect(self.update_map_target)
        self.ent_ty = QLineEdit("0.2"); self.ent_ty.textChanged.connect(self.update_map_target)
        self.ent_tz = QLineEdit("0.5")
        h_curr.addWidget(self.ent_tx); h_curr.addWidget(self.ent_ty); h_curr.addWidget(self.ent_tz)
        l_points.addLayout(h_curr)
        
        # Point Buttons
        h_pbtns = QHBoxLayout()
        btn_add_wp = QPushButton("Add WP")
        btn_add_wp.clicked.connect(self.add_waypoint)
        btn_del_wp = QPushButton("Delete WP")
        btn_del_wp.clicked.connect(self.del_waypoint)
        btn_clear_wp = QPushButton("Clear WPs")
        btn_clear_wp.clicked.connect(self.wp_list.clear)
        h_pbtns.addWidget(btn_add_wp); h_pbtns.addWidget(btn_del_wp); h_pbtns.addWidget(btn_clear_wp)
        l_points.addLayout(h_pbtns)
        
        # Workspace Info
        lbl_info = QLabel("Workspace Limit: Max Radius = 0.85m\n(Max X/Y/Z approx ±0.85m)")
        lbl_info.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        lbl_info.setAlignment(Qt.AlignCenter)
        l_points.addWidget(lbl_info)
        
        gb_points.setLayout(l_points)
        v1.addWidget(gb_points)
        
        # Show Marker Toggle
        self.chk_show_target = QRadioButton("Show Target Markers")
        self.chk_show_target.setChecked(True)
        self.chk_show_target.setAutoExclusive(False)
        v1.addWidget(self.chk_show_target)
        
        v1.addStretch()
        
        # Column 2: Controller & Tuning
        col2 = QWidget()
        v2 = QVBoxLayout(col2)
        
        # 3. Feedback Controller
        gb_ctrl = QGroupBox("3. Feedback Controller")
        v_ctrl = QVBoxLayout()
        self.rb_ctrl_lqr = QRadioButton("Joint LQR")
        self.rb_ctrl_pid = QRadioButton("Joint PID")
        self.rb_ctrl_ctc = QRadioButton("Joint CTC")
        self.rb_ctrl_mpc = QRadioButton("Joint MPC (MPPI)")
        self.rb_ctrl_imp = QRadioButton("Cartesian Impedance")
        self.rb_ctrl_osc = QRadioButton("OSC (Task Space)")
        self.rb_ctrl_ik = QRadioButton("IK (Geometric)")
        
        self.rb_ctrl_lqr.setChecked(True)
        
        v_ctrl.addWidget(self.rb_ctrl_lqr)
        v_ctrl.addWidget(self.rb_ctrl_pid)
        v_ctrl.addWidget(self.rb_ctrl_ctc)
        v_ctrl.addWidget(self.rb_ctrl_mpc)
        v_ctrl.addWidget(self.rb_ctrl_imp)
        v_ctrl.addWidget(self.rb_ctrl_osc)
        v_ctrl.addWidget(self.rb_ctrl_ik)
        gb_ctrl.setLayout(v_ctrl)
        v2.addWidget(gb_ctrl)
        
        # 4. Gains
        gb_lqr = QGroupBox("Controller Gains")
        f_lqr = QFormLayout()
        self.ent_q_pos = QLineEdit("1000.0")
        self.ent_q_vel = QLineEdit("10.0")
        self.ent_r = QLineEdit("1.0")
        f_lqr.addRow("Position Cost (Q):", self.ent_q_pos)
        f_lqr.addRow("Velocity Cost (Q):", self.ent_q_vel)
        f_lqr.addRow("Control Effort (R):", self.ent_r)
        gb_lqr.setLayout(f_lqr)
        v2.addWidget(gb_lqr)
        
        layout.addWidget(col1)
        layout.addWidget(col2)

    def add_waypoint(self):
        try:
            x, y, z = float(self.ent_tx.text()), float(self.ent_ty.text()), float(self.ent_tz.text())
            
            # Workspace Validation (UR5e Reach ~0.85m)
            dist = (x**2 + y**2 + z**2)**0.5
            if dist > 0.85:
                QMessageBox.warning(self, "Workspace Error", 
                                    f"Point ({x}, {y}, {z}) is outside the robot's workspace (Max Radius: 0.85m).\nDistance: {dist:.2f}m")
                return

            self.wp_list.addItem(f"{x}, {y}, {z}")
            self.refresh_preview_waypoints()
            
        except ValueError:
            pass

    def refresh_preview_waypoints(self):
        # Extract XY for map preview
        wps = []
        for i in range(self.wp_list.count()):
            txt = self.wp_list.item(i).text()
            parts = [float(v) for v in txt.split(',')]
            wps.append((parts[0], parts[1]))
        self.map_preview.set_waypoints(wps)

    def del_waypoint(self):
        row = self.wp_list.currentRow()
        if row >= 0:
            self.wp_list.takeItem(row)
            self.refresh_preview_waypoints()
        else:
            QMessageBox.warning(self, "Selection Error", "Select a Waypoint to delete.")

    def update_ctrl_options(self):
        if self.rb_plan_rrt.isChecked() or self.rb_plan_rrt_star.isChecked() or self.rb_plan_prm.isChecked():
            # RRT/PRM produces joint path -> LQR or PID or CTC or MPC
            self.rb_ctrl_lqr.setChecked(True)
            self.rb_ctrl_osc.setEnabled(False)
            self.rb_ctrl_ik.setEnabled(False)
            self.rb_ctrl_imp.setEnabled(False) 
            self.rb_ctrl_pid.setEnabled(True)
            self.rb_ctrl_ctc.setEnabled(True)
            self.rb_ctrl_mpc.setEnabled(True)
            self.status.setText("Mode: RRT requires Joint Control (LQR/PID/CTC/MPC)")
        else:
            self.rb_ctrl_osc.setEnabled(True)
            self.rb_ctrl_ik.setEnabled(True)
            self.rb_ctrl_imp.setEnabled(True)
            self.rb_ctrl_pid.setEnabled(True)
            self.rb_ctrl_ctc.setEnabled(True)
            self.rb_ctrl_mpc.setEnabled(True)
            self.status.setText("Mode: Direct Control")

    # --- HELPER LOGIC ---
    def create_hbox(self, widgets):
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0,0,0,0)
        for wi in widgets: l.addWidget(wi)
        return w

    def update_dims_ui(self):
        t = self.cb_type.currentText()
        if t == "Box":
            self.ent_d1.setPlaceholderText("Length")
            self.ent_d2.setPlaceholderText("Width")
            self.ent_d3.setPlaceholderText("Height")
            self.ent_d2.show(); self.ent_d3.show()
        elif t == "Sphere":
            self.ent_d1.setPlaceholderText("Radius")
            self.ent_d2.hide(); self.ent_d3.hide()
        elif t == "Cylinder":
            self.ent_d1.setPlaceholderText("Radius")
            self.ent_d2.setPlaceholderText("Height")
            self.ent_d2.show(); self.ent_d3.hide()

    def refresh_obs_list(self):
        self.obs_list.clear()
        for i, o in enumerate(self.obstacles):
            t = o['type'].title()
            c = o['color']
            pos = f"({o['x']:.2f}, {o['y']:.2f})"
            self.obs_list.addItem(f"{i+1}. {c} {t} at {pos}")
        self.map_preview.update_map()

    def add_obstacle(self):
        try:
            obs = {
                'type': self.cb_type.currentText().lower(),
                'color': self.cb_color.currentText(),
                'x': float(self.ent_px.text()),
                'y': float(self.ent_py.text()),
                'z': float(self.ent_pz.text())
            }
            # Dims
            t = obs['type']
            if t == 'box':
                obs['dx'] = float(self.ent_d1.text())
                obs['dy'] = float(self.ent_d2.text())
                obs['dz'] = float(self.ent_d3.text())
            elif t == 'sphere':
                obs['r'] = float(self.ent_d1.text())
            elif t == 'cylinder':
                obs['r'] = float(self.ent_d1.text())
                obs['h'] = float(self.ent_d2.text())
            
            self.obstacles.append(obs)
            self.refresh_obs_list()
            self.status.setText(f"Added {t} at {obs['x']}, {obs['y']}")
            
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid numeric values for position and dimensions.")

    def del_obstacle(self):
        row = self.obs_list.currentRow()
        if row == -1:
            QMessageBox.warning(self, "Selection Error", "Please select an object from the list to delete.")
            return
            
        removed_item = self.obstacles.pop(row)
        self.refresh_obs_list()
        self.status.setText(f"Removed {removed_item['type']}")

    def build_task_tab(self):
        layout = QHBoxLayout(self.tab_task)
        
        # LEFT: Task List
        gb_list = QGroupBox("Task Sequence")
        v_list = QVBoxLayout()
        self.task_list_widget = QListWidget()
        v_list.addWidget(self.task_list_widget)
        
        # Controls for List
        h_list_ctrl = QHBoxLayout()
        btn_up = QPushButton("Up")
        btn_down = QPushButton("Down")
        btn_del = QPushButton("Delete")
        btn_clear = QPushButton("Clear")
        
        btn_up.clicked.connect(self.move_task_up)
        btn_down.clicked.connect(self.move_task_down)
        btn_del.clicked.connect(self.del_task)
        btn_clear.clicked.connect(self.task_list_widget.clear)
        
        h_list_ctrl.addWidget(btn_up)
        h_list_ctrl.addWidget(btn_down)
        h_list_ctrl.addWidget(btn_del)
        h_list_ctrl.addWidget(btn_clear)
        v_list.addLayout(h_list_ctrl)
        
        gb_list.setLayout(v_list)
        
        # RIGHT: Add Tasks
        gb_add = QGroupBox("Add Steps")
        v_add = QVBoxLayout()
        
        # 1. Move Step
        gb_move = QGroupBox("Move Action")
        f_move = QFormLayout()
        self.ent_tm_x = QLineEdit("0.4")
        self.ent_tm_y = QLineEdit("-0.4")
        self.ent_tm_z = QLineEdit("0.4")
        f_move.addRow("X:", self.ent_tm_x)
        f_move.addRow("Y:", self.ent_tm_y)
        f_move.addRow("Z:", self.ent_tm_z)
        btn_add_move = QPushButton("Add Move Task")
        btn_add_move.setObjectName("actionBtn")
        btn_add_move.clicked.connect(self.add_task_move)
        
        v_add.addWidget(gb_move)
        gb_move.setLayout(f_move)
        v_add.addWidget(btn_add_move)
        
        # 2. Gripper Step
        gb_grip = QGroupBox("Gripper Action")
        h_grip = QHBoxLayout()
        btn_grip_open = QPushButton("Open")
        btn_grip_close = QPushButton("Close")
        btn_grip_open.clicked.connect(lambda: self.add_task_gripper(0.0))
        btn_grip_close.clicked.connect(lambda: self.add_task_gripper(1.0))
        h_grip.addWidget(btn_grip_open)
        h_grip.addWidget(btn_grip_close)
        gb_grip.setLayout(h_grip)
        v_add.addWidget(gb_grip)
        
        # 3. Sleep Step
        gb_sleep = QGroupBox("Wait Action")
        h_sleep = QHBoxLayout()
        self.ent_sleep = QLineEdit("1.0")
        btn_add_sleep = QPushButton("Add Wait (s)")
        btn_add_sleep.clicked.connect(self.add_task_sleep)
        h_sleep.addWidget(self.ent_sleep)
        h_sleep.addWidget(btn_add_sleep)
        gb_sleep.setLayout(h_sleep)
        v_add.addWidget(gb_sleep)
        
        v_add.addStretch()
        
        gb_add.setLayout(v_add)
        
        layout.addWidget(gb_list, 2)
        layout.addWidget(gb_add, 1)
        
    def add_task_move(self):
        try:
            x, y, z = float(self.ent_tm_x.text()), float(self.ent_tm_y.text()), float(self.ent_tm_z.text())
            txt = f"MOVE -> [{x}, {y}, {z}]"
            # Store data in item
            item = QListWidgetItem(txt)
            item.setData(Qt.UserRole, {"action": "move", "target": [x, y, z]})
            self.task_list_widget.addItem(item)
        except: pass
        
    def add_task_gripper(self, val):
        txt = f"GRIPPER -> {'CLOSE' if val > 0.5 else 'OPEN'}"
        item = QListWidgetItem(txt)
        item.setData(Qt.UserRole, {"action": "gripper", "value": val})
        self.task_list_widget.addItem(item)
        
    def add_task_sleep(self):
        try:
            val = float(self.ent_sleep.text())
            txt = f"WAIT -> {val}s"
            item = QListWidgetItem(txt)
            item.setData(Qt.UserRole, {"action": "sleep", "value": val})
            self.task_list_widget.addItem(item)
        except: pass
        
    def del_task(self):
        row = self.task_list_widget.currentRow()
        if row != -1: self.task_list_widget.takeItem(row)
        
    def move_task_up(self):
        row = self.task_list_widget.currentRow()
        if row > 0:
            item = self.task_list_widget.takeItem(row)
            self.task_list_widget.insertItem(row-1, item)
            self.task_list_widget.setCurrentRow(row-1)
            
    def move_task_down(self):
        row = self.task_list_widget.currentRow()
        if row < self.task_list_widget.count()-1 and row != -1:
            item = self.task_list_widget.takeItem(row)
            self.task_list_widget.insertItem(row+1, item)
            self.task_list_widget.setCurrentRow(row+1)

    def update_map_target(self):
        try:
            x, y = float(self.ent_tx.text()), float(self.ent_ty.text())
            self.map_preview.set_target(x, y)
        except: pass

    def launch_sim(self):
        self.status.setText("Generating Scene and Launching...")
        QApplication.processEvents()
        
        xml_path = scene_utils.generate_dynamic_scene(self.obstacles)
        
        # Collect Waypoints
        waypoints = []
        for i in range(self.wp_list.count()):
            txt = self.wp_list.item(i).text()
            waypoints.append([float(v) for v in txt.split(',')])
            
        # If no waypoints, use single target as one waypoint
        if not waypoints:
            waypoints.append([float(self.ent_tx.text()), float(self.ent_ty.text()), float(self.ent_tz.text())])
            
        # Collect Tasks
        tasks = []
        if hasattr(self, 'task_list_widget'):
            for i in range(self.task_list_widget.count()):
                item = self.task_list_widget.item(i)
                data = item.data(Qt.UserRole)
                if data: tasks.append(data)
            
        config = {
            "planner": "prm" if self.rb_plan_prm.isChecked() else ("rrt_star" if self.rb_plan_rrt_star.isChecked() else ("rrt" if self.rb_plan_rrt.isChecked() else "direct")),
            "smooth": self.chk_smooth.isChecked(),
            "controller": "osc" if self.rb_ctrl_osc.isChecked() else ("ik" if self.rb_ctrl_ik.isChecked() else ("pid" if self.rb_ctrl_pid.isChecked() else ("ctc" if self.rb_ctrl_ctc.isChecked() else ("mpc" if self.rb_ctrl_mpc.isChecked() else ("impedance" if self.rb_ctrl_imp.isChecked() else "lqr"))))),
            "start_pos": [float(self.ent_sx.text()), float(self.ent_sy.text()), float(self.ent_sz.text())],
            "waypoints": waypoints,
            "tasks": tasks,
            "show_target": self.chk_show_target.isChecked(),
            "lqr": {
                "q_pos": float(self.ent_q_pos.text()),
                "q_vel": float(self.ent_q_vel.text()),
                "r": float(self.ent_r.text())
            },
            "xml_path": xml_path
        }
        
        with open("sim_config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # Always run mujoco_sim.py as it now handles everything (IK/OSC/LQR/RRT)
        # Wait, osc_sim.py was specific for OSC test.
        # But user wants "waypoints". `mujoco_sim.py` is best suited for sequencing.
        # I should merge OSC logic into `mujoco_sim.py` or keep switching?
        # Let's direct ALL traffic to `mujoco_sim.py` and implement the controller switching there.
        # This simplifies the architecture significantly.
        
        script = "mujoco_sim.py"
        
        try:
            subprocess.Popen([sys.executable, script])
            self.status.setText(f"Running {script}...")
        except Exception as e:
            self.status.setText("Launch Failed")
            QMessageBox.critical(self, "Launch Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProManipulatorGUI()
    window.show()
    sys.exit(app.exec())
