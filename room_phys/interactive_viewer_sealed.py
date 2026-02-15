#!/usr/bin/env python3
"""
interactive_viewer_sealed.py — Viewer for sealed competition modules
=====================================================================
Usage:
    python interactive_viewer_sealed.py --module render_scene_a1b
    python interactive_viewer_sealed.py --module render_scene_a1b --seed_setup 42
"""

import sys, os, argparse, importlib
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QLabel, QDoubleSpinBox, QPushButton, QSlider,
    QGroupBox, QFormLayout, QComboBox, QTextEdit,
)
from PyQt5.QtCore import Qt, QTimer
from pyvistaqt import QtInteractor


class SealedSimWrapper:
    """Wraps a sealed render_scene_XXX module."""
    def __init__(self, mod, func_id, seed_setup=42):
        self.mod = mod
        self.func_id = func_id
        self.seed_setup = seed_setup
        self._render_fn = getattr(mod, f"render_scene_{func_id}")
        self._desc_fn = getattr(mod, f"describe_scene_{func_id}", None)
        # Access internal engine for viewer display (scene geometry + simulation)
        self._engine = getattr(mod, "_engine", None)
        self._load_scene()

    def _load_scene(self):
        if self._engine:
            self.scene = self._engine._generate_scene(self.seed_setup)
            # Extract seed_g from the render closure (internal, not public API)
            render_fn = self.mod._W.get("render")
            seed_g = None
            if render_fn and hasattr(render_fn, "__closure__") and render_fn.__closure__:
                for cell in render_fn.__closure__:
                    try:
                        v = cell.cell_contents
                        if isinstance(v, int):
                            seed_g = v
                            break
                    except ValueError:
                        pass
            if seed_g is not None:
                self._grav = self._engine._generate_gravity(seed_g)
            else:
                self._grav = None
        else:
            self.scene = {"balls": [], "ramps": [], "ground_color": [0.85]*3}
            self._grav = None

    def regenerate(self, seed_setup):
        self.seed_setup = seed_setup
        self._load_scene()

    def get_ball_positions(self, t):
        """Compute positions via internal engine (not exposed to participants)."""
        if self._engine and self._grav:
            positions = []
            for ball in self.scene["balls"]:
                pos = self._engine._simulate_ball(
                    ball, self._grav, t, self.scene["ramps"])
                positions.append(pos)
            return positions
        return []

    def describe(self):
        if self._desc_fn:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self._desc_fn(self.seed_setup)
            return buf.getvalue()
        return f"Scene seed_setup={self.seed_setup}"


class MainWindow(QMainWindow):
    def __init__(self, mod, func_id, seed_setup=42):
        super().__init__()
        self.setWindowTitle(f"Sealed Viewer — render_scene_{func_id}")
        self.resize(1500, 950)

        self.sim = SealedSimWrapper(mod, func_id, seed_setup)
        self.current_time = 0.0
        self.is_playing = False
        self.time_step = 0.02
        self.speed = 1.0

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_step)
        self.cam_timer = QTimer()
        self.cam_timer.timeout.connect(self.update_camera_info)
        self.cam_timer.start(200)

        self.setup_ui()
        QTimer.singleShot(200, self.delayed_init)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(central)
        self.plotter.set_background([0.80, 0.85, 0.90])
        self.plotter.enable_anti_aliasing()
        layout.addWidget(self.plotter, stretch=4)

        panel = QFrame()
        panel.setFixedWidth(360)
        ctrl = QVBoxLayout(panel)
        layout.addWidget(panel, stretch=0)

        # Camera
        grp_cam = QGroupBox("Camera (live)")
        cl = QVBoxLayout()
        self.lbl_cam = QLabel("...")
        self.lbl_cam.setStyleSheet("font-family:monospace;font-size:10px;")
        self.lbl_cam.setWordWrap(True)
        cl.addWidget(self.lbl_cam)
        btn_bird = QPushButton("Bird's Eye")
        btn_bird.clicked.connect(self.set_bird_view)
        cl.addWidget(btn_bird)
        grp_cam.setLayout(cl)
        ctrl.addWidget(grp_cam)

        # Time
        grp_time = QGroupBox("Time Control")
        vt = QVBoxLayout()
        self.lbl_time = QLabel("t = 0.00 s")
        self.lbl_time.setStyleSheet("font-size:14px;font-weight:bold;")
        vt.addWidget(self.lbl_time)
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, 3000)
        self.slider_time.valueChanged.connect(self.on_slider_change)
        vt.addWidget(self.slider_time)

        row_spd = QHBoxLayout()
        row_spd.addWidget(QLabel("Speed:"))
        self.combo_speed = QComboBox()
        for lbl, v in [("0.25x",0.25),("0.5x",0.5),("1x",1.0),
                       ("2x",2.0),("4x",4.0)]:
            self.combo_speed.addItem(lbl, v)
        self.combo_speed.setCurrentIndex(2)
        self.combo_speed.currentIndexChanged.connect(
            lambda: setattr(self, 'speed', self.combo_speed.currentData()))
        row_spd.addWidget(self.combo_speed)
        vt.addLayout(row_spd)

        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setStyleSheet("font-weight:bold;padding:6px;")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_row.addWidget(self.btn_play)
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.on_reset)
        btn_row.addWidget(btn_reset)
        btn_step = QPushButton("+0.1s")
        btn_step.clicked.connect(self.on_step)
        btn_row.addWidget(btn_step)
        vt.addLayout(btn_row)
        grp_time.setLayout(vt)
        ctrl.addWidget(grp_time)

        # Scene seed
        grp_seed = QGroupBox("Scene")
        fs = QFormLayout()
        self.spin_seed = QDoubleSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setDecimals(0)
        self.spin_seed.setSingleStep(1)
        self.spin_seed.setValue(self.sim.seed_setup)
        fs.addRow("seed_setup:", self.spin_seed)
        btn_regen = QPushButton("Load Scene")
        btn_regen.clicked.connect(self.regenerate_scene)
        fs.addRow(btn_regen)
        grp_seed.setLayout(fs)
        ctrl.addWidget(grp_seed)

        # Info
        grp_info = QGroupBox("Scene Info")
        il = QVBoxLayout()
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setStyleSheet("font-family:monospace;font-size:10px;")
        self.txt_info.setMaximumHeight(280)
        il.addWidget(self.txt_info)
        grp_info.setLayout(il)
        ctrl.addWidget(grp_info)

        ctrl.addStretch()

    def delayed_init(self):
        self.render_scene()
        self.update_balls()
        self.plotter.add_light(pv.Light(position=(0,0,50),
                                        show_actor=False, intensity=0.8))
        self.plotter.add_light(pv.Light(light_type='headlight', intensity=0.5))
        self.set_bird_view()
        self.update_info()
        self.plotter.render()

    def render_scene(self):
        self.plotter.clear()
        scene = self.sim.scene
        gc = scene.get("ground_color", [0.85]*3)
        if isinstance(gc, np.ndarray):
            gc = gc.tolist()

        ground = pv.Plane(center=(0,0,0), direction=(0,0,1),
                          i_size=200, j_size=200,
                          i_resolution=200, j_resolution=200)
        self.plotter.add_mesh(ground, color=gc, show_edges=True,
                              edge_color='#d0d0d0')

        for rm in scene.get("ramps", []):
            pts = rm.ramp_vertices()
            faces = np.hstack([
                [4,0,1,5,4],[3,0,3,4],[3,1,2,5],
                [4,0,1,2,3],[4,3,2,5,4],
            ])
            self.plotter.add_mesh(pv.PolyData(pts, faces), color=rm.color,
                                  show_edges=True, edge_color=[0.65]*3)

    def update_balls(self):
        positions = self.sim.get_ball_positions(self.current_time)
        balls = self.sim.scene.get("balls", [])

        for i in range(max(len(balls), 5)):
            for pfx in (f"ball_{i}", f"ind_{i}", f"shadow_{i}"):
                try: self.plotter.remove_actor(pfx)
                except: pass

        for i, (pos, ball) in enumerate(zip(positions, balls)):
            if pos is None:
                continue
            sphere = pv.Sphere(radius=ball.radius, center=pos,
                               theta_resolution=32, phi_resolution=32)
            self.plotter.add_mesh(sphere, color=ball.color,
                                  smooth_shading=True, specular=0.4,
                                  name=f"ball_{i}")
            d = ball.radius
            if ball.side == +1:
                p1 = np.array([3.0, -d, 0.1])
                p2 = np.array([3.0,  d, 0.1])
            else:
                p1 = np.array([-3.0, -d, 0.1])
                p2 = np.array([-3.0,  d, 0.1])
            self.plotter.add_mesh(pv.Line(p1, p2), color='red',
                                  line_width=3, name=f"ind_{i}")
            shadow = pv.Disc(center=(pos[0], pos[1], 0.05),
                             normal=(0,0,1), inner=0, outer=ball.radius,
                             r_res=1, c_res=36)
            self.plotter.add_mesh(shadow, color=[0.25]*3, opacity=0.55,
                                  lighting=False, name=f"shadow_{i}")

    def update_info(self):
        self.txt_info.setPlainText(self.sim.describe())

    def update_camera_info(self):
        try:
            cam = self.plotter.camera
            pos = cam.position; fp = cam.focal_point
            look = np.array(fp) - np.array(pos)
            ln = np.linalg.norm(look)
            if ln > 1e-9: look /= ln
            az = np.degrees(np.arctan2(look[1], look[0]))
            el = np.degrees(np.arcsin(np.clip(look[2], -1, 1)))
            self.lbl_cam.setText(
                f"pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                f"az: {az:.1f}  el: {el:.1f}  fov: {cam.view_angle:.0f}")
        except: pass

    def set_bird_view(self):
        self.plotter.camera.position = (0, -60, 40)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.render()

    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("Play")
            self.is_playing = False
        else:
            self.timer.start(30)
            self.btn_play.setText("Pause")
            self.is_playing = True

    def animate_step(self):
        self.current_time += self.time_step * self.speed
        if self.current_time > 30.0:
            self.current_time = 0.0
        self.slider_time.blockSignals(True)
        self.slider_time.setValue(int(self.current_time * 100))
        self.slider_time.blockSignals(False)
        self.lbl_time.setText(f"t = {self.current_time:.2f} s")
        self.update_balls()

    def on_slider_change(self, v):
        self.current_time = v / 100.0
        self.lbl_time.setText(f"t = {self.current_time:.2f} s")
        if not self.is_playing:
            self.update_balls()

    def on_reset(self):
        if self.is_playing: self.toggle_play()
        self.current_time = 0.0
        self.slider_time.setValue(0)
        self.lbl_time.setText("t = 0.00 s")
        self.update_balls()

    def on_step(self):
        if self.is_playing: return
        self.current_time += 0.1
        self.slider_time.setValue(int(self.current_time * 100))
        self.lbl_time.setText(f"t = {self.current_time:.2f} s")
        self.update_balls()

    def regenerate_scene(self):
        if self.is_playing: self.toggle_play()
        self.current_time = 0.0
        self.slider_time.setValue(0)
        self.lbl_time.setText("t = 0.00 s")
        self.sim.regenerate(int(self.spin_seed.value()))
        self.render_scene()
        self.update_balls()
        self.update_info()
        self.plotter.add_light(pv.Light(position=(0,0,50),
                                        show_actor=False, intensity=0.8))
        self.plotter.add_light(pv.Light(light_type='headlight', intensity=0.5))
        self.set_bird_view()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", type=str, required=True,
                    help="Sealed module, e.g. render_scene_a1b")
    ap.add_argument("--seed_setup", type=int, default=42)
    args = ap.parse_args()

    mod_name = args.module
    func_id = mod_name.replace("render_scene_", "")
    sys.path.insert(0, ".")
    mod = importlib.import_module(mod_name)

    app = QApplication(sys.argv)
    window = MainWindow(mod, func_id, args.seed_setup)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
