#!/usr/bin/env python3
"""
interactive_viewer.py  —  Interactive 3D Physics Viewer (v2)
=============================================================
Shows detailed scene info: ball vel0, pos0, ramp geometry.
Displays camera parameters (position, focal, azimuth, elevation)
live as user rotates the 3D view.

Requirements:  pip install pyvista pyvistaqt PyQt5 numpy
"""

import sys, os, argparse
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QLabel, QDoubleSpinBox, QPushButton, QSlider,
    QGroupBox, QFormLayout, QComboBox, QTextEdit,
)
from PyQt5.QtCore import Qt, QTimer
from pyvistaqt import QtInteractor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics_sim


class SimulationWrapper:
    def __init__(self, seed_setup=42, seed_g=7):
        self.seed_setup = seed_setup
        self.seed_g = seed_g
        self.scene = None
        self.grav = None
        self.regenerate()

    def regenerate(self, s_setup=None, s_g=None):
        if s_setup is not None: self.seed_setup = s_setup
        if s_g is not None:     self.seed_g = s_g
        self.scene = physics_sim._generate_scene(self.seed_setup)
        self.grav = physics_sim._generate_gravity(self.seed_g)

    def get_ball_positions(self, t):
        return [physics_sim._simulate_ball(b, self.grav, t, self.scene['ramps'])
                for b in self.scene['balls']]

    def gravity_str(self):
        g = self.grav
        return f"g(z) = {g.g0:.3f} + {g.alpha:.3f}/(z+10)"


class MainWindow(QMainWindow):
    def __init__(self, seed_setup=42, seed_g=7):
        super().__init__()
        self.setWindowTitle("Physics Simulation - Interactive Viewer v2")
        self.resize(1500, 950)

        self.sim = SimulationWrapper(seed_setup, seed_g)
        self.current_time = 0.0
        self.is_playing = False
        self.time_step = 0.02
        self.speed = 1.0
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_step)

        # Camera poll timer (updates camera info display)
        self.cam_timer = QTimer()
        self.cam_timer.timeout.connect(self.update_camera_info)
        self.cam_timer.start(200)  # 5 Hz

        self.setup_ui()
        QTimer.singleShot(200, self.delayed_init)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left: 3D view
        self.plotter = QtInteractor(central)
        self.plotter.set_background([0.80, 0.85, 0.90])
        self.plotter.enable_anti_aliasing()
        main_layout.addWidget(self.plotter, stretch=4)

        # Right: controls
        panel = QFrame()
        panel.setFixedWidth(360)
        ctrl = QVBoxLayout(panel)
        main_layout.addWidget(panel, stretch=0)

        # -- Observer --
        grp_obs = QGroupBox("Observer Position")
        form_obs = QFormLayout()
        self.spin_x = self._sb(-200, 200, 0.0)
        self.spin_y = self._sb(-200, 200, 0.0)
        self.spin_z = self._sb(0.1, 100, 2.0)
        self.spin_x.valueChanged.connect(self.update_camera_from_ui)
        self.spin_y.valueChanged.connect(self.update_camera_from_ui)
        self.spin_z.valueChanged.connect(self.update_camera_from_ui)
        form_obs.addRow("X:", self.spin_x)
        form_obs.addRow("Y:", self.spin_y)
        form_obs.addRow("Z:", self.spin_z)
        btn_apply = QPushButton("Apply View")
        btn_apply.clicked.connect(self.update_camera_from_ui)
        form_obs.addRow(btn_apply)
        btn_bird = QPushButton("Bird's Eye")
        btn_bird.clicked.connect(self.set_bird_view)
        form_obs.addRow(btn_bird)
        grp_obs.setLayout(form_obs)
        ctrl.addWidget(grp_obs)

        # -- Camera info (live) --
        grp_cam = QGroupBox("Camera (live)")
        cam_layout = QVBoxLayout()
        self.lbl_cam = QLabel("...")
        self.lbl_cam.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.lbl_cam.setWordWrap(True)
        cam_layout.addWidget(self.lbl_cam)
        grp_cam.setLayout(cam_layout)
        ctrl.addWidget(grp_cam)

        # -- Time --
        grp_time = QGroupBox("Time Control")
        vt = QVBoxLayout()
        self.lbl_time = QLabel("t = 0.00 s")
        self.lbl_time.setStyleSheet("font-size: 14px; font-weight: bold;")
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
        self.btn_play.setStyleSheet("font-weight:bold; padding:6px;")
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

        # -- Seeds --
        grp_seed = QGroupBox("Scene Generation")
        form_seed = QFormLayout()
        self.spin_seed_setup = self._sb(0, 999999, self.sim.seed_setup, 1, 0)
        self.spin_seed_g = self._sb(0, 999999, self.sim.seed_g, 1, 0)
        btn_regen = QPushButton("Regenerate")
        btn_regen.clicked.connect(self.regenerate_scene)
        form_seed.addRow("seed_setup:", self.spin_seed_setup)
        form_seed.addRow("seed_g:", self.spin_seed_g)
        form_seed.addRow(btn_regen)
        grp_seed.setLayout(form_seed)
        ctrl.addWidget(grp_seed)

        # -- Detailed scene info --
        grp_info = QGroupBox("Scene Info")
        il = QVBoxLayout()
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.txt_info.setMaximumHeight(250)
        il.addWidget(self.txt_info)
        grp_info.setLayout(il)
        ctrl.addWidget(grp_info)

        ctrl.addStretch()
        btn_ss = QPushButton("Screenshot")
        btn_ss.clicked.connect(self.on_screenshot)
        ctrl.addWidget(btn_ss)

    def _sb(self, lo, hi, default, step=0.5, decimals=2):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setValue(default)
        sb.setSingleStep(step); sb.setDecimals(int(decimals))
        return sb

    # -----------------------------------------------------------------
    def delayed_init(self):
        self.render_static_scene()
        self.update_dynamic_actors()
        self.plotter.add_light(pv.Light(position=(0,0,50), show_actor=False, intensity=0.8))
        self.plotter.add_light(pv.Light(light_type='headlight', intensity=0.5))
        self.set_bird_view()
        self.update_scene_info()
        self.plotter.render()

    def render_static_scene(self):
        self.plotter.clear()
        scene = self.sim.scene
        gc = scene['ground_color']

        ground = pv.Plane(center=(0,0,0), direction=(0,0,1),
                          i_size=200, j_size=200,
                          i_resolution=200, j_resolution=200)
        self.plotter.add_mesh(ground, color=gc, show_edges=True,
                              edge_color='#d0d0d0')

        for rm in scene['ramps']:
            pts = rm.ramp_vertices()
            faces = np.hstack([
                [4,0,1,5,4],[3,0,3,4],[3,1,2,5],
                [4,0,1,2,3],[4,3,2,5,4],
            ])
            self.plotter.add_mesh(pv.PolyData(pts, faces), color=rm.color,
                                  show_edges=True, edge_color=[0.65]*3)

    def update_dynamic_actors(self):
        positions = self.sim.get_ball_positions(self.current_time)
        balls = self.sim.scene['balls']

        for i, (pos, ball) in enumerate(zip(positions, balls)):
            for pfx in (f"ball_{i}", f"ind_{i}", f"shadow_{i}"):
                try: self.plotter.remove_actor(pfx)
                except: pass

            if pos is None:
                continue

            sphere = pv.Sphere(radius=ball.radius, center=pos,
                               theta_resolution=32, phi_resolution=32)
            self.plotter.add_mesh(sphere, color=ball.color,
                                  smooth_shading=True, specular=0.4,
                                  name=f"ball_{i}")

            # Fixed red scale bars at (3,0,0.1) / (-3,0,0.1)
            d = ball.radius
            if ball.side == +1:
                p1 = np.array([3.0, -d, 0.1])
                p2 = np.array([3.0,  d, 0.1])
            else:
                p1 = np.array([-3.0, -d, 0.1])
                p2 = np.array([-3.0,  d, 0.1])
            self.plotter.add_mesh(pv.Line(p1, p2), color='red',
                                  line_width=3, name=f"ind_{i}")

            # Ground shadow disc
            shadow = pv.Disc(center=(pos[0], pos[1], 0.05),
                             normal=(0, 0, 1),
                             inner=0, outer=ball.radius,
                             r_res=1, c_res=36)
            self.plotter.add_mesh(shadow, color=[0.25, 0.25, 0.25],
                                  opacity=0.55, lighting=False,
                                  name=f"shadow_{i}")

    # -----------------------------------------------------------------
    def update_scene_info(self):
        """Detailed info with ball vel0, pos0, ramp geometry."""
        s = self.sim
        lines = [f"Gravity: {s.gravity_str()}",
                 f"Seeds: setup={s.seed_setup}  g={s.seed_g}", ""]

        for i, rm in enumerate(s.scene['ramps']):
            ux, uy = rm.uphill_dir_xy()
            lines.append(f"--- Ramp {i} ---")
            lines.append(f"  slope: {rm.angle_deg:.1f} deg")
            lines.append(f"  orient vs Y: {rm.orientation_deg:.1f} deg")
            lines.append(f"  height: {rm.height:.2f}m  width: {rm.width:.2f}m")
            lines.append(f"  run: {rm.run_length:.2f}m")
            lines.append(f"  base: ({rm.base_x:.2f}, {rm.base_y:.2f})")
            lines.append(f"  uphill dir: ({ux:.3f}, {uy:.3f})")
            lines.append("")

        if not s.scene['ramps']:
            lines.append("Ramp: None\n")

        for i, b in enumerate(s.scene['balls']):
            tag = 'X>0' if b.side == 1 else 'X<0'
            lines.append(f"--- Ball {i} [{tag}] ---")
            lines.append(f"  type: {b.motion_type}")
            lines.append(f"  radius: {b.radius:.3f}m")
            lines.append(f"  pos0: ({b.pos0[0]:.3f}, {b.pos0[1]:.3f}, {b.pos0[2]:.3f})")
            lines.append(f"  vel0: ({b.vel0[0]:.3f}, {b.vel0[1]:.3f}, {b.vel0[2]:.3f})")
            lines.append(f"  |vel0|: {np.linalg.norm(b.vel0):.3f} m/s")
            lines.append("")

        self.txt_info.setPlainText("\n".join(lines))

    def update_camera_info(self):
        """Poll camera and display parameters."""
        try:
            cam = self.plotter.camera
            pos = cam.position
            fp = cam.focal_point
            up = cam.up
            va = cam.view_angle

            # compute azimuth/elevation from look direction
            look = np.array(fp) - np.array(pos)
            look_norm = np.linalg.norm(look)
            if look_norm > 1e-9:
                look /= look_norm
            az = np.degrees(np.arctan2(look[1], look[0]))
            el = np.degrees(np.arcsin(np.clip(look[2], -1, 1)))

            self.lbl_cam.setText(
                f"pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                f"focal: ({fp[0]:.1f}, {fp[1]:.1f}, {fp[2]:.1f})\n"
                f"up: ({up[0]:.2f}, {up[1]:.2f}, {up[2]:.2f})\n"
                f"az: {az:.1f} deg  el: {el:.1f} deg\n"
                f"view_angle: {va:.1f} deg"
            )
        except Exception:
            pass

    # -----------------------------------------------------------------
    def update_camera_from_ui(self):
        x, y, z = self.spin_x.value(), self.spin_y.value(), self.spin_z.value()
        self.plotter.camera.position = (x, y, z)
        self.plotter.camera.focal_point = (x+10, y, z)
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.render()

    def set_bird_view(self):
        self.plotter.camera.position = (0, -60, 40)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.render()

    # -----------------------------------------------------------------
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
        self.update_dynamic_actors()

    def on_slider_change(self, value):
        self.current_time = value / 100.0
        self.lbl_time.setText(f"t = {self.current_time:.2f} s")
        if not self.is_playing:
            self.update_dynamic_actors()

    def on_reset(self):
        if self.is_playing: self.toggle_play()
        self.current_time = 0.0
        self.slider_time.setValue(0)
        self.lbl_time.setText("t = 0.00 s")
        self.update_dynamic_actors()

    def on_step(self):
        if self.is_playing: return
        self.current_time += 0.1
        self.slider_time.setValue(int(self.current_time * 100))
        self.lbl_time.setText(f"t = {self.current_time:.2f} s")
        self.update_dynamic_actors()

    def regenerate_scene(self):
        if self.is_playing: self.toggle_play()
        self.current_time = 0.0
        self.slider_time.setValue(0)
        self.lbl_time.setText("t = 0.00 s")
        self.sim.regenerate(int(self.spin_seed_setup.value()),
                            int(self.spin_seed_g.value()))
        self.render_static_scene()
        self.update_dynamic_actors()
        self.update_scene_info()
        self.plotter.add_light(pv.Light(position=(0,0,50), show_actor=False, intensity=0.8))
        self.plotter.add_light(pv.Light(light_type='headlight', intensity=0.5))
        self.set_bird_view()

    def on_screenshot(self):
        p = f"screenshot_t{self.current_time:.2f}.png"
        self.plotter.screenshot(p)
        self.statusBar().showMessage(f"Saved: {p}", 3000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_setup", type=int, default=42)
    ap.add_argument("--seed_g",     type=int, default=7)
    args = ap.parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args.seed_setup, args.seed_g)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
