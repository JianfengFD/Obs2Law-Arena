"""
physics_sim.py  —  3D Physics Simulation with PyVista  (v2)
=============================================================
Changes vs v1:
  (1) Ball radius: Gaussian mu=0.30m sig=0.20m, clipped [0.10, 1.50]
  (2) Ramp orientation: angle to Y axis sampled from bimodal ~20/160 deg
      Ramp position: within 20m of origin, |x| >= 3m, only X>0
  (3) Ramp ball: centre offset by radius along surface normal (not embedded)
  (4) Red scale bars: fixed positions (0,3,0.1) for X>0 ball,
      (0,-3,0.1) for X<0 ball, length = ball diameter, parallel to Y
  (5) render_scene: observer_pos can be single or list;
      output images have NO text overlays
  (6) Camera: strict pin-hole model, observer position exact,
      view_up = horizon-level (eyes horizontal), no zoom
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence, Union
import warnings

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    warnings.warn("pyvista not found – pip install pyvista")


# =========================================================================
# 1. Data classes
# =========================================================================

@dataclass
class GravityConfig:
    g0: float
    alpha: float
    def g(self, z: float) -> float:
        return self.g0 + self.alpha / (z + 10.0)

@dataclass
class Ramp:
    angle_deg: float       # slope angle 10-50 deg
    height: float          # 1.2-2.0 m
    width: float           # 2.0-3.0 m
    base_x: float          # x of LOW end centre
    base_y: float          # y of LOW end centre
    orientation_deg: float # angle of ramp "uphill" direction vs +Y axis
    color: np.ndarray = field(default_factory=lambda: np.array([0.85, 0.85, 0.80]))

    @property
    def angle_rad(self):
        return np.radians(self.angle_deg)

    @property
    def run_length(self):
        return self.height / np.tan(self.angle_rad)

    @property
    def orient_rad(self):
        return np.radians(self.orientation_deg)

    def uphill_dir_xy(self):
        """Unit vector in XY plane pointing from low end toward high end."""
        th = self.orient_rad
        return np.array([np.sin(th), np.cos(th)])  # angle from +Y

    def surface_normal(self):
        """Outward normal of the slope surface (pointing away from solid)."""
        ux, uy = self.uphill_dir_xy()
        ca, sa = np.cos(self.angle_rad), np.sin(self.angle_rad)
        # normal = -sin(slope)*uphill_xy + cos(slope)*z
        return np.array([-sa * ux, -sa * uy, ca])

    def ramp_vertices(self):
        """6 vertices of the wedge: 0-3 ground, 4-5 top edge."""
        ux, uy = self.uphill_dir_xy()
        hw = self.width / 2.0
        # perpendicular to uphill in XY
        px, py = -uy, ux  # 90 deg left of uphill

        bx, by = self.base_x, self.base_y
        run = self.run_length
        # high end centre
        hx = bx + ux * run
        hy = by + uy * run

        pts = np.array([
            [bx - px*hw, by - py*hw, 0],   # 0 base left
            [bx + px*hw, by + py*hw, 0],   # 1 base right
            [hx + px*hw, hy + py*hw, 0],   # 2 far right ground
            [hx - px*hw, hy - py*hw, 0],   # 3 far left ground
            [hx - px*hw, hy - py*hw, self.height],  # 4 far left top
            [hx + px*hw, hy + py*hw, self.height],  # 5 far right top
        ])
        return pts

    def z_at_xy(self, x, y):
        """Height of ramp surface at world (x,y), or None if outside."""
        ux, uy = self.uphill_dir_xy()
        px, py = -uy, ux
        # vector from base centre to query point
        dx, dy = x - self.base_x, y - self.base_y
        # project onto uphill and perp
        along = dx * ux + dy * uy
        perp = dx * px + dy * py
        if along < 0 or along > self.run_length:
            return None
        if abs(perp) > self.width / 2.0:
            return None
        frac = along / self.run_length
        return frac * self.height

@dataclass
class Ball:
    radius: float
    color: np.ndarray
    motion_type: str
    pos0: np.ndarray
    vel0: np.ndarray
    side: int
    ramp: Optional[Ramp] = None
    on_ramp_initially: bool = False


# =========================================================================
# 2. Scene generation
# =========================================================================

def _random_ball_color(rng):
    return rng.uniform(0.30, 0.78, size=3)

def _random_pale_color(rng):
    base = rng.uniform(0.76, 0.91)
    return np.clip(np.full(3, base) + rng.uniform(-0.04, 0.04, size=3), 0, 1)

def _sample_ball_radius(rng):
    """Gaussian mu=0.40 sig=0.20, clipped to [0.10, 1.50]."""
    r = rng.normal(0.40, 0.20)
    return float(np.clip(r, 0.10, 1.50))

def _sample_ramp_orientation(rng):
    """Bimodal distribution peaked at ~20 and ~160 deg (90 least likely).
    p(th) ~ exp(-(th-20)^2/200) + exp(-(th-160)^2/200), th in [0,180]."""
    sig2 = 2 * 10**2  # 2*sig^2 = 200
    # rejection sampling
    for _ in range(1000):
        th = rng.uniform(0, 180)
        p = np.exp(-(th - 20)**2 / sig2) + np.exp(-(th - 160)**2 / sig2)
        if rng.random() < p:
            return th
    return rng.uniform(0, 180)  # fallback

def _sample_ball_xy(rng, side: int):
    """Sample (px, py) from Gaussian centred at (side*10, 0), sig=6.
    Ensures px has the correct sign for the given side."""
    sig = 6.0
    cx = side * 10.0  # +10 for X>0, -10 for X<0
    for _ in range(500):
        px = rng.normal(cx, sig)
        py = rng.normal(0.0, sig)
        # must stay on correct side and not too close to x=0
        if side == +1 and px >= 2.0:
            return px, py
        if side == -1 and px <= -2.0:
            return px, py
    # fallback
    return side * 10.0, 0.0

def _generate_gravity(seed_g: int) -> GravityConfig:
    rng = np.random.default_rng(seed_g)
    return GravityConfig(g0=rng.uniform(9.0, 10.0),
                         alpha=rng.uniform(-10.0, 10.0))

def _generate_scene(seed_setup: int) -> dict:
    rng = np.random.default_rng(seed_setup)
    ground_color = _random_pale_color(rng)

    ramps: List[Ramp] = []
    ramp_obj: Optional[Ramp] = None
    has_ramp = rng.random() < 0.70

    if has_ramp:
        slope_angle = rng.uniform(10, 50)
        h = rng.uniform(1.2, 2.0)
        w = rng.uniform(2.0, 3.0)
        orient = _sample_ramp_orientation(rng)

        # position: within 20m of origin, |x| >= 3, x > 0
        for _ in range(200):
            bx = rng.uniform(3, 20)
            by = rng.uniform(-20, 20)
            if bx**2 + by**2 <= 20**2:
                break

        ramp_obj = Ramp(angle_deg=slope_angle, height=h, width=w,
                        base_x=bx, base_y=by,
                        orientation_deg=orient,
                        color=ground_color.copy())
        ramps.append(ramp_obj)

    balls: List[Ball] = []

    def _make_ball(side: int):
        if side == 1 and has_ramp and rng.random() < 0.55:
            mtype = 'ramp'
        else:
            mtype = rng.choice(['projectile', 'freefall'])

        r = _sample_ball_radius(rng)
        color = _random_ball_color(rng)

        if mtype == 'projectile':
            h0 = rng.uniform(0.3, 1.5)
            speed = rng.uniform(5, 10)
            elev = rng.uniform(10, 70)
            az = rng.uniform(-60, 60)
            vx = speed * np.cos(np.radians(elev)) * np.cos(np.radians(az))
            vy = speed * np.cos(np.radians(elev)) * np.sin(np.radians(az))
            vz = speed * np.sin(np.radians(elev))
            px, py = _sample_ball_xy(rng, side)
            pos0 = np.array([px, py, h0 + r])
            vel0 = np.array([side * abs(vx), vy, vz])
            balls.append(Ball(r, color, mtype, pos0, vel0, side))

        elif mtype == 'freefall':
            h0 = rng.normal(10.0, 3.0)
            h0 = max(h0, 1.0)  # floor at 1m
            px, py = _sample_ball_xy(rng, side)
            pos0 = np.array([px, py, h0 + r])
            vel0 = np.zeros(3)
            balls.append(Ball(r, color, mtype, pos0, vel0, side))

        else:  # ramp
            ramp = ramp_obj
            frac = rng.uniform(0.50, 0.92)
            ux, uy = ramp.uphill_dir_xy()
            run = ramp.run_length
            along = frac * run
            # position on surface
            sx = ramp.base_x + ux * along
            sy = ramp.base_y + uy * along
            sz = frac * ramp.height
            # offset ball centre by radius along surface normal
            n = ramp.surface_normal()
            px = sx + n[0] * r
            py = sy + n[1] * r
            pz = sz + n[2] * r

            init_speed = 0.0  # ramp ball starts from rest
            vel0 = np.zeros(3)
            pos0 = np.array([px, py, pz])
            balls.append(Ball(r, color, mtype, pos0, vel0, side,
                              ramp=ramp, on_ramp_initially=True))

    _make_ball(+1)
    _make_ball(-1)
    return dict(balls=balls, ramps=ramps, ground_color=ground_color)


# =========================================================================
# 3. Physics engine
# =========================================================================

def _simulate_ball(ball: Ball, grav: GravityConfig, t: float,
                   ramps: List[Ramp]) -> Optional[np.ndarray]:
    DT = 0.0005
    steps = int(t / DT)
    remainder = t - steps * DT

    pos = ball.pos0.astype(np.float64).copy()
    vel = ball.vel0.astype(np.float64).copy()
    r = ball.radius
    on_ramp = ball.on_ramp_initially
    ramp = ball.ramp

    for i in range(steps + 1):
        h = DT if i < steps else remainder
        if h <= 0:
            break

        g_val = grav.g(max(pos[2], 0.0))

        if on_ramp and ramp is not None:
            theta = ramp.angle_rad
            ux, uy = ramp.uphill_dir_xy()
            n = ramp.surface_normal()
            # downhill unit along slope
            dh_x = -ux * np.cos(theta)
            dh_y = -uy * np.cos(theta)
            dh_z = -np.sin(theta)
            a_down = (5.0 / 7.0) * g_val * np.sin(theta)

            vel[0] += a_down * dh_x * h
            vel[1] += a_down * dh_y * h
            vel[2] += a_down * dh_z * h
            pos += vel * h

            # check still on ramp using CONTACT POINT (not ball centre)
            contact_x = pos[0] - n[0] * r
            contact_y = pos[1] - n[1] * r
            rz = ramp.z_at_xy(contact_x, contact_y)
            if rz is None:
                on_ramp = False
                pos[2] = max(pos[2], r)
            else:
                # constrain ball centre to surface + r * normal
                pos[2] = rz + n[2] * r
                # also correct xy to stay on the normal offset line
                pos[0] = contact_x + n[0] * r
                pos[1] = contact_y + n[1] * r

        else:
            vel[2] -= g_val * h
            pos += vel * h

            # ground bounce
            if pos[2] < r:
                pos[2] = 2 * r - pos[2]
                vel[2] = abs(vel[2])

            # ramp collision (use contact point)
            for rm in ramps:
                n_rm = rm.surface_normal()
                cx = pos[0] - n_rm[0] * r
                cy = pos[1] - n_rm[1] * r
                rz = rm.z_at_xy(cx, cy)
                if rz is not None and (pos[2] - n_rm[2] * r) < rz:
                    vn = np.dot(vel, n_rm)
                    if vn < 0:
                        vel -= 2.0 * vn * n_rm
                    pos[2] = rz + r * n_rm[2]
                    pos[0] = cx + n_rm[0] * r
                    pos[1] = cy + n_rm[1] * r

        # boundary checks
        if ball.side == +1 and pos[0] < 0:
            return None
        if ball.side == -1 and pos[0] > 0:
            return None
        if abs(pos[0]) > 100 or abs(pos[1]) > 100:
            return None

    return pos


# =========================================================================
# 4. Rendering
# =========================================================================

def _add_scene_meshes(pl, scene, grav, t, observer_pos):
    """Add ground, ramps, balls, indicators. Returns ball positions."""
    gc = scene['ground_color']
    balls = scene['balls']
    ramps = scene['ramps']

    # ground extent
    xs = [observer_pos[0]] + [b.pos0[0] for b in balls]
    ys = [observer_pos[1]] + [b.pos0[1] for b in balls]
    for rm in ramps:
        verts = rm.ramp_vertices()
        xs.extend(verts[:, 0].tolist())
        ys.extend(verts[:, 1].tolist())

    pad = 25
    cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2
    sx = max(max(xs)-min(xs)+2*pad, 60)
    sy = max(max(ys)-min(ys)+2*pad, 60)

    ground = pv.Plane(center=(cx,cy,0), direction=(0,0,1),
                      i_size=sx, j_size=sy,
                      i_resolution=int(sx), j_resolution=int(sy))
    pl.add_mesh(ground, color=gc, show_edges=True,
                edge_color=[gc[0]*0.88, gc[1]*0.88, gc[2]*0.88])

    # Ramps
    for rm in ramps:
        pts = rm.ramp_vertices()
        faces = np.hstack([
            [4,0,1,5,4], [3,0,3,4], [3,1,2,5],
            [4,0,1,2,3], [4,3,2,5,4],
        ])
        pl.add_mesh(pv.PolyData(pts, faces), color=rm.color,
                    show_edges=True, edge_color=[0.65]*3)
        # height ticks on back face
        for tz in np.arange(1.0, rm.height+0.01, 1.0):
            if tz > rm.height: break
            pl.add_mesh(pv.Line(pts[4] + (pts[5]-pts[4]) * 0,
                                pts[5]),
                        color='gray', line_width=1.0)

    # Balls
    ball_positions = []
    for ball in balls:
        pos = _simulate_ball(ball, grav, t, ramps)
        ball_positions.append(pos)
        if pos is None:
            continue

        sphere = pv.Sphere(radius=ball.radius, center=pos,
                           theta_resolution=32, phi_resolution=32)
        pl.add_mesh(sphere, color=ball.color, smooth_shading=True,
                    specular=0.4, specular_power=15)

    # Fixed red scale bars (NO text labels in output)
    # X>0 ball indicator centred at (3, 0, 0.1), X<0 at (-3, 0, 0.1)
    # parallel to Y axis, length = ball diameter
    for ball, pos in zip(balls, ball_positions):
        if pos is None:
            continue
        d = ball.radius  # half-length = radius => full length = diameter
        if ball.side == +1:
            p1 = np.array([3.0, -d, 0.1])
            p2 = np.array([3.0,  d, 0.1])
        else:
            p1 = np.array([-3.0, -d, 0.1])
            p2 = np.array([-3.0,  d, 0.1])
        pl.add_mesh(pv.Line(p1, p2), color='red', line_width=3)

    # Ground shadow (vertical projection) — opaque dark disc slightly above ground
    # Using z=0.05 and opaque to avoid z-fighting with ground plane
    for ball, pos in zip(balls, ball_positions):
        if pos is None:
            continue
        shadow = pv.Disc(center=(pos[0], pos[1], 0.05),
                         normal=(0, 0, 1),
                         inner=0, outer=ball.radius,
                         r_res=1, c_res=36)
        # Blend with ground: dark grey, slight transparency
        pl.add_mesh(shadow, color=[0.25, 0.25, 0.25], opacity=0.55,
                    lighting=False)  # lighting=False prevents it from being washed out

    # Lighting
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(60,60,100), intensity=0.8))
    pl.add_light(pv.Light(position=(-40,-40,50), intensity=0.35))
    pl.add_light(pv.Light(position=(0,0,10), intensity=0.25))

    return ball_positions


def _setup_camera(pl, observer_pos, azimuth, elevation):
    """
    Strict pin-hole camera at observer_pos.
    azimuth: 0=+X, 90=+Y (degrees)
    elevation: 0=horizontal, >0=up (degrees)
    view_up = always horizontal (eyes level), computed perpendicular.
    No zoom/dolly — view_angle is fixed at 60 deg (standard human FOV).
    """
    obs = np.asarray(observer_pos, dtype=float)
    az = np.radians(azimuth)
    el = np.radians(elevation)

    # look direction
    look = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])

    # "up" must be perpendicular to look and horizontal
    # right = look x world_z, then up = right x look
    world_z = np.array([0.0, 0.0, 1.0])
    right = np.cross(look, world_z)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-9:
        # looking straight up or down, pick arbitrary horizontal up
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= r_norm
    up = np.cross(right, look)
    up /= np.linalg.norm(up)

    focal = obs + look * 50.0

    pl.camera.position = tuple(obs)
    pl.camera.focal_point = tuple(focal)
    pl.camera.up = tuple(up)
    pl.camera.view_angle = 60.0  # fixed, no zoom


def _render_frame(scene, grav, t, observer_pos, azimuth, elevation,
                  image_size=(512, 512)):
    if not HAS_PYVISTA:
        raise RuntimeError("pyvista required")

    pl = pv.Plotter(off_screen=True, window_size=list(image_size))
    pl.set_background([0.92, 0.94, 0.97])

    obs = np.asarray(observer_pos, dtype=float)
    _add_scene_meshes(pl, scene, grav, t, obs)
    _setup_camera(pl, obs, azimuth, elevation)

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# =========================================================================
# 5. Public API
# =========================================================================

def render_scene(
    seed_setup: int,
    seed_g: int,
    times: Union[float, Sequence[float]],
    angles: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    observer_pos: Union[
        Tuple[float, float, float],
        Sequence[Tuple[float, float, float]]
    ] = (0.0, 0.0, 2.0),
    image_size: Tuple[int, int] = (512, 512),
) -> List[np.ndarray]:
    """
    Render the physics scene.

    Parameters
    ----------
    seed_setup, seed_g : int
    times : float or list[float]
    angles : (az, el) or list[(az, el)]
    observer_pos : (x,y,z) or list[(x,y,z)]
        Single position or list. If list, must pair with times/angles.
    image_size : (w, h)

    Pairing rules (applied across times, angles, observer_pos):
      All singletons are broadcast. Otherwise lengths must match.

    Returns: list[np.ndarray]  H x W x 3 uint8, NO text overlays.
    """
    # normalise times
    if isinstance(times, (int, float)):
        times = [float(times)]
    else:
        times = [float(t) for t in times]

    # normalise angles
    if (isinstance(angles, (tuple, list)) and len(angles) == 2
            and isinstance(angles[0], (int, float))):
        angles = [tuple(angles)]
    else:
        angles = [tuple(a) for a in angles]

    # normalise observer_pos: single (x,y,z) or list of (x,y,z)
    if (isinstance(observer_pos, (tuple, list))
            and len(observer_pos) == 3
            and isinstance(observer_pos[0], (int, float))):
        obs_list = [tuple(observer_pos)]
    else:
        obs_list = [tuple(o) for o in observer_pos]

    # determine N
    lengths = [len(times), len(angles), len(obs_list)]
    non_one = [l for l in lengths if l > 1]
    if non_one:
        N = non_one[0]
        if not all(l == N for l in non_one):
            raise ValueError(f"Cannot pair lengths {lengths}")
    else:
        N = 1

    def broadcast(lst):
        return lst * N if len(lst) == 1 else lst

    times = broadcast(times)
    angles = broadcast(angles)
    obs_list = broadcast(obs_list)

    scene = _generate_scene(seed_setup)
    grav = _generate_gravity(seed_g)

    print(f"[physics_sim] g(z) = {grav.g0:.4f} + {grav.alpha:.4f}/(z+10)")
    for b in scene['balls']:
        tag = 'X>0' if b.side == 1 else 'X<0'
        print(f"  [{tag}] {b.motion_type:>10s} r={b.radius:.2f}m "
              f"pos0={np.round(b.pos0,2)} vel0={np.round(b.vel0,2)}")

    images = []
    for idx in range(N):
        t, (az, el), obs = times[idx], angles[idx], obs_list[idx]
        print(f"  frame {idx+1}/{N}: t={t:.3f}s az={az:.1f} el={el:.1f} "
              f"obs={obs}", end=" ... ", flush=True)
        img = _render_frame(scene, grav, t, obs, az, el, image_size)
        print("done")
        images.append(img)

    return images


# =========================================================================
# 6. Utilities
# =========================================================================

def describe_scene(seed_setup: int, seed_g: int):
    scene = _generate_scene(seed_setup)
    grav = _generate_gravity(seed_g)
    sep = "=" * 65
    print(f"\n{sep}\n  SCENE DESCRIPTION\n{sep}")
    print(f"  gravity: g(z) = {grav.g0:.4f} + {grav.alpha:.4f}/(z+10)")
    print(f"  ground: RGB {np.round(scene['ground_color'],3)}")
    for i, rm in enumerate(scene['ramps']):
        print(f"  ramp {i}: slope={rm.angle_deg:.1f} deg  orient={rm.orientation_deg:.1f} deg")
        print(f"    h={rm.height:.2f}m  w={rm.width:.2f}m  "
              f"base=({rm.base_x:.1f},{rm.base_y:.1f})  run={rm.run_length:.2f}m")
    for i, b in enumerate(scene['balls']):
        tag = 'X>0' if b.side == 1 else 'X<0'
        print(f"  ball {i} [{tag}] {b.motion_type:>10s} r={b.radius:.3f}m")
        print(f"    pos0={np.round(b.pos0,3)}  vel0={np.round(b.vel0,3)}")
    print(sep)

def get_ball_positions(seed_setup, seed_g, t):
    scene = _generate_scene(seed_setup)
    grav = _generate_gravity(seed_g)
    return [_simulate_ball(b, grav, t, scene['ramps']) for b in scene['balls']]
