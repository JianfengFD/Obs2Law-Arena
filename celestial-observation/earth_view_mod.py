import numpy as np
import collections
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from PIL import Image
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# ==========================================
# 核心渲染器类：虚拟望远镜
# ==========================================
class VirtualTelescope:
    def __init__(self, simulation_data_path, star_catalog_path='tycho2_entire_sky.fits'):
        """
        初始化虚拟望远镜环境
        :param simulation_data_path: 轨道演化数据文件路径 (例如 'simulation_data_3.0yrs.txt')
        :param star_catalog_path: 第谷星表 FITS 文件路径
        """
        self.AU2M = 1.495978707e11
        self.KM2M = 1000
        self.Pix = 1024
        self.Th_view0 = 10 / 180 * np.pi
        
        # 预加载背景星表
        print(f"正在加载背景星表: {star_catalog_path} ...")
        self.star_table = Table.read(star_catalog_path, format='fits')
        
        # 定义天体物理属性 (转换为米)
        self.Radii = {
            'sun': 695700.0 * self.KM2M,
            'earth': 6378.0 * self.KM2M,
            'moon': 1737.4 * self.KM2M,
            'mercury': 2439.7 * self.KM2M,
            'venus': 6051.8 * self.KM2M,
            'mars': 3389.5 * self.KM2M
        }
        self.colors = {
            'sun': 'yellow', 'earth': 'blue', 'moon': 'gray',
            'mercury': 'orange', 'venus': 'gold', 'mars': 'red'
        }
        
        # 加载并构建插值器
        print(f"正在加载轨道数据并构建插值器: {simulation_data_path} ...")
        self._build_interpolators(simulation_data_path)
        print("虚拟望远镜初始化完成。")

    def _build_interpolators(self, filepath):
        """解析 TXT 文件并为每个坐标轴构建一维插值函数"""
        with open(filepath, 'r') as f:
            header = f.readline().strip().split()
        
        # 提取被追踪的天体名称
        self.tracked_bodies = []
        for col in header:
            if col.endswith("_X"):
                self.tracked_bodies.append(col.replace("_X", ""))
                
        # 加载数据矩阵
        data = np.loadtxt(filepath, skiprows=1)
        self.t_days = data[:, 0]  # 第一列是时间 (天)
        
        self.interpolators = {}
        for body in self.tracked_bodies:
            idx_x = header.index(f"{body}_X")
            idx_y = header.index(f"{body}_Y")
            idx_z = header.index(f"{body}_Z")
            
            # 使用 cubic 插值以保证速度/加速度的平滑连续性
            interp_x = interp1d(self.t_days, data[:, idx_x], kind='cubic', bounds_error=False, fill_value="extrapolate")
            interp_y = interp1d(self.t_days, data[:, idx_y], kind='cubic', bounds_error=False, fill_value="extrapolate")
            interp_z = interp1d(self.t_days, data[:, idx_z], kind='cubic', bounds_error=False, fill_value="extrapolate")
            
            self.interpolators[body] = (interp_x, interp_y, interp_z)

    def get_positions_at_time(self, t_day):
        """获取精确到任意时刻的天体位置 (单位转为米)"""
        positions = {}
        for body, (ix, iy, iz) in self.interpolators.items():
            # 获取 AU 并转换为 Meters
            x = float(ix(t_day)) * self.AU2M
            y = float(iy(t_day)) * self.AU2M
            z = float(iz(t_day)) * self.AU2M
            positions[body] = np.array([x, y, z])
        return positions

    def capture(self, time_days, lon_deg, lat_deg, phi_deg, theta_deg, zoom=1.0, tilt_deg=23.439281):
        """
        主控函数：拍摄天空图像
        输入参数可以是标量，也可以是列表。如果是列表，将生成多张图像。
        """
        # 将输入统一转为 numpy 数组以便广播
        params = [np.atleast_1d(x) for x in (time_days, lon_deg, lat_deg, phi_deg, theta_deg, zoom)]
        max_len = max(len(p) for p in params)
        
        # 补齐数组长度
        params = [np.pad(p, (0, max_len - len(p)), mode='edge') if len(p) < max_len else p for p in params]
        t_arr, lon_arr, lat_arr, phi_arr, theta_arr, zoom_arr = params
        
        deg2rad = np.pi / 180.0
        images = []
        
        for i in range(max_len):
            t = t_arr[i]
            lon = lon_arr[i] * deg2rad
            lat = lat_arr[i] * deg2rad
            phi = phi_arr[i] * deg2rad
            theta = theta_arr[i] * deg2rad
            z = zoom_arr[i]
            tilt = tilt_deg * deg2rad
            
            # 1. 插值获取当前时刻行星位置
            positions = self.get_positions_at_time(t)
            
            # 2. 渲染太阳系天体与掩膜
            img_solar, u_view, u_W, u_H = self._render_solar_system(lon, lat, phi, theta, z, tilt, positions)
            
            # 3. 渲染背景恒星
            img_stars = self._render_stars(u_view, u_W, u_H, z)
            
            # 4. 图像合成
            img_composite = self._composite(img_solar, img_stars)
            
            # 5. 翻转以符合常理视觉
            img_pil = Image.fromarray(img_composite).transpose(Image.FLIP_TOP_BOTTOM)
            images.append(img_pil)
            
        return images if max_len > 1 else images[0]

    # --- 以下为内部渲染与坐标计算方法 ---
    def _render_solar_system(self, lon, lat, phi, theta, zoom, th_axis, positions):
        Th_view = self.Th_view0 / zoom
        MaxH = np.tan(Th_view)
        MaxW = np.tan(Th_view)
        H_hor = (MaxH - np.tan(theta)) * self.Pix / (2 * MaxH)

        u_view0 = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)])
        x_view0, y_view0, z_view0 = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])

        # 旋转逻辑 (受 tilt 影响)
        u1 = np.array([-1, 0, 0])
        u_axis0 = np.array([0, 0, 1])
        rotation1 = R.from_rotvec(th_axis * u1)
        u_axis = rotation1.apply(u_axis0)
        u2 = np.cross(u1, u_axis)

        rotation2 = R.from_rotvec((np.pi / 2 - lat) * u2)
        rotation3 = R.from_rotvec(lon * u_axis)

        rot_total = lambda v: rotation3.apply(rotation2.apply(rotation1.apply(v)))
        u_view = rot_total(u_view0)
        z_view = rot_total(z_view0)

        u_pos0 = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        u_pos = rotation1.apply(u_pos0)

        posE = positions['earth']
        RE = self.Radii['earth']
        bodies = [b for b in self.tracked_bodies if b != 'earth']

        pos_array = np.array([positions[b] for b in bodies])
        dpos_array = pos_array - posE - u_pos * RE
        dp_array = np.linalg.norm(dpos_array, axis=1)
        up_array = dpos_array / dp_array[:, np.newaxis]

        in_view = np.dot(up_array, u_view) > np.cos(Th_view + 3/180*np.pi)
        up_array = up_array[in_view]
        bodies = [b for b, flag in zip(bodies, in_view) if flag]

        Hu = z_view - np.dot(z_view, u_view) * u_view
        Hu /= np.linalg.norm(Hu)
        Wu = -np.cross(Hu, u_view)
        Wu /= np.linalg.norm(Wu)

        P_vec = (1 / np.dot(up_array, u_view))[:, np.newaxis] * up_array - u_view
        Hp = (np.dot(P_vec, Hu) + MaxH) * self.Pix / (2 * MaxH)
        Wp = (np.dot(P_vec, Wu) + MaxW) * self.Pix / (2 * MaxW)
        Rp = np.array([self.Radii[b] for b in bodies]) / dp_array[in_view] * self.Pix / (2 * MaxH)

        image = np.zeros((self.Pix, self.Pix, 3), dtype=np.float32)
        xx, yy = np.meshgrid(np.arange(self.Pix), np.arange(self.Pix))

        color_map = {
            'yellow': (1.,1.,0.), 'blue': (0.,0.,1.), 'gray': (0.5,0.5,0.5),
            'orange': (1.,0.5,0.), 'gold': (1.,0.84,0.), 'red': (1.,0.,0.)
        }

        for b, w, h, r in zip(bodies, Wp, Hp, Rp):
            mask = np.sqrt((xx - int(w))**2 + (yy - int(h))**2) <= r
            image[mask] = color_map.get(self.colors.get(b), (1.,1.,1.))

        # 地平线遮蔽
        image[yy < H_hor] = [0.1, 0.1, 0.0]
        return image, u_view, Wu, Hu

    def _render_stars(self, u_view, u_W, u_H, zoom):
        ra, dec = self._unit2radec(u_view)
        radius_sky = (self.Th_view0 / zoom) * 180 / np.pi
        
        # 筛选天区
        coords = SkyCoord(ra=self.star_table['RA_ICRS_'], dec=self.star_table['DE_ICRS_'], unit='deg', frame='icrs')
        target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        local_table = self.star_table[coords.separation(target) <= radius_sky * u.deg]
        
        u_stars = self._radec2unit(local_table['RA_ICRS_'], local_table['DE_ICRS_'])
        v_stars = u_stars - u_view
        xdeg = np.sum(v_stars * u_W, axis=1) * 180 / np.pi
        ydeg = np.sum(v_stars * u_H, axis=1) * 180 / np.pi
        
        vtmag = local_table['VTmag']
        vtmag_filled = np.where(np.isnan(vtmag), 99, vtmag)
        star_size = 10 / (vtmag_filled + 2)
        
        # 绘制星图
        img_star = np.full((self.Pix, self.Pix, 3), [0, 0, 40], dtype=np.uint8)
        if len(xdeg) == 0: return img_star
        
        x_pix = ((xdeg - np.min(xdeg)) / (np.max(xdeg) - np.min(xdeg) + 1e-9) * (self.Pix - 1)).astype(int)
        y_pix = ((ydeg - np.min(ydeg)) / (np.max(ydeg) - np.min(ydeg) + 1e-9) * (self.Pix - 1)).astype(int)
        
        # === 核心修改点：调整亮度和星星半径 ===
        # 亮度：原基础上乘以 0.75，限制最大值为 255
        colors = np.clip((vtmag_filled * 20).astype(int) * 2 * 0.5, 0, 255).astype(int)
        
        for x, y, c, s in zip(x_pix, y_pix, colors, star_size):
            # 半径：变成原来的 1/3
            r = int((s * 2) / 2.0)
            
            # 即使缩减后 r == 0，也通过 y_s:y_e (长度为1的区间) 画一个 1x1 像素的点，避免暗星消失
            if r > 0:
                y_s, y_e = max(y-r, 0), min(y+r+1, self.Pix)
                x_s, x_e = max(x-r, 0), min(x+r+1, self.Pix)
                img_star[y_s:y_e, x_s:x_e] = np.maximum(img_star[y_s:y_e, x_s:x_e], c)
                
        return img_star

    def _composite(self, img_solar, img_star):
        img_solar_8bit = (np.clip(img_solar, 0, 1) * 255).astype(np.uint8)
        black_mask = np.all(img_solar_8bit == 0, axis=-1, keepdims=True)
        comp = np.where(black_mask, img_star, img_solar_8bit)
        
        xx, yy = np.meshgrid(np.arange(self.Pix), np.arange(self.Pix))
        circle_mask = np.sqrt((xx - self.Pix//2)**2 + (yy - self.Pix//2)**2) <= self.Pix//2
        comp[~circle_mask] = [0, 0, 0]
        return comp

    @staticmethod
    def _radec2unit(ra_deg, dec_deg):
        ra, dec = np.radians(np.atleast_1d(ra_deg)), np.radians(np.atleast_1d(dec_deg))
        xyz = np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
        return xyz[0] if xyz.shape[0] == 1 and np.isscalar(ra_deg) else xyz

    @staticmethod
    def _unit2radec(xyz):
        xyz = np.atleast_2d(xyz)
        ra = np.degrees(np.arctan2(xyz[:,1], xyz[:,0]))
        dec = np.degrees(np.arcsin(xyz[:,2]))
        return (ra[0], dec[0]) if xyz.shape[0] == 1 else (ra, dec)


# ==========================================
# 主程序：直接运行时的默认演示代码
# ==========================================
if __name__ == "__main__":
    telescope = VirtualTelescope(
        simulation_data_path='simulation_data_20.0yrs.txt', # 替换为你实际生成的文件名
        star_catalog_path='tycho2_entire_sky.fits'
    )
    
    print("\n--- 测试用例 1: 渲染单个特定时刻 ---")
    time_target = 10.123 
    img1 = telescope.capture(
        time_days=time_target,
        lon_deg=86.0, lat_deg=0.0,   
        phi_deg=-90.0, theta_deg=5.0, 
        zoom=1.0,
        tilt_deg=23.44  
    )
    img1.save('sky_view_single.png')
    print("单张测试图已保存至 sky_view_single.png")
    
    print("\n--- 测试用例 2: 同一时刻，多角度环视 (生成多图) ---")
    phis = [-90, -45, 0, 45, 90]
    imgs = telescope.capture(
        time_days=time_target,  
        lon_deg=86.0, lat_deg=0.0,
        phi_deg=phis,           
        theta_deg=5.0, zoom=1.0
    )
    for i, img in enumerate(imgs):
        img.save(f'sky_view_multi_phi_{phis[i]}.png')
    print(f"多角度测试组已保存 ({len(imgs)} 张图片)。")