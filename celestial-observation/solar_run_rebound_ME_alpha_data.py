import sys
import rebound
from skyfield.api import Loader, load
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==========================================
# 1. 解析命令行参数：获取模拟年数
# ==========================================
try:
    years = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
except ValueError:
    print("输入的年数无效，默认计算 1 年。")
    years = 1.0

# ==========================================
# 2. 常量与天体定义
# ==========================================
AU_TO_KM = 149597870.7
DAY_TO_SEC = 86400

# 请确保此处的本地星历路径正确
load = Loader('/Users/lijf/Documents/work/others/AI/AI4SCI/AINewton/COMPARE_STUDY')
ts = load.timescale()
eph = load('de421.bsp')

# 定义模拟时间范围
start_date = datetime(2020, 1, 1)
end_date = start_date + timedelta(days=365.25 * years)
time_step = 0.5  # 采样与存储步长：0.5小时

t_start = ts.utc(start_date.year, start_date.month, start_date.day)

# 天体列表与名称
bodies = [
    'sun', 'mercury barycenter', 'venus barycenter', 'earth',
    'mars barycenter', 'jupiter barycenter', 'saturn barycenter',
    'uranus barycenter', 'neptune barycenter', 'moon'
]
body_names = [
    'sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn',
    'uranus', 'neptune', 'moon'
]

# 设定要落盘保存并追踪其轨迹的重点天体（火星以内及太阳、月亮）
tracked_bodies = ['sun', 'mercury', 'venus', 'earth', 'mars', 'moon']

# 独立地月质量
masses = {
    'sun': 1.0,
    'mercury': 1.6601e-7,
    'venus': 2.4478e-6,
    'earth': 3.0035e-6, 
    'mars': 3.2272e-7,
    'jupiter': 9.5479e-4,
    'saturn': 2.8588e-4,
    'uranus': 4.3662e-5,
    'neptune': 5.1514e-5,
    'moon': 3.6942e-8
}

# 提取起始时刻的位置和速度
positions = {}
velocities = {}
for body, name in zip(bodies, body_names):
    planet = eph[body]
    astrometric = planet.at(t_start)
    positions[name] = astrometric.position.au
    velocities[name] = astrometric.velocity.au_per_d

# ==========================================
# 3. 配置 REBOUND 与修改引力定律
# ==========================================
sim = rebound.Simulation()
sim.units = ('AU', 'days', 'Msun')

# 使用 WHFast 辛积分器
sim.integrator = "whfast"
sim.dt = 2.0 / 24.0  # 积分器的基础步长设定为2小时

for name in body_names:
    sim.add(m=masses[name],
            x=positions[name][0], y=positions[name][1], z=positions[name][2],
            vx=velocities[name][0], vy=velocities[name][1], vz=velocities[name][2])

def modified_gravity_residual(sim_pointer):
    """自定义修正引力 F ∝ (1+alpha) / r^(2+alpha) 的残差注入"""
    alpha =  0.01
    for i in range(sim.N):
        for j in range(i + 1, sim.N):
            p1 = sim.particles[i]
            p2 = sim.particles[j]
            dx, dy, dz = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
            r2 = dx**2 + dy**2 + dz**2
            r = np.sqrt(r2)
            if r > 0:
                term_mod = (1.0 + alpha) / (r2 * r * (r ** alpha))
                term_newton = 1.0 / (r2 * r)
                factor = sim.G * (term_mod - term_newton)
                
                p1.ax -= factor * p2.m * dx
                p1.ay -= factor * p2.m * dy
                p1.az -= factor * p2.m * dz
                
                p2.ax += factor * p1.m * dx
                p2.ay += factor * p1.m * dy
                p2.az += factor * p1.m * dz

sim.additional_forces = modified_gravity_residual
sim.move_to_com()

# ==========================================
# 4. 演化积分与数据落盘
# ==========================================
dt_days = time_step / 24.0
total_days = 365.25 * years
num_points = int(total_days / dt_days) + 1
times = np.linspace(0, total_days, num_points)

# 初始化绘图数据存储字典
plot_data = {name: {'x': [], 'y': [], 'z': []} for name in tracked_bodies}

output_file = f"simulation_data_{years}yrs.txt"
print(f"开始模拟 {years} 年的数据演化...")
print(f"位置和速度数据将保存在: {output_file}")

with open(output_file, 'w') as f:
    # 写入表头
    header = "Time(days) " + " ".join([f"{n}_X {n}_Y {n}_Z {n}_VX {n}_VY {n}_VZ" for n in tracked_bodies])
    f.write(header + "\n")

    for i, t in enumerate(times):
        if i % (max(1, num_points // 10)) == 0:
            print(f"  计算进度... {i/num_points*100:.0f}%")
            
        sim.integrate(t)
        
        line_data = [f"{t:.4f}"]
        
        for name in tracked_bodies:
            p = sim.particles[body_names.index(name)]
            # 记录用于绘图的三维坐标
            plot_data[name]['x'].append(p.x)
            plot_data[name]['y'].append(p.y)
            plot_data[name]['z'].append(p.z)
            
            # 格式化当前天体的状态写入文件
            line_data.append(f"{p.x:.8e} {p.y:.8e} {p.z:.8e} {p.vx:.8e} {p.vy:.8e} {p.vz:.8e}")
            
        f.write(" ".join(line_data) + "\n")

print("模拟与数据存储完成。开始绘制黄道面轨迹图...")

# ==========================================
# 5. 坐标系转换 (赤道面 ICRF -> 黄道面 Ecliptic) 与 绘图
# ==========================================
# 地球自转轴倾角约为 23.439281 度
epsilon = np.radians(23.439281)

cos_e = np.cos(epsilon)
sin_e = np.sin(epsilon)

def to_ecliptic_xy(x_eq, y_eq, z_eq):
    """将 ICRF 赤道面坐标绕 X 轴旋转，投影到黄道面的 X-Y 上"""
    x_ecl = np.array(x_eq)
    y_ecl = np.array(y_eq) * cos_e + np.array(z_eq) * sin_e
    return x_ecl, y_ecl

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Subplot 1: 内行星黄道面轨迹 ---
colors = {'sun': 'orange', 'mercury': 'gray', 'venus': 'gold', 'earth': 'blue', 'mars': 'red'}
for name in ['sun', 'mercury', 'venus', 'earth', 'mars']:
    x_ecl, y_ecl = to_ecliptic_xy(plot_data[name]['x'], plot_data[name]['y'], plot_data[name]['z'])
    ax1.plot(x_ecl, y_ecl, label=name.capitalize(), color=colors[name], linewidth=1)
    # 画出终点位置
    ax1.scatter(x_ecl[-1], y_ecl[-1], color=colors[name], s=20)

ax1.set_aspect('equal')
ax1.set_xlabel('Ecliptic X (AU)')
ax1.set_ylabel('Ecliptic Y (AU)')
ax1.set_title(f"Inner Planets Ecliptic Trajectories ({years} Years, Alpha={modified_gravity_residual.__defaults__ if hasattr(modified_gravity_residual, '__defaults__') else '0.01'})")
ax1.legend()
ax1.grid(True)

# --- Subplot 2: 以地球为中心的月球黄道面轨迹 ---
e_x, e_y = to_ecliptic_xy(plot_data['earth']['x'], plot_data['earth']['y'], plot_data['earth']['z'])
m_x, m_y = to_ecliptic_xy(plot_data['moon']['x'], plot_data['moon']['y'], plot_data['moon']['z'])

# 计算月球相对于地球的黄道面坐标
moon_rel_x = m_x - e_x
moon_rel_y = m_y - e_y

ax2.plot(moon_rel_x, moon_rel_y, label='Moon Orbit', color='gray', linewidth=0.8)
ax2.scatter(0, 0, color='blue', label='Earth (Center)', s=50, zorder=5) # 地球固定在原点
ax2.set_aspect('equal')
ax2.set_xlabel('Relative Ecliptic X (AU)')
ax2.set_ylabel('Relative Ecliptic Y (AU)')
ax2.set_title(f"Moon Ecliptic Trajectory Relative to Earth ({years} Years)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(f'solar_system_orbits_{years}yrs.png', dpi=300, bbox_inches='tight')
print(f"轨迹图已保存为 solar_system_orbits_{years}yrs.png")
plt.show()