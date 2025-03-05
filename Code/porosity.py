import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import time

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 'Heiti SC'
plt.rcParams['axes.unicode_minus'] = False 

# 设置随机种子确保结果可重复
torch.manual_seed(1234)
np.random.seed(1234)

class PorosityEvolutionPINN(nn.Module):
    def __init__(self, hidden_layers=6, neurons=50):
        super(PorosityEvolutionPINN, self).__init__()
        
        # 输入层: (x, y, z, t) -> 输出层: (孔隙率, 渗透率, 矿物浓度1, 矿物浓度2)
        # 矿物浓度1代表原始矿物(如长石)，矿物浓度2代表二次矿物(如石膏和黄铁矿)
        
        layers = [nn.Linear(4, neurons), nn.Tanh()]
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
            
        # 输出4个变量
        layers.append(nn.Linear(neurons, 4))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        outputs = self.network(inputs)
        
        # 分离输出并应用适当的激活函数
        porosity = torch.sigmoid(outputs[:, 0:1]) * 0.5  # 孔隙率范围: 0-0.5
        permeability = torch.exp(outputs[:, 1:2] - 10)   # 渗透率采用对数尺度
        mineral1 = torch.sigmoid(outputs[:, 2:3])        # 原始矿物的体积分数: 0-1
        mineral2 = torch.sigmoid(outputs[:, 3:4]) * 0.3  # 二次矿物的体积分数: 0-0.3
        
        return porosity, permeability, mineral1, mineral2
    
    def compute_derivatives(self, x, y, z, t):
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        t.requires_grad_(True)
        
        porosity, permeability, mineral1, mineral2 = self.forward(x, y, z, t)
        
        # 计算时间导数
        porosity_t = torch.autograd.grad(
            porosity, t, 
            grad_outputs=torch.ones_like(porosity),
            create_graph=True,
            retain_graph=True
        )[0]
        
        mineral1_t = torch.autograd.grad(
            mineral1, t,
            grad_outputs=torch.ones_like(mineral1),
            create_graph=True,
            retain_graph=True
        )[0]
        
        mineral2_t = torch.autograd.grad(
            mineral2, t,
            grad_outputs=torch.ones_like(mineral2),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 计算空间导数 (用于流体流动)
        porosity_x = torch.autograd.grad(
            porosity, x,
            grad_outputs=torch.ones_like(porosity),
            create_graph=True,
            retain_graph=True
        )[0]
        
        porosity_y = torch.autograd.grad(
            porosity, y,
            grad_outputs=torch.ones_like(porosity),
            create_graph=True,
            retain_graph=True
        )[0]
        
        porosity_z = torch.autograd.grad(
            porosity, z,
            grad_outputs=torch.ones_like(porosity),
            create_graph=True,
            retain_graph=True
        )[0]
        
        permeability_x = torch.autograd.grad(
            permeability, x,
            grad_outputs=torch.ones_like(permeability),
            create_graph=True,
            retain_graph=True
        )[0]
        
        permeability_y = torch.autograd.grad(
            permeability, y,
            grad_outputs=torch.ones_like(permeability),
            create_graph=True,
            retain_graph=True
        )[0]
        
        permeability_z = torch.autograd.grad(
            permeability, z,
            grad_outputs=torch.ones_like(permeability),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return (porosity, permeability, mineral1, mineral2,
                porosity_t, mineral1_t, mineral2_t,
                porosity_x, porosity_y, porosity_z,
                permeability_x, permeability_y, permeability_z)

# 模型参数
class Parameters:
    def __init__(self):
        # 时空尺度
        self.length_scale = 1000.0  # 1000米
        self.time_scale = 100.0    # 100年
        self.depth_scale = 1000.0  # 1000米
        
        # 地质参数
        self.initial_porosity = 0.05  # 初始孔隙率
        self.initial_permeability = 1e-15  # 初始渗透率 (m^2)
        self.initial_mineral1 = 0.8  # 初始原生矿物含量 (体积分数)
        self.initial_mineral2 = 0.01  # 初始二次矿物含量
        
        # 化学反应参数
        self.dissolution_rate = 1e-10  # 溶解速率常数
        self.precipitation_rate = 5e-11  # 沉淀速率常数
        self.activation_energy_dissolution = 50000  # 溶解活化能 (J/mol)
        self.activation_energy_precipitation = 80000  # 沉淀活化能 (J/mol)
        self.gas_constant = 8.314  # 气体常数
        
        # 水文参数
        self.fluid_viscosity = 1e-3  # 流体粘度 (Pa·s)
        self.fluid_density = 1000  # 流体密度 (kg/m^3)
        self.gravity = 9.8  # 重力加速度 (m/s^2)
        
        # 温度场 (线性温度梯度)
        self.surface_temperature = 283  # 地表温度 (K)
        self.temperature_gradient = 25 / 1000  # 温度梯度 (K/m)
        
        # 异质性设置 (区域内的裂隙、断层等)
        self.n_heterogeneities = 5  # 异质性区域数量
        self.heterogeneity_locations = torch.rand(self.n_heterogeneities, 3)  # 位置 (x,y,z)
        self.heterogeneity_strength = 0.5 + 0.5 * torch.rand(self.n_heterogeneities)  # 强度系数
        self.heterogeneity_range = 100 / self.length_scale  # 影响范围

# 生成训练数据
def generate_training_data(params, n_boundary=2000, n_initial=3000, n_collocation=20000):
    # 生成边界点 (区域边界)
    # 随机采样边界上的点
    
    # 确保边界点数是6的倍数
    n_boundary = (n_boundary // 6) * 6  # 这样每个面的点数相同，且总数是6的倍数
    points_per_face = n_boundary // 6
    
    # 生成边界点 (区域边界)
    # 随机采样边界上的点
    
    # 六个面的坐标
    faces = []
    
    # 面 x=0
    x_face1 = torch.zeros(points_per_face, 1)
    y_face1 = torch.rand(points_per_face, 1)
    z_face1 = torch.rand(points_per_face, 1)
    faces.append((x_face1, y_face1, z_face1))
    
    # 面 x=1
    x_face2 = torch.ones(points_per_face, 1)
    y_face2 = torch.rand(points_per_face, 1)
    z_face2 = torch.rand(points_per_face, 1)
    faces.append((x_face2, y_face2, z_face2))
    
    # 面 y=0
    x_face3 = torch.rand(points_per_face, 1)
    y_face3 = torch.zeros(points_per_face, 1)
    z_face3 = torch.rand(points_per_face, 1)
    faces.append((x_face3, y_face3, z_face3))
    
    # 面 y=1
    x_face4 = torch.rand(points_per_face, 1)
    y_face4 = torch.ones(points_per_face, 1)
    z_face4 = torch.rand(points_per_face, 1)
    faces.append((x_face4, y_face4, z_face4))
    
    # 面 z=0 (顶部)
    x_face5 = torch.rand(points_per_face, 1)
    y_face5 = torch.rand(points_per_face, 1)
    z_face5 = torch.zeros(points_per_face, 1)
    faces.append((x_face5, y_face5, z_face5))
    
    # 面 z=1 (底部)
    x_face6 = torch.rand(points_per_face, 1)
    y_face6 = torch.rand(points_per_face, 1)
    z_face6 = torch.ones(points_per_face, 1)
    faces.append((x_face6, y_face6, z_face6))
    
    # 合并所有面
    x_boundary = torch.cat([f[0] for f in faces])
    y_boundary = torch.cat([f[1] for f in faces])
    z_boundary = torch.cat([f[2] for f in faces])
    t_boundary = torch.rand(n_boundary, 1)  
    
    # 生成初始条件点 (t=0)
    x_initial = torch.rand(n_initial, 1)
    y_initial = torch.rand(n_initial, 1)
    z_initial = torch.rand(n_initial, 1)
    t_initial = torch.zeros(n_initial, 1)
    
    # 生成配置点 (用于PDE训练)
    x_collocation = torch.rand(n_collocation, 1)
    y_collocation = torch.rand(n_collocation, 1)
    z_collocation = torch.rand(n_collocation, 1)
    t_collocation = torch.rand(n_collocation, 1)
    
    # 计算异质性影响系数
    heterogeneity_factor = torch.ones_like(x_collocation)
    
    for i in range(params.n_heterogeneities):
        loc = params.heterogeneity_locations[i]
        strength = params.heterogeneity_strength[i]
        
        # 计算距离
        dist = torch.sqrt((x_collocation - loc[0])**2 + 
                          (y_collocation - loc[1])**2 + 
                          (z_collocation - loc[2])**2)
        
        # 更新异质性系数
        factor = 1 + strength * torch.exp(-dist / params.heterogeneity_range)
        heterogeneity_factor = heterogeneity_factor * factor
    
    return (x_boundary, y_boundary, z_boundary, t_boundary,
            x_initial, y_initial, z_initial, t_initial,
            x_collocation, y_collocation, z_collocation, t_collocation,
            heterogeneity_factor)

# 计算当前位置的温度 (根据深度)
def calculate_temperature(z, params):
    # z已归一化 (0到1)，将其转换回实际深度
    depth = z * params.depth_scale
    temperature = params.surface_temperature + depth * params.temperature_gradient
    return temperature

# 训练模型
def train_model(model, params, epochs=10000, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=300, factor=0.5, verbose=True)
    
    # 生成训练数据
    print("生成训练数据...")
    (x_boundary, y_boundary, z_boundary, t_boundary,
     x_initial, y_initial, z_initial, t_initial,
     x_collocation, y_collocation, z_collocation, t_collocation,
     heterogeneity_factor) = generate_training_data(params)
    
    # 计算温度场
    temperature_initial = calculate_temperature(z_initial, params)
    temperature_collocation = calculate_temperature(z_collocation, params)
    
    # 定义渗透率与孔隙率关系的Kozeny-Carman方程函数
    def kozeny_carman(porosity, initial_porosity, initial_permeability):
        return initial_permeability * ((porosity ** 3) / ((1 - porosity) ** 2)) * (((1 - initial_porosity) ** 2) / (initial_porosity ** 3))
    
    # 定义温度修正的反应速率函数
    def reaction_rate(base_rate, activation_energy, temperature):
        return base_rate * torch.exp(-activation_energy / (params.gas_constant * temperature))
    
    # 记录损失历史
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 1000  # 早停耐心值
    
    print(f"开始训练, 总共 {epochs} 轮...")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 计算边界条件损失
        porosity_boundary, permeability_boundary, mineral1_boundary, mineral2_boundary = model(
            x_boundary, y_boundary, z_boundary, t_boundary
        )
        
        # 边界渗透率与孔隙率应满足Kozeny-Carman关系
        expected_permeability_boundary = kozeny_carman(
            porosity_boundary, 
            params.initial_porosity, 
            params.initial_permeability
        )
        
        loss_boundary_perm = torch.mean((permeability_boundary - expected_permeability_boundary) ** 2)
        
        # 顶部边界条件: z=0处的特定条件
        top_indices = (z_boundary < 0.01).flatten()
        if torch.sum(top_indices) > 0:
            # 顶部有更高的溶解速率
            loss_top_boundary = torch.mean((porosity_boundary[top_indices] - 0.1 * t_boundary[top_indices]) ** 2)
        else:
            loss_top_boundary = torch.tensor(0.0)
        
        # 计算初始条件损失
        porosity_initial, permeability_initial, mineral1_initial, mineral2_initial = model(
            x_initial, y_initial, z_initial, t_initial
        )
        
        loss_initial_porosity = torch.mean((porosity_initial - params.initial_porosity) ** 2)
        loss_initial_permeability = torch.mean((permeability_initial - params.initial_permeability) ** 2)
        loss_initial_mineral1 = torch.mean((mineral1_initial - params.initial_mineral1) ** 2)
        loss_initial_mineral2 = torch.mean((mineral2_initial - params.initial_mineral2) ** 2)
        
        # 计算PDE约束损失
        (porosity, permeability, mineral1, mineral2,
         porosity_t, mineral1_t, mineral2_t,
         porosity_x, porosity_y, porosity_z,
         permeability_x, permeability_y, permeability_z) = model.compute_derivatives(
            x_collocation, y_collocation, z_collocation, t_collocation
        )
        
        # 温度依赖的反应速率
        temperature = calculate_temperature(z_collocation, params)
        dissolution_rate = reaction_rate(params.dissolution_rate, params.activation_energy_dissolution, temperature)
        precipitation_rate = reaction_rate(params.precipitation_rate, params.activation_energy_precipitation, temperature)
        
        # 应用异质性因子增强局部区域的溶解和沉淀速率
        dissolution_rate = dissolution_rate * heterogeneity_factor.reshape(-1, 1)
        precipitation_rate = precipitation_rate * heterogeneity_factor.reshape(-1, 1)
        
        # 1. 孔隙率演化方程: dφ/dt = -dMineral1/dt + dMineral2/dt
        # 矿物1的溶解速率正比于其浓度和水的存在(孔隙率)
        expected_mineral1_t = -dissolution_rate * mineral1 * porosity
        
        # 矿物2的沉淀速率正比于孔隙率和已溶解的矿物1量
        # 简化: 假设溶解的矿物1一部分转化为矿物2，一部分保持溶解状态
        conversion_efficiency = 0.3  # 30%的溶解物质转化为二次矿物
        expected_mineral2_t = conversion_efficiency * precipitation_rate * (1 - mineral1) * porosity * (1 - mineral2)
        
        # 孔隙率变化率应等于矿物体积变化率之和
        expected_porosity_t = -expected_mineral1_t - expected_mineral2_t
        
        # 2. 渗透率与孔隙率关系: Kozeny-Carman方程
        expected_permeability = kozeny_carman(porosity, params.initial_porosity, params.initial_permeability)
        
        # 各PDE损失项
        loss_porosity_pde = torch.mean((porosity_t - expected_porosity_t) ** 2)
        loss_mineral1_pde = torch.mean((mineral1_t - expected_mineral1_t) ** 2)
        loss_mineral2_pde = torch.mean((mineral2_t - expected_mineral2_t) ** 2)
        loss_permeability = torch.mean((permeability - expected_permeability) ** 2)
        
        # 添加约束: 确保矿物体积分数和孔隙率总和为1
        loss_volume_conservation = torch.mean((mineral1 + mineral2 + porosity - 1.0) ** 2)
        
        # 总损失
        loss = (0.1 * loss_boundary_perm + 0.1 * loss_top_boundary +
                0.2 * (loss_initial_porosity + loss_initial_permeability + loss_initial_mineral1 + loss_initial_mineral2) +
                0.4 * (loss_porosity_pde + loss_mineral1_pde + loss_mineral2_pde) +
                0.2 * loss_permeability +
                0.1 * loss_volume_conservation)
        

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        loss_history.append(loss.item())
        
        # 每500轮打印一次损失
        if (epoch+1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, 用时: {elapsed:.2f}秒")
            print(f"  边界损失: {loss_boundary_perm.item():.6f}, 顶部边界: {loss_top_boundary.item():.6f}")
            print(f"  初始孔隙率损失: {loss_initial_porosity.item():.6f}, 初始渗透率损失: {loss_initial_permeability.item():.6f}")
            print(f"  初始矿物1损失: {loss_initial_mineral1.item():.6f}, 初始矿物2损失: {loss_initial_mineral2.item():.6f}")
            print(f"  孔隙率PDE损失: {loss_porosity_pde.item():.6f}")
            print(f"  矿物1 PDE损失: {loss_mineral1_pde.item():.6f}, 矿物2 PDE损失: {loss_mineral2_pde.item():.6f}")
            print(f"  渗透率关系损失: {loss_permeability.item():.6f}")
            print(f"  体积守恒损失: {loss_volume_conservation.item():.6f}")
            start_time = time.time()
        
        # 早停策略
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return loss_history

# 可视化结果
def visualize_results(model, params):
    model.eval()
    
    print("正在生成可视化数据...")
    
    # 2D切片可视化
    n_points = 50
    
    # 创建网格
    x = torch.linspace(0, 1, n_points)
    y = torch.linspace(0, 1, n_points)
    z = torch.linspace(0, 1, n_points)
    
    # 不同时间点
    t_values = [0, 0.25, 0.5, 0.75, 1.0]  # 归一化时间: 0, 25, 50, 75, 100年
    
    # 准备不同切片的可视化
    # 1. x-y平面 (固定z值)
    fixed_z_values = [0.2, 0.5, 0.8]  # 浅、中、深层
    
    # 2. x-z平面 (固定y值)
    fixed_y = 0.5
    
    # 创建大图布局
    plt.figure(figsize=(20, 15))
    
    # 计数器
    plot_idx = 1
    
    # 对每个时间点
    for t_idx, t_val in enumerate(t_values):
        # 固定时间和y值
        t = torch.ones((n_points, n_points)) * t_val
        
        # === x-z切片（显示深度变化）===
        y_fixed = torch.ones((n_points, n_points)) * fixed_y
        
        X, Z = torch.meshgrid(x, z)
        X_flat = X.reshape(-1, 1)
        Z_flat = Z.reshape(-1, 1)
        Y_flat = y_fixed.reshape(-1, 1)
        T_flat = t.reshape(-1, 1)
        
        with torch.no_grad():
            porosity, permeability, mineral1, mineral2 = model(
                X_flat, Y_flat, Z_flat, T_flat
            )
        
        porosity = porosity.reshape(X.shape)
        permeability = permeability.reshape(X.shape)
        mineral1 = mineral1.reshape(X.shape)
        mineral2 = mineral2.reshape(X.shape)
        
        # 孔隙率
        plt.subplot(5, 4, plot_idx)
        plt.contourf(X.numpy(), Z.numpy(), porosity.numpy(), 50, cmap='viridis')
        plt.colorbar(label='孔隙率')
        plt.xlabel('X 坐标')
        plt.ylabel('深度 Z')
        plt.title(f'孔隙率 (t={t_val*params.time_scale:.0f}年, y={fixed_y*params.length_scale:.0f}m)')
        plt.gca().invert_yaxis()  # 深度轴反转
        plot_idx += 1
        
        # 渗透率
        plt.subplot(5, 4, plot_idx)
        plt.contourf(X.numpy(), Z.numpy(), np.log10(permeability.numpy()), 50, cmap='plasma')
        plt.colorbar(label='log10(渗透率)')
        plt.xlabel('X 坐标')
        plt.ylabel('深度 Z')
        plt.title(f'渗透率 (t={t_val*params.time_scale:.0f}年, y={fixed_y*params.length_scale:.0f}m)')
        plt.gca().invert_yaxis()
        plot_idx += 1
        
        # 矿物1
        plt.subplot(5, 4, plot_idx)
        plt.contourf(X.numpy(), Z.numpy(), mineral1.numpy(), 50, cmap='Blues')
        plt.colorbar(label='原始矿物含量')
        plt.xlabel('X 坐标')
        plt.ylabel('深度 Z')
        plt.title(f'原始矿物 (t={t_val*params.time_scale:.0f}年, y={fixed_y*params.length_scale:.0f}m)')
        plt.gca().invert_yaxis()
        plot_idx += 1
        
        # 矿物2
        plt.subplot(5, 4, plot_idx)
        plt.contourf(X.numpy(), Z.numpy(), mineral2.numpy(), 50, cmap='Reds')
        plt.colorbar(label='二次矿物含量')
        plt.xlabel('X 坐标')
        plt.ylabel('深度 Z')
        plt.title(f'二次矿物 (t={t_val*params.time_scale:.0f}年, y={fixed_y*params.length_scale:.0f}m)')
        plt.gca().invert_yaxis()
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('porosity_evolution_over_time.png', dpi=300)
    plt.show()
    
    
    # === 3D可视化最终状态 ===
    print("生成3D可视化...")
    fig = plt.figure(figsize=(20, 15))

    # 创建立方体顶点和体素
    n_points_3d = 20  # 降低点数以提高性能
    x_3d = torch.linspace(0, 1, n_points_3d)
    y_3d = torch.linspace(0, 1, n_points_3d)
    z_3d = torch.linspace(0, 1, n_points_3d)

    X_3d, Y_3d, Z_3d = torch.meshgrid(x_3d, y_3d, z_3d)

    points = torch.stack([X_3d.flatten(), Y_3d.flatten(), Z_3d.flatten()], dim=1)
    t_final = torch.ones((points.shape[0], 1))  # t=1 (100年)

    with torch.no_grad():
        porosity_3d, permeability_3d, mineral1_3d, mineral2_3d = model(
            points[:, 0:1], points[:, 1:2], points[:, 2:3], t_final
        )

    porosity_3d = porosity_3d.reshape(n_points_3d, n_points_3d, n_points_3d).numpy()
    permeability_3d = permeability_3d.reshape(n_points_3d, n_points_3d, n_points_3d).numpy()
    mineral1_3d = mineral1_3d.reshape(n_points_3d, n_points_3d, n_points_3d).numpy()
    mineral2_3d = mineral2_3d.reshape(n_points_3d, n_points_3d, n_points_3d).numpy()

    # 使用体素渲染创建3D图
    # 1. 孔隙率
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=porosity_3d.flatten(), cmap='viridis', 
                        s=50, alpha=0.05)

    # 添加高孔隙率等值面 - 修复等值面级别检查
    porosity_min, porosity_max = np.min(porosity_3d), np.max(porosity_3d)
    print(f"孔隙率范围: {porosity_min:.4f} 到 {porosity_max:.4f}")

    # 确保阈值在数据范围内
    high_porosity_threshold = min(0.15, porosity_max * 0.8)  # 取最大值的80%或0.15，以较小者为准
    if high_porosity_threshold > porosity_min:  # 确保阈值大于最小值
        try:
            verts, faces, _, _ = measure_marching_cubes(
                porosity_3d, 
                high_porosity_threshold, 
                spacing=(1.0/n_points_3d, 1.0/n_points_3d, 1.0/n_points_3d)
            )
            
            mesh = Poly3DCollection(verts[faces], alpha=0.3, color='orange')
            ax1.add_collection3d(mesh)
        except Exception as e:
            print(f"绘制孔隙率等值面时出错: {e}")
            # 如果等值面提取失败，则跳过绘制等值面
    else:
        print("孔隙率值范围太小，无法提取有意义的等值面")

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (深度)')
    ax1.set_title(f'100年后孔隙率分布 (等值面: {high_porosity_threshold:.4f})')
    fig.colorbar(scatter, ax=ax1, label='孔隙率')

    # 2. 渗透率 - 同样修复
    ax2 = fig.add_subplot(222, projection='3d')
    log_perm = np.log10(permeability_3d.flatten())
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=log_perm, cmap='plasma',
                        s=50, alpha=0.05)

    # 添加高渗透率等值面 - 同样检查值范围
    log_perm_volume = np.log10(permeability_3d)
    perm_min, perm_max = np.min(log_perm_volume), np.max(log_perm_volume)
    print(f"log10(渗透率)范围: {perm_min:.4f} 到 {perm_max:.4f}")

    # 根据实际数据范围调整阈值
    high_perm_threshold = (perm_min + perm_max) * 0.7  # 范围的70%处
    try:
        verts, faces, _, _ = measure_marching_cubes(
            log_perm_volume, 
            high_perm_threshold, 
            spacing=(1.0/n_points_3d, 1.0/n_points_3d, 1.0/n_points_3d)
        )
        
        mesh = Poly3DCollection(verts[faces], alpha=0.3, color='red')
        ax2.add_collection3d(mesh)
    except Exception as e:
        print(f"绘制渗透率等值面时出错: {e}")

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z (深度)')
    ax2.set_title(f'100年后渗透率分布 (log10, 等值面: {high_perm_threshold:.4f})')
    fig.colorbar(scatter2, ax=ax2, label='log10(渗透率)')

    # 3. 原始矿物 - 同样修复
    ax3 = fig.add_subplot(223, projection='3d')
    scatter3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=mineral1_3d.flatten(), cmap='Blues',
                        s=50, alpha=0.05)

    # 添加低原始矿物等值面 (高溶解区域)
    mineral1_min, mineral1_max = np.min(mineral1_3d), np.max(mineral1_3d)
    print(f"原始矿物含量范围: {mineral1_min:.4f} 到 {mineral1_max:.4f}")

    low_mineral1_threshold = mineral1_min + (mineral1_max - mineral1_min) * 0.3  # 范围的30%处
    try:
        verts, faces, _, _ = measure_marching_cubes(
            mineral1_3d, 
            low_mineral1_threshold, 
            spacing=(1.0/n_points_3d, 1.0/n_points_3d, 1.0/n_points_3d)
        )
        
        mesh = Poly3DCollection(verts[faces], alpha=0.3, color='blue')
        ax3.add_collection3d(mesh)
    except Exception as e:
        print(f"绘制原始矿物等值面时出错: {e}")

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z (深度)')
    ax3.set_title(f'100年后原始矿物分布 (等值面: {low_mineral1_threshold:.4f})')
    fig.colorbar(scatter3, ax=ax3, label='原始矿物含量')

    # 4. 二次矿物 - 同样修复
    ax4 = fig.add_subplot(224, projection='3d')
    scatter4 = ax4.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=mineral2_3d.flatten(), cmap='Reds',
                        s=50, alpha=0.05)

    # 添加高二次矿物等值面
    mineral2_min, mineral2_max = np.min(mineral2_3d), np.max(mineral2_3d)
    print(f"二次矿物含量范围: {mineral2_min:.4f} 到 {mineral2_max:.4f}")

    high_mineral2_threshold = mineral2_min + (mineral2_max - mineral2_min) * 0.7  # 范围的70%处
    try:
        verts, faces, _, _ = measure_marching_cubes(
            mineral2_3d, 
            high_mineral2_threshold, 
            spacing=(1.0/n_points_3d, 1.0/n_points_3d, 1.0/n_points_3d)
        )
        
        mesh = Poly3DCollection(verts[faces], alpha=0.3, color='red')
        ax4.add_collection3d(mesh)
    except Exception as e:
        print(f"绘制二次矿物等值面时出错: {e}")

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z (深度)')
    ax4.set_title(f'100年后二次矿物分布 (等值面: {high_mineral2_threshold:.4f})')
    fig.colorbar(scatter4, ax=ax4, label='二次矿物含量')
        
        
    
    
    plt.tight_layout()
    plt.savefig('3d_porosity_final_state.png', dpi=300)
    plt.show()
    
    # === 随深度变化的剖面图 ===
    plt.figure(figsize=(15, 10))
    
    # 选择几个固定的x, y位置进行深度剖面展示
    fixed_positions = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)]
    colors = ['blue', 'green', 'red']
    
    z_profile = torch.linspace(0, 1, 100)
    
    for t_idx, t_val in enumerate([0, 0.5, 1.0]):  # 0年、50年、100年
        plt.subplot(2, 2, t_idx+1)
        
        for i, (x_pos, y_pos) in enumerate(fixed_positions):
            # 创建深度剖面点
            x_profile = torch.ones_like(z_profile) * x_pos
            y_profile = torch.ones_like(z_profile) * y_pos
            t_profile = torch.ones_like(z_profile) * t_val
            
            with torch.no_grad():
                porosity_profile, permeability_profile, mineral1_profile, mineral2_profile = model(
                    x_profile.unsqueeze(1), 
                    y_profile.unsqueeze(1), 
                    z_profile.unsqueeze(1), 
                    t_profile.unsqueeze(1)
                )
            
            # 绘制孔隙率随深度变化曲线
            plt.plot(porosity_profile.numpy(), z_profile.numpy(), 
                     color=colors[i], linewidth=2,
                     label=f'位置 ({x_pos*params.length_scale:.0f}m, {y_pos*params.length_scale:.0f}m)')
        
        plt.ylim(1, 0)  # 反转y轴以表示深度增加
        plt.xlabel('孔隙率')
        plt.ylabel('深度 (归一化)')
        plt.title(f'时间 t={t_val*params.time_scale:.0f}年的孔隙率-深度剖面')
        plt.legend()
        plt.grid(True)
    
    # 绘制渗透率随时间变化
    plt.subplot(2, 2, 4)
    
    x_center, y_center, z_center = 0.5, 0.5, 0.5  # 区域中心点
    t_series = torch.linspace(0, 1, 100)
    
    x_series = torch.ones_like(t_series) * x_center
    y_series = torch.ones_like(t_series) * y_center
    z_series = torch.ones_like(t_series) * z_center
    
    with torch.no_grad():
        porosity_series, permeability_series, mineral1_series, mineral2_series = model(
            x_series.unsqueeze(1), 
            y_series.unsqueeze(1), 
            z_series.unsqueeze(1), 
            t_series.unsqueeze(1)
        )
    
    # 转换为numpy数组用于绘图
    t_np = t_series.numpy() * params.time_scale
    porosity_np = porosity_series.numpy()
    perm_np = permeability_series.numpy()
    mineral1_np = mineral1_series.numpy()
    mineral2_np = mineral2_series.numpy()
    
    # 创建双y轴图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    
    # 左y轴：孔隙率和矿物含量
    ax1.plot(t_np, porosity_np, 'b-', label='孔隙率')
    ax1.plot(t_np, mineral1_np, 'g-', label='原始矿物')
    ax1.plot(t_np, mineral2_np, 'r-', label='二次矿物')
    ax1.set_xlabel('时间 (年)')
    ax1.set_ylabel('体积分数')
    ax1.set_ylim(0, 1)
    
    # 右y轴：渗透率（对数尺度）
    ax2.plot(t_np, np.log10(perm_np), 'k--', label='log10(渗透率)')
    ax2.set_ylabel('log10(渗透率)')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'区域中心点 ({x_center*params.length_scale:.0f}m, {y_center*params.length_scale:.0f}m, {z_center*params.depth_scale:.0f}m) 随时间变化')
    plt.tight_layout()
    plt.savefig('porosity_depth_profiles.png', dpi=300)
    plt.show()
    
    # === 特定深度的空间分布动画 ===
    print("创建动画可视化...")
    fixed_depth = 0.3  # 固定深度
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建网格
    x_anim = torch.linspace(0, 1, n_points)
    y_anim = torch.linspace(0, 1, n_points)
    X_anim, Y_anim = torch.meshgrid(x_anim, y_anim)
    
    X_flat = X_anim.flatten().unsqueeze(1)
    Y_flat = Y_anim.flatten().unsqueeze(1)
    Z_flat = torch.ones_like(X_flat) * fixed_depth
    
    # 初始化图像
    contour = ax.contourf(X_anim.numpy(), Y_anim.numpy(), 
                         np.zeros((n_points, n_points)), 
                         50, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('孔隙率')
    title = ax.set_title('孔隙率演化动画 (深度: 300m)')
    ax.set_xlabel('X 坐标 (m)')
    ax.set_ylabel('Y 坐标 (m)')
    
    # 更新函数
    def update(frame):
        frame = frame / 50.0  # 将帧数转换为归一化时间
        t_val = torch.ones_like(X_flat) * frame
        
        with torch.no_grad():
            porosity_frame, _, _, _ = model(X_flat, Y_flat, Z_flat, t_val)
        
        porosity_frame = porosity_frame.reshape(n_points, n_points).numpy()
        
        # 清除旧的等值线并绘制新的
        ax.clear()
        contour = ax.contourf(X_anim.numpy() * params.length_scale, 
                             Y_anim.numpy() * params.length_scale, 
                             porosity_frame, 
                             50, cmap='viridis')
        ax.set_title(f'深度{fixed_depth*params.depth_scale:.0f}m处孔隙率演化 (t={frame*params.time_scale:.1f}年)')
        ax.set_xlabel('X 坐标 (m)')
        ax.set_ylabel('Y 坐标 (m)')
        
        return contour,
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=51, interval=200, blit=False)
    
    # 保存动画
    anim.save('porosity_evolution.gif', writer='pillow', fps=10, dpi=100)
    
    plt.show()
    
    print("可视化完成!")

# marching cubes算法实现（用于3D等值面提取）
def measure_marching_cubes(volume, level, spacing=(1, 1, 1)):
    """
    使用marching cubes算法提取体积数据中的等值面
    """
    
    # 使用scikit-image的marching cubes实现
    verts, faces, normals, values = measure.marching_cubes(volume, level, spacing=spacing)
    
    return verts, faces, normals, values

# 主函数
def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # 导入需要的库

    # 初始化参数
    params = Parameters()
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PorosityEvolutionPINN().to(device)
    # 然后将所有张量移到该设备
    # x_boundary = x_boundary.to(device)
    
    # 训练模型
    print("开始训练模型...")
    loss_history = train_model(model, params, epochs=5000, learning_rate=1e-3)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值 (对数尺度)')
    plt.title('训练损失曲线')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('training_loss.png', dpi=300)
    plt.show()
    
    # 可视化结果
    visualize_results(model, params)

if __name__ == "__main__":
    main()