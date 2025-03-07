import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import time
from torch.utils.tensorboard import SummaryWriter


# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 'Heiti SC'
plt.rcParams['axes.unicode_minus'] = False 

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 设置设备（GPU或CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义地质模型参数
class GeoParams:
    def __init__(self):
        # 地质参数
        self.quartz_fraction = 0.30
        self.feldspar_fraction = 0.45
        self.amphibole_fraction = 0.15
        self.mica_fraction = 0.05
        self.others_fraction = 0.05
        
        # 化学参数
        self.initial_pH = 6.5
        self.acid_pH = 4.0
        self.feldspar_dissolution_rate = 1e-10  # mol/(m²·s)
        self.quartz_dissolution_rate = 3e-12    # mol/(m²·s)
        
        # 力学参数
        self.initial_youngs_modulus = 65.0  # GPa
        self.initial_ucs = 200.0  # MPa
        self.poissons_ratio = 0.26
        self.initial_porosity = 0.02  # 2%
        self.initial_permeability = 1e-16  # m²
        
        # 地热参数
        self.thermal_gradient = 25.0  # °C/km
        self.surface_temperature = 4.5  # °C
        self.thermal_conductivity = 2.8  # W/(m·K)
        
        # 域参数
        self.Lx = 500.0  # 域宽度 (m)
        self.Ly = 500.0  # 域高度 (m)
        self.Lz = 100.0  # 域深度 (m)
        self.T = 100.0   # 模拟总时间 (years)
        
        # 流体参数
        self.fluid_viscosity = 8.9e-4  # Pa·s at 25°C
        self.fluid_density = 1000.0  # kg/m³
        self.p_inlet = 3.0e6  # Pa
        self.p_outlet = 1.0e6  # Pa
        
        # 反应参数
        self.activation_energy_feldspar = 65.0  # kJ/mol
        self.activation_energy_quartz = 90.0  # kJ/mol
        self.gas_constant = 8.314e-3  # kJ/(mol·K)
        self.specific_surface_area = 1.0  # m²/g
        
        # 裂隙网络参数
        self.fracture_density = 0.2  # 裂隙密度
        self.fracture_aperture = 100e-6  # 裂隙开度 (m)
        self.main_fracture_orientation = np.array([[1, 1], [-1, 1]])  # 主要裂隙方向
        
        # 污染物参数
        self.metal_distribution_coefficient = 0.5  # 分配系数
        self.metal_diffusion_coefficient = 1e-9  # m²/s
        
    def calculate_temperature(self, z):
        """计算给定深度的温度"""
        return self.surface_temperature + self.thermal_gradient * z / 1000.0
    
    def calculate_fracture_influence(self, x, y, z):
        """计算裂隙网络影响因子"""
        # 创建两条主要裂隙线
        f1 = torch.abs(y - x) / (self.Lx + self.Ly) * 2  # NW-SE 方向
        f2 = torch.abs(y - (self.Ly - x)) / (self.Lx + self.Ly) * 2  # SW-NE 方向
        
        # 加入z方向衰减
        depth_factor = torch.exp(-z / (self.Lz * 0.5))
        
        # 计算最小距离（裂隙影响）
        fracture_influence = (1.0 - torch.min(f1, f2)) * depth_factor
        
        # 加入非线性转换
        fracture_influence = torch.sigmoid(fracture_influence * 10 - 2) * 0.9 + 0.1
        
        return fracture_influence
    
    def kozeny_carman(self, porosity):
        """根据Kozeny-Carman方程计算渗透率"""
        k0 = self.initial_permeability
        phi0 = self.initial_porosity
        return k0 * (porosity**3 * (1 - phi0)**2) / (phi0**3 * (1 - porosity)**2)
    
    def calculate_mechanical_properties(self, porosity):
        """计算基于孔隙率的力学性能"""
        # 计算杨氏模量变化
        E0 = self.initial_youngs_modulus
        E = E0 * torch.exp(-7.0 * (porosity - self.initial_porosity))
        
        # 计算强度变化
        UCS0 = self.initial_ucs
        UCS = UCS0 * torch.exp(-8.0 * (porosity - self.initial_porosity))
        
        return E, UCS

# 定义物理信息神经网络
class PINN(nn.Module):
    def __init__(self, params):
        super(PINN, self).__init__()
        self.params = params
        
        # 神经网络架构 - 8层，每层有64个神经元
        self.layers = nn.Sequential(
            nn.Linear(4, 64),  # 输入: x, y, z, t
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5)   # 输出: porosity, pH, feldspar_conc, quartz_conc, metal_conc
        )
        
        # 初始化参数
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, y, z, t):
        X = torch.cat([x, y, z, t], dim=1)
        output = self.layers(X)
        
        # 分解输出
        porosity = torch.sigmoid(output[:, 0:1]) * 0.15 + 0.005  # 范围0.5%-15%
        pH = torch.sigmoid(output[:, 1:2]) * 4.0 + 3.0  # 范围3-7
        feldspar_conc = torch.relu(output[:, 2:3]) * 0.01  # 正值，最大0.01 mol/m³
        quartz_conc = torch.relu(output[:, 3:4]) * 0.001  # 正值，最大0.001 mol/m³
        metal_conc = torch.relu(output[:, 4:5]) * 0.0001  # 正值，最大0.0001 mol/m³
        
        return porosity, pH, feldspar_conc, quartz_conc, metal_conc
    
    def compute_permeability(self, porosity):
        """计算渗透率"""
        return self.params.kozeny_carman(porosity)
    
    def compute_dissolution_rate(self, pH, temperature):
        """计算矿物溶解速率"""
        # 温度修正因子 (Arrhenius方程)
        T_ref = 298.15  # K
        T = temperature + 273.15  # K
        
        # 长石溶解速率
        E_act_feldspar = self.params.activation_energy_feldspar
        feldspar_temp_factor = torch.exp(-E_act_feldspar/self.params.gas_constant * (1/T - 1/T_ref))
        feldspar_rate = self.params.feldspar_dissolution_rate * torch.pow(10, -0.5 * (pH - 7.0)) * feldspar_temp_factor
        
        # 石英溶解速率
        E_act_quartz = self.params.activation_energy_quartz
        quartz_temp_factor = torch.exp(-E_act_quartz/self.params.gas_constant * (1/T - 1/T_ref))
        quartz_rate = self.params.quartz_dissolution_rate * torch.pow(10, -0.3 * (pH - 7.0)) * quartz_temp_factor
        
        return feldspar_rate, quartz_rate

# 定义物理损失函数
class PhysicsLoss:
    def __init__(self, model, params):
        self.model = model
        self.params = params
    
    def compute_spatial_gradients(self, porosity, pH, x, y, z, t):
        """计算空间梯度"""
        # 计算孔隙率梯度
        dphi_dx = torch.autograd.grad(
            porosity, x, grad_outputs=torch.ones_like(porosity),
            create_graph=True, retain_graph=True
        )[0]
        
        dphi_dy = torch.autograd.grad(
            porosity, y, grad_outputs=torch.ones_like(porosity),
            create_graph=True, retain_graph=True
        )[0]
        
        dphi_dz = torch.autograd.grad(
            porosity, z, grad_outputs=torch.ones_like(porosity),
            create_graph=True, retain_graph=True
        )[0]
        
        # 计算pH梯度
        dpH_dx = torch.autograd.grad(
            pH, x, grad_outputs=torch.ones_like(pH),
            create_graph=True, retain_graph=True
        )[0]
        
        dpH_dy = torch.autograd.grad(
            pH, y, grad_outputs=torch.ones_like(pH),
            create_graph=True, retain_graph=True
        )[0]
        
        dpH_dz = torch.autograd.grad(
            pH, z, grad_outputs=torch.ones_like(pH),
            create_graph=True, retain_graph=True
        )[0]
        
        return {
            'dphi_dx': dphi_dx, 'dphi_dy': dphi_dy, 'dphi_dz': dphi_dz,
            'dpH_dx': dpH_dx, 'dpH_dy': dpH_dy, 'dpH_dz': dpH_dz
        }
    
    def compute_time_derivatives(self, porosity, pH, feldspar_conc, quartz_conc, metal_conc, t):
        """计算时间导数"""
        dphi_dt = torch.autograd.grad(
            porosity, t, grad_outputs=torch.ones_like(porosity),
            create_graph=True, retain_graph=True
        )[0]
        
        dpH_dt = torch.autograd.grad(
            pH, t, grad_outputs=torch.ones_like(pH),
            create_graph=True, retain_graph=True
        )[0]
        
        dfeldspar_dt = torch.autograd.grad(
            feldspar_conc, t, grad_outputs=torch.ones_like(feldspar_conc),
            create_graph=True, retain_graph=True
        )[0]
        
        dquartz_dt = torch.autograd.grad(
            quartz_conc, t, grad_outputs=torch.ones_like(quartz_conc),
            create_graph=True, retain_graph=True
        )[0]
        
        dmetal_dt = torch.autograd.grad(
            metal_conc, t, grad_outputs=torch.ones_like(metal_conc),
            create_graph=True, retain_graph=True
        )[0]
        
        return {
            'dphi_dt': dphi_dt, 'dpH_dt': dpH_dt,
            'dfeldspar_dt': dfeldspar_dt, 'dquartz_dt': dquartz_dt,
            'dmetal_dt': dmetal_dt
        }
    
    def compute_residuals(self, x, y, z, t):
        """计算PDE残差"""
        # 将输入转为需要梯度的张量
        x_tensor = torch.tensor(x, requires_grad=True, device=device)
        y_tensor = torch.tensor(y, requires_grad=True, device=device)
        z_tensor = torch.tensor(z, requires_grad=True, device=device)
        t_tensor = torch.tensor(t, requires_grad=True, device=device)
        
        # 前向传播获取预测值
        porosity, pH, feldspar_conc, quartz_conc, metal_conc = self.model(
            x_tensor, y_tensor, z_tensor, t_tensor
        )
        
        # 计算温度
        temperature = torch.tensor(self.params.calculate_temperature(z), device=device)
        
        # 计算裂隙影响因子
        fracture_factor = self.params.calculate_fracture_influence(x_tensor, y_tensor, z_tensor)
        
        # 计算渗透率
        permeability = self.model.compute_permeability(porosity)
        
        # 计算溶解速率
        feldspar_rate, quartz_rate = self.model.compute_dissolution_rate(pH, temperature)
        
        # 计算空间梯度
        gradients = self.compute_spatial_gradients(porosity, pH, x_tensor, y_tensor, z_tensor, t_tensor)
        
        # 计算时间导数
        derivatives = self.compute_time_derivatives(
            porosity, pH, feldspar_conc, quartz_conc, metal_conc, t_tensor
        )
        
        # 计算对流-扩散-反应方程残差
        
        # 孔隙率方程残差
        # dphi/dt = R_feldspar * V_m,feldspar + R_quartz * V_m,quartz
        V_m_feldspar = 1e-4  # 长石摩尔体积 (m³/mol)
        V_m_quartz = 2.3e-5  # 石英摩尔体积 (m³/mol)
        
        dissolution_contribution = (
            feldspar_rate * V_m_feldspar * self.params.feldspar_fraction +
            quartz_rate * V_m_quartz * self.params.quartz_fraction
        )
        
        # 考虑裂隙影响
        effective_dissolution = dissolution_contribution * (1.0 + 5.0 * fracture_factor)
        
        porosity_residual = derivatives['dphi_dt'] - effective_dissolution
        
        # pH方程残差
        # 简化模型：pH变化与长石溶解速率相关
        buffer_capacity = 0.01  # pH缓冲能力
        pH_residual = derivatives['dpH_dt'] + buffer_capacity * (pH - self.params.initial_pH) - 0.5 * feldspar_rate
        
        # 浓度方程残差
        # 长石浓度方程
        D_feldspar = 1e-9  # 扩散系数 (m²/s)
        feldspar_diffusion = D_feldspar * (
            gradients['dphi_dx'] * gradients['dphi_dx'] + 
            gradients['dphi_dy'] * gradients['dphi_dy'] + 
            gradients['dphi_dz'] * gradients['dphi_dz']
        )
        feldspar_reaction = feldspar_rate * self.params.feldspar_fraction
        feldspar_residual = derivatives['dfeldspar_dt'] - feldspar_diffusion - feldspar_reaction
        
        # 石英浓度方程
        D_quartz = 5e-10  # 扩散系数 (m²/s)
        quartz_diffusion = D_quartz * (
            gradients['dphi_dx'] * gradients['dphi_dx'] + 
            gradients['dphi_dy'] * gradients['dphi_dy'] + 
            gradients['dphi_dz'] * gradients['dphi_dz']
        )
        quartz_reaction = quartz_rate * self.params.quartz_fraction
        quartz_residual = derivatives['dquartz_dt'] - quartz_diffusion - quartz_reaction
        
        # 重金属浓度方程
        D_metal = self.params.metal_diffusion_coefficient  # 扩散系数 (m²/s)
        metal_diffusion = D_metal * (
            gradients['dphi_dx'] * gradients['dphi_dx'] + 
            gradients['dphi_dy'] * gradients['dphi_dy'] + 
            gradients['dphi_dz'] * gradients['dphi_dz']
        )
        # 金属释放与pH和长石溶解相关
        metal_release = 0.01 * feldspar_rate * torch.exp(-2.0 * (pH - 4.0))
        metal_residual = derivatives['dmetal_dt'] - metal_diffusion - metal_release
        
        return {
            'porosity': porosity_residual,
            'pH': pH_residual,
            'feldspar': feldspar_residual,
            'quartz': quartz_residual,
            'metal': metal_residual
        }
    
    def compute_loss(self, residuals):
        """计算总损失"""
        loss_porosity = torch.mean(torch.square(residuals['porosity']))
        loss_pH = torch.mean(torch.square(residuals['pH']))
        loss_feldspar = torch.mean(torch.square(residuals['feldspar']))
        loss_quartz = torch.mean(torch.square(residuals['quartz']))
        loss_metal = torch.mean(torch.square(residuals['metal']))
        
        # 总损失
        total_loss = (
            1.0 * loss_porosity +
            1.0 * loss_pH +
            0.1 * loss_feldspar +
            0.1 * loss_quartz +
            0.1 * loss_metal
        )
        
        return total_loss, {
            'porosity': loss_porosity.item(),
            'pH': loss_pH.item(),
            'feldspar': loss_feldspar.item(),
            'quartz': loss_quartz.item(),
            'metal': loss_metal.item(),
            'total': total_loss.item()
        }

# 边界条件损失
class BoundaryConditionsLoss:
    def __init__(self, model, params):
        self.model = model
        self.params = params
    
    def initial_condition_loss(self, x, y, z):
        """初始条件损失"""
        batch_size = x.shape[0]
        t = torch.zeros((batch_size, 1), device=device)
        
        porosity, pH, feldspar_conc, quartz_conc, metal_conc = self.model(x, y, z, t)
        
        # 计算裂隙影响因子
        fracture_factor = self.params.calculate_fracture_influence(x, y, z)
        
        # 初始孔隙率与裂隙影响相关
        target_porosity = self.params.initial_porosity * (1.0 + 2.0 * fracture_factor)
        
        # 初始pH值为均匀分布
        target_pH = torch.ones_like(pH, device=device) * self.params.initial_pH
        
        # 初始浓度均为零
        target_feldspar = torch.zeros_like(feldspar_conc, device=device)
        target_quartz = torch.zeros_like(quartz_conc, device=device)
        target_metal = torch.zeros_like(metal_conc, device=device)
        
        loss_porosity = torch.mean(torch.square(porosity - target_porosity))
        loss_pH = torch.mean(torch.square(pH - target_pH))
        loss_feldspar = torch.mean(torch.square(feldspar_conc - target_feldspar))
        loss_quartz = torch.mean(torch.square(quartz_conc - target_quartz))
        loss_metal = torch.mean(torch.square(metal_conc - target_metal))
        
        total_loss = loss_porosity + loss_pH + loss_feldspar + loss_quartz + loss_metal
        
        return total_loss
    
    def west_boundary_loss(self, y, z, t):
        """西边界条件（酸性溶液注入）"""
        batch_size = y.shape[0]
        x = torch.zeros((batch_size, 1), device=device)
        
        porosity, pH, feldspar_conc, quartz_conc, metal_conc = self.model(x, y, z, t)
        
        # 西边界pH条件
        target_pH = torch.ones_like(pH, device=device) * self.params.acid_pH
        
        # 西边界金属浓度条件 - 随时间变化
        metal_input = 5e-5 * torch.sin(t * np.pi / 25) ** 2  # 周期性污染
        
        loss_pH = torch.mean(torch.square(pH - target_pH))
        loss_metal = torch.mean(torch.square(metal_conc - metal_input))
        
        total_loss = loss_pH + 0.5 * loss_metal
        
        return total_loss
    
    def east_boundary_loss(self, y, z, t):
        """东边界条件（出流边界）"""
        batch_size = y.shape[0]
        x = torch.ones((batch_size, 1), device=device) * self.params.Lx
        
        # 对于出流边界，我们设置导数为零（Neumann条件）
        x_tensor = torch.tensor(x.detach().clone(), requires_grad=True, device=device)
        y_tensor = torch.tensor(y.detach().clone(), device=device)
        z_tensor = torch.tensor(z.detach().clone(), device=device)
        t_tensor = torch.tensor(t.detach().clone(), device=device)
        
        porosity, pH, feldspar_conc, quartz_conc, metal_conc = self.model(
            x_tensor, y_tensor, z_tensor, t_tensor
        )
        
        # 计算x方向梯度
        dphi_dx = torch.autograd.grad(
            porosity, x_tensor, grad_outputs=torch.ones_like(porosity),
            create_graph=True, retain_graph=True
        )[0]
        
        dpH_dx = torch.autograd.grad(
            pH, x_tensor, grad_outputs=torch.ones_like(pH),
            create_graph=True, retain_graph=True
        )[0]
        
        loss_dphi_dx = torch.mean(torch.square(dphi_dx))
        loss_dpH_dx = torch.mean(torch.square(dpH_dx))
        
        total_loss = loss_dphi_dx + loss_dpH_dx
        
        return total_loss
    
    def compute_boundary_loss(self, x, y, z, t):
        """计算所有边界条件损失"""
        # 初始条件
        batch_size = x.shape[0]
        ic_indices = np.random.choice(batch_size, size=int(batch_size*0.2), replace=False)
        loss_ic = self.initial_condition_loss(
            x[ic_indices], y[ic_indices], z[ic_indices]
        )
        
        # 西边界条件
        west_indices = np.random.choice(batch_size, size=int(batch_size*0.1), replace=False)
        loss_west = self.west_boundary_loss(
            y[west_indices], z[west_indices], t[west_indices]
        )
        
        # 东边界条件
        east_indices = np.random.choice(batch_size, size=int(batch_size*0.1), replace=False)
        loss_east = self.east_boundary_loss(
            y[east_indices], z[east_indices], t[east_indices]
        )
        
        # 总边界损失
        total_boundary_loss = 10.0 * loss_ic + 5.0 * loss_west + 1.0 * loss_east
        
        return total_boundary_loss, {
            'initial': loss_ic.item(),
            'west': loss_west.item(),
            'east': loss_east.item(),
            'total': total_boundary_loss.item()
        }

# 生成训练数据
def generate_training_data(params, n_points=10000):
    """生成训练数据点"""
    # 内部点
    x = np.random.uniform(0, params.Lx, (n_points, 1))
    y = np.random.uniform(0, params.Ly, (n_points, 1))
    z = np.random.uniform(0, params.Lz, (n_points, 1))
    t = np.random.uniform(0, params.T, (n_points, 1))
    
    # 转换为张量
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
    
    return x_tensor, y_tensor, z_tensor, t_tensor

# 训练模型
def train_model(model, params, n_epochs=20000, batch_size=1024, lr=1e-4):
    """训练PINN模型"""
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, verbose=True
    )
    
    # 创建损失计算器
    physics_loss = PhysicsLoss(model, params)
    bc_loss = BoundaryConditionsLoss(model, params)
    
    # 创建TensorBoard日志
    writer = SummaryWriter('runs/granite_erosion')
    
    # 生成训练数据
    x, y, z, t = generate_training_data(params, n_points=100000)
    
    # 训练循环
    n_batches = len(x) // batch_size
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # 打乱数据
        permutation = torch.randperm(len(x))
        running_loss = 0.0
        
        for i in range(n_batches):
            # 获取mini-batch
            indices = permutation[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch = x[indices], y[indices]
            z_batch, t_batch = z[indices], t[indices]
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 计算PDE残差
            residuals = physics_loss.compute_residuals(x_batch, y_batch, z_batch, t_batch)
            
            # 计算物理损失
            physics_total_loss, physics_losses = physics_loss.compute_loss(residuals)
            
            # 计算边界条件损失
            bc_total_loss, bc_losses = bc_loss.compute_boundary_loss(x_batch, y_batch, z_batch, t_batch)
            
            # 总损失
            loss = physics_total_loss + bc_total_loss
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()
        
        # 计算平均损失
        epoch_loss = running_loss / n_batches
        
        # 更新学习率
        scheduler.step(epoch_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Total', epoch_loss, epoch)
        writer.add_scalar('Loss/Physics', physics_losses['total'], epoch)
        writer.add_scalar('Loss/Boundary', bc_losses['total'], epoch)
        
        # 打印损失
        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                  f"Loss: {epoch_loss:.6f}, "
                  f"Physics: {physics_losses['total']:.6f}, "
                  f"Boundary: {bc_losses['total']:.6f}, "
                  f"Time: {elapsed:.2f}s")
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 每1000个epoch可视化一次结果
        if (epoch + 1) % 1000 == 0:
            visualize_results(model, params, epoch+1)
    
    writer.close()
    print("Training completed!")
    
    return model

# 可视化结果
def visualize_results(model, params, epoch=None):
    """可视化PINN模型的预测结果"""
    model.eval()  # 设置为评估模式
    
    # 创建可视化网格
    x_grid = np.linspace(0, params.Lx, 100)
    y_grid = np.linspace(0, params.Ly, 100)
    z_value = params.Lz / 2  # 中间深度的切片
    
    X, Y = np.meshgrid(x_grid, y_grid)
    X_flat = X.flatten()[:, np.newaxis]
    Y_flat = Y.flatten()[:, np.newaxis]
    Z_flat = np.ones_like(X_flat) * z_value
    
    # 选择时间点
    time_points = [0, 25, 50, 75, 100]  # 年
    
    for t_value in time_points:
        T_flat = np.ones_like(X_flat) * t_value
        
        # 转换为张量
        x_tensor = torch.tensor(X_flat, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(Y_flat, dtype=torch.float32, device=device)
        z_tensor = torch.tensor(Z_flat, dtype=torch.float32, device=device)
        t_tensor = torch.tensor(T_flat, dtype=torch.float32, device=device)
        
        # 批量预测以避免内存问题
        batch_size = 10000
        n_batches = len(x_tensor) // batch_size + (1 if len(x_tensor) % batch_size != 0 else 0)
        
        porosity_list = []
        pH_list = []
        feldspar_list = []
        quartz_list = []
        metal_list = []
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(x_tensor))
                
                porosity_batch, pH_batch, feldspar_batch, quartz_batch, metal_batch = model(
                    x_tensor[start_idx:end_idx],
                    y_tensor[start_idx:end_idx],
                    z_tensor[start_idx:end_idx],
                    t_tensor[start_idx:end_idx]
                )
                
                porosity_list.append(porosity_batch.cpu().numpy())
                pH_list.append(pH_batch.cpu().numpy())
                feldspar_list.append(feldspar_batch.cpu().numpy())
                quartz_list.append(quartz_batch.cpu().numpy())
                metal_list.append(metal_batch.cpu().numpy())
        
        # 合并批次结果
        porosity_flat = np.vstack(porosity_list)
        pH_flat = np.vstack(pH_list)
        feldspar_flat = np.vstack(feldspar_list)
        quartz_flat = np.vstack(quartz_list)
        metal_flat = np.vstack(metal_list)
        
        # 重塑为网格形状
        porosity_grid = porosity_flat.reshape(X.shape)
        pH_grid = pH_flat.reshape(X.shape)
        feldspar_grid = feldspar_flat.reshape(X.shape)
        quartz_grid = quartz_flat.reshape(X.shape)
        metal_grid = metal_flat.reshape(X.shape)
        
        # 计算岩石力学性能
        E_grid, UCS_grid = params.calculate_mechanical_properties(torch.tensor(porosity_flat))
        E_grid = E_grid.cpu().numpy().reshape(X.shape)
        UCS_grid = UCS_grid.cpu().numpy().reshape(X.shape)
        
        # 创建图像
        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Chemical-Mechanical Evolution - Time: {t_value} years', fontsize=16)
        
        # 自定义颜色映射
        porosity_cmap = plt.cm.viridis
        pH_cmap = plt.cm.plasma
        concentration_cmap = plt.cm.YlOrBr
        mechanical_cmap = plt.cm.RdYlGn
        
        # 绘制孔隙率
        im1 = axs[0, 0].imshow(porosity_grid, origin='lower', extent=[0, params.Lx, 0, params.Ly],
                              cmap=porosity_cmap, vmin=0, vmax=0.15)
        axs[0, 0].set_title('Porosity')
        axs[0, 0].set_xlabel('X (m)')
        axs[0, 0].set_ylabel('Y (m)')
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im1, cax=cax)
        
        # 绘制pH值
        im2 = axs[0, 1].imshow(pH_grid, origin='lower', extent=[0, params.Lx, 0, params.Ly],
                              cmap=pH_cmap, vmin=3, vmax=7)
        axs[0, 1].set_title('pH Value')
        axs[0, 1].set_xlabel('X (m)')
        axs[0, 1].set_ylabel('Y (m)')
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax)
        
        # 绘制长石溶解浓度
        im3 = axs[1, 0].imshow(feldspar_grid, origin='lower', extent=[0, params.Lx, 0, params.Ly],
                              cmap=concentration_cmap, vmin=0, vmax=0.01)
        axs[1, 0].set_title('Feldspar Dissolution (mol/m³)')
        axs[1, 0].set_xlabel('X (m)')
        axs[1, 0].set_ylabel('Y (m)')
        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im3, cax=cax)
        
        # 绘制石英溶解浓度
        im4 = axs[1, 1].imshow(quartz_grid, origin='lower', extent=[0, params.Lx, 0, params.Ly],
                              cmap=concentration_cmap, vmin=0, vmax=0.001)
        axs[1, 1].set_title('Quartz Dissolution (mol/m³)')
        axs[1, 1].set_xlabel('X (m)')
        axs[1, 1].set_ylabel('Y (m)')
        divider = make_axes_locatable(axs[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im4, cax=cax)
        
        # 绘制杨氏模量变化
        im5 = axs[2, 0].imshow(E_grid, origin='lower', extent=[0, params.Lx, 0, params.Ly],
                              cmap=mechanical_cmap, vmin=20, vmax=70)
        axs[2, 0].set_title('Young\'s Modulus (GPa)')
        axs[2, 0].set_xlabel('X (m)')
        axs[2, 0].set_ylabel('Y (m)')
        divider = make_axes_locatable(axs[2, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im5, cax=cax)
        
        # 绘制单轴抗压强度变化
        im6 = axs[2, 1].imshow(UCS_grid, origin='lower', extent=[0, params.Lx, 0, params.Ly],
                              cmap=mechanical_cmap, vmin=50, vmax=200)
        axs[2, 1].set_title('UCS (MPa)')
        axs[2, 1].set_xlabel('X (m)')
        axs[2, 1].set_ylabel('Y (m)')
        divider = make_axes_locatable(axs[2, 1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im6, cax=cax)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图像
        save_dir = 'results'
        os.makedirs(save_dir, exist_ok=True)
        
        if epoch is not None:
            plt.savefig(f'{save_dir}/result_epoch{epoch}_time{t_value}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{save_dir}/result_time{t_value}.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    print(f"Visualization completed for time points: {time_points}")

# 运行模型
def main():
    # 创建参数对象
    params = GeoParams()
    
    # 创建模型
    model = PINN(params).to(device)
    
    # 训练模型
    model = train_model(model, params, n_epochs=20000, lr=1e-4)
    
    # 可视化最终结果
    visualize_results(model, params)

if __name__ == "__main__":
    main()