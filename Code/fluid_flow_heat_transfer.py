import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import time
import os
from datetime import datetime

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 'Heiti SC'
plt.rcParams['axes.unicode_minus'] = False 

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置随机种子以确保结果可重复
torch.manual_seed(1234)
np.random.seed(1234)

# 创建保存结果的目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"pinn_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"结果将保存到: {results_dir}")

# 定义问题的物理参数
class PhysicalParams:
    def __init__(self):
        # 几何参数 (m)
        self.Lx = 500.0
        self.Ly = 500.0
        self.Lz = 100.0
        
        # 时间参数 (年)
        self.T_max = 100.0  # 最大模拟时间 (年)
        
        # 花岗岩参数
        self.k0_granite = 1e-16  # 初始渗透率 (m^2)
        self.phi0_granite = 0.05  # 初始孔隙率
        
        # 石英闪长岩参数
        self.k0_diorite = 5e-17  # 初始渗透率 (m^2)
        self.phi0_diorite = 0.03  # 初始孔隙率
        
        # 流体参数
        self.mu = 1e-3  # 流体黏度 (Pa·s)
        self.rho_f = 1000.0  # 流体密度 (kg/m^3)
        self.cf = 4200.0  # 流体比热容 (J/kg·K)
        
        # 固体参数
        self.rho_s = 2650.0  # 固体密度 (kg/m^3)
        self.cs = 800.0  # 固体比热容 (J/kg·K)
        self.lambda_s = 3.0  # 固体热导率 (W/m·K)
        
        # 反应参数
        self.Da = 1e-10  # 达姆克勒数 (无量纲)
        self.Ea = 40000.0  # 活化能 (J/mol)
        self.R = 8.314  # 气体常数 (J/mol·K)
        self.alpha = 0.5  # 孔隙率-渗透率关系指数
        self.beta = 3.0  # 孔隙率-渗透率关系指数
        
        # 边界条件
        self.p_top = 1e5  # 顶部压力 (Pa)
        self.p_bottom = 1.1e5  # 底部压力 (Pa)
        self.T_top = 283.15  # 顶部温度 (K)
        self.T_bottom = 293.15  # 底部温度 (K)
        
        # 重力加速度
        self.g = 9.81  # 重力加速度 (m/s^2)
        
        # 化学反应相关
        self.C0 = 0.0  # 初始浓度 (mol/m^3)
        self.D = 1e-9  # 扩散系数 (m^2/s)

params = PhysicalParams()

# 定义PINN网络结构
class PINN(nn.Module):
    def __init__(self, hidden_layers, neurons_per_layer):
        super(PINN, self).__init__()
        
        # 输入层: [x, y, z, t]
        self.input_layer = nn.Linear(4, neurons_per_layer)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        
        # 输出层: [p, T, phi, k, C]
        # p: 压力, T: 温度, phi: 孔隙率, k: 渗透率, C: 浓度
        self.output_layer = nn.Linear(neurons_per_layer, 5)
        
        # 激活函数
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        outputs = self.output_layer(x)
        
        # 使用sigmoid和缩放确保输出物理合理性
        p = outputs[:, 0:1]  # 压力不做约束
        T = self.T_top + (self.T_bottom - self.T_top) * torch.sigmoid(outputs[:, 1:2])  # 温度范围[T_top, T_bottom]
        phi = 0.01 + 0.29 * torch.sigmoid(outputs[:, 2:3])  # 孔隙率范围[0.01, 0.3]
        k = torch.exp(outputs[:, 3:4] - 35)  # 渗透率范围[~1e-18, ~1e-14]
        C = torch.sigmoid(outputs[:, 4:5])  # 浓度范围[0, 1]
        
        return torch.cat([p, T, phi, k, C], dim=1)
    
    # 物理参数作为神经网络的属性
    @property
    def T_top(self):
        return torch.tensor(params.T_top, device=device)
    
    @property
    def T_bottom(self):
        return torch.tensor(params.T_bottom, device=device)

# 初始化PINN模型
model = PINN(hidden_layers=6, neurons_per_layer=80).to(device)
print(f"模型结构: {model}")

# 定义损失函数计算模块
class PINNLoss:
    def __init__(self, model, params):
        self.model = model
        self.params = params
    
    def compute_gradients(self, x, y, z, t):
        """计算物理量对x, y, z, t的梯度"""
        inputs = torch.cat([x, y, z, t], dim=1).requires_grad_(True)
        outputs = self.model(inputs)
        
        p, T, phi, k, C = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:, 3:4], outputs[:, 4:5]
        
        # 计算一阶梯度
        dp_dx = torch.autograd.grad(p, inputs, grad_outputs=torch.ones_like(p), 
                                    create_graph=True)[0][:, 0:1]
        dp_dy = torch.autograd.grad(p, inputs, grad_outputs=torch.ones_like(p), 
                                    create_graph=True)[0][:, 1:2]
        dp_dz = torch.autograd.grad(p, inputs, grad_outputs=torch.ones_like(p), 
                                    create_graph=True)[0][:, 2:3]
        dp_dt = torch.autograd.grad(p, inputs, grad_outputs=torch.ones_like(p), 
                                    create_graph=True)[0][:, 3:4]
        
        dT_dx = torch.autograd.grad(T, inputs, grad_outputs=torch.ones_like(T), 
                                   create_graph=True)[0][:, 0:1]
        dT_dy = torch.autograd.grad(T, inputs, grad_outputs=torch.ones_like(T), 
                                   create_graph=True)[0][:, 1:2]
        dT_dz = torch.autograd.grad(T, inputs, grad_outputs=torch.ones_like(T), 
                                   create_graph=True)[0][:, 2:3]
        dT_dt = torch.autograd.grad(T, inputs, grad_outputs=torch.ones_like(T), 
                                   create_graph=True)[0][:, 3:4]
        
        dphi_dt = torch.autograd.grad(phi, inputs, grad_outputs=torch.ones_like(phi), 
                                     create_graph=True)[0][:, 3:4]
        
        dC_dx = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), 
                                   create_graph=True)[0][:, 0:1]
        dC_dy = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), 
                                   create_graph=True)[0][:, 1:2]
        dC_dz = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), 
                                   create_graph=True)[0][:, 2:3]
        dC_dt = torch.autograd.grad(C, inputs, grad_outputs=torch.ones_like(C), 
                                   create_graph=True)[0][:, 3:4]
        
        # 计算二阶梯度
        d2p_dx2 = torch.autograd.grad(dp_dx, inputs, grad_outputs=torch.ones_like(dp_dx), 
                                     create_graph=True)[0][:, 0:1]
        d2p_dy2 = torch.autograd.grad(dp_dy, inputs, grad_outputs=torch.ones_like(dp_dy), 
                                     create_graph=True)[0][:, 1:2]
        d2p_dz2 = torch.autograd.grad(dp_dz, inputs, grad_outputs=torch.ones_like(dp_dz), 
                                     create_graph=True)[0][:, 2:3]
        
        d2T_dx2 = torch.autograd.grad(dT_dx, inputs, grad_outputs=torch.ones_like(dT_dx), 
                                     create_graph=True)[0][:, 0:1]
        d2T_dy2 = torch.autograd.grad(dT_dy, inputs, grad_outputs=torch.ones_like(dT_dy), 
                                     create_graph=True)[0][:, 1:2]
        d2T_dz2 = torch.autograd.grad(dT_dz, inputs, grad_outputs=torch.ones_like(dT_dz), 
                                     create_graph=True)[0][:, 2:3]
        
        d2C_dx2 = torch.autograd.grad(dC_dx, inputs, grad_outputs=torch.ones_like(dC_dx), 
                                     create_graph=True)[0][:, 0:1]
        d2C_dy2 = torch.autograd.grad(dC_dy, inputs, grad_outputs=torch.ones_like(dC_dy), 
                                     create_graph=True)[0][:, 1:2]
        d2C_dz2 = torch.autograd.grad(dC_dz, inputs, grad_outputs=torch.ones_like(dC_dz), 
                                     create_graph=True)[0][:, 2:3]
        
        return {
            'p': p, 'T': T, 'phi': phi, 'k': k, 'C': C,
            'dp_dx': dp_dx, 'dp_dy': dp_dy, 'dp_dz': dp_dz, 'dp_dt': dp_dt,
            'dT_dx': dT_dx, 'dT_dy': dT_dy, 'dT_dz': dT_dz, 'dT_dt': dT_dt,
            'dphi_dt': dphi_dt,
            'dC_dx': dC_dx, 'dC_dy': dC_dy, 'dC_dz': dC_dz, 'dC_dt': dC_dt,
            'd2p_dx2': d2p_dx2, 'd2p_dy2': d2p_dy2, 'd2p_dz2': d2p_dz2,
            'd2T_dx2': d2T_dx2, 'd2T_dy2': d2T_dy2, 'd2T_dz2': d2T_dz2,
            'd2C_dx2': d2C_dx2, 'd2C_dy2': d2C_dy2, 'd2C_dz2': d2C_dz2
        }
    
    def rock_type(self, x, y, z):
        """定义岩石类型，这里使用简单的几何划分"""
        # 这里假设x<250为花岗岩，x>=250为石英闪长岩
        rock_type = torch.zeros_like(x)
        rock_type[x >= 250] = 1  # 1表示石英闪长岩，0表示花岗岩
        return rock_type
    
    def pde_loss(self, x, y, z, t):
        """计算PDE方程的损失"""
        grads = self.compute_gradients(x, y, z, t)
        rock_type = self.rock_type(x, y, z)
        
        # 提取变量和梯度
        p, T, phi, k, C = grads['p'], grads['T'], grads['phi'], grads['k'], grads['C']
        
        # 根据岩石类型设置初始参数
        k0 = torch.where(rock_type == 0, 
                         torch.ones_like(x) * self.params.k0_granite, 
                         torch.ones_like(x) * self.params.k0_diorite)
        phi0 = torch.where(rock_type == 0, 
                          torch.ones_like(x) * self.params.phi0_granite, 
                          torch.ones_like(x) * self.params.phi0_diorite)
        
        # Darcy流动方程: ∇·(k/μ·∇p - ρ·g·∇z) = 0
        # 简化为: ∇²p = 0 (假设垂直方向的重力效应已在初始和边界条件中考虑)
        darcy_eq = grads['d2p_dx2'] + grads['d2p_dy2'] + grads['d2p_dz2']
        
        # 能量方程: (ρc)_eff·∂T/∂t + (ρc)_f·u·∇T = ∇·(λ_eff·∇T)
        # 有效热容: (ρc)_eff = φ·(ρc)_f + (1-φ)·(ρc)_s
        # 有效热导率: λ_eff = φ·λ_f + (1-φ)·λ_s
        # 流速: u = -k/μ·∇p
        rho_c_eff = phi * self.params.rho_f * self.params.cf + (1 - phi) * self.params.rho_s * self.params.cs
        lambda_eff = phi * 0.6 + (1 - phi) * self.params.lambda_s  # 假设流体热导率为0.6 W/m·K
        
        # 计算Darcy流速
        ux = -k / self.params.mu * grads['dp_dx']
        uy = -k / self.params.mu * grads['dp_dy']
        uz = -k / self.params.mu * grads['dp_dz']
        
        # 对流项
        convection = self.params.rho_f * self.params.cf * (ux * grads['dT_dx'] + uy * grads['dT_dy'] + uz * grads['dT_dz'])
        
        # 传导项
        conduction = lambda_eff * (grads['d2T_dx2'] + grads['d2T_dy2'] + grads['d2T_dz2'])
        
        energy_eq = rho_c_eff * grads['dT_dt'] + convection - conduction
        
        # 孔隙率演化方程: ∂φ/∂t = Da·(1-φ)·exp(-Ea/RT)·C
        reaction_rate = self.params.Da * (1 - phi) * torch.exp(-self.params.Ea / (self.params.R * T)) * C
        porosity_eq = grads['dphi_dt'] - reaction_rate
        
        # 渗透率与孔隙率关系: k = k0·(φ/φ0)^β
        k_relation = k - k0 * torch.pow(phi / phi0, self.params.beta)
        
        # 浓度守恒方程: ∂C/∂t + u·∇C = D·∇²C - R
        # R是反应源项，与孔隙率变化率相关
        diffusion = self.params.D * (grads['d2C_dx2'] + grads['d2C_dy2'] + grads['d2C_dz2'])
        advection = ux * grads['dC_dx'] + uy * grads['dC_dy'] + uz * grads['dC_dz']
        concentration_eq = grads['dC_dt'] + advection - diffusion + reaction_rate
        
        # 计算总损失
        loss_darcy = torch.mean(torch.square(darcy_eq))
        loss_energy = torch.mean(torch.square(energy_eq))
        loss_porosity = torch.mean(torch.square(porosity_eq))
        loss_k_relation = torch.mean(torch.square(k_relation))
        loss_concentration = torch.mean(torch.square(concentration_eq))
        
        pde_loss = loss_darcy + loss_energy + loss_porosity + loss_k_relation + loss_concentration
        
        return pde_loss, {
            'darcy': loss_darcy.item(),
            'energy': loss_energy.item(),
            'porosity': loss_porosity.item(),
            'k_relation': loss_k_relation.item(),
            'concentration': loss_concentration.item()
        }
    
    def ic_loss(self, x, y, z, t):
        """计算初始条件的损失"""
        outputs = self.model(torch.cat([x, y, z, t], dim=1))
        p, T, phi, k, C = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:, 3:4], outputs[:, 4:5]
        
        rock_type = self.rock_type(x, y, z)
        
        # 设置初始条件
        p_init = self.params.p_top + (self.params.p_bottom - self.params.p_top) * z / self.params.Lz
        T_init = self.params.T_top + (self.params.T_bottom - self.params.T_top) * z / self.params.Lz
        
        phi_init = torch.where(rock_type == 0, 
                              torch.ones_like(x) * self.params.phi0_granite, 
                              torch.ones_like(x) * self.params.phi0_diorite)
        
        k_init = torch.where(rock_type == 0, 
                            torch.ones_like(x) * self.params.k0_granite, 
                            torch.ones_like(x) * self.params.k0_diorite)
        
        C_init = torch.ones_like(x) * self.params.C0
        
        # 计算初始条件损失
        loss_p_ic = torch.mean(torch.square(p - p_init))
        loss_T_ic = torch.mean(torch.square(T - T_init))
        loss_phi_ic = torch.mean(torch.square(phi - phi_init))
        loss_k_ic = torch.mean(torch.square(k - k_init))
        loss_C_ic = torch.mean(torch.square(C - C_init))
        
        ic_loss = loss_p_ic + loss_T_ic + loss_phi_ic + loss_k_ic + loss_C_ic
        
        return ic_loss, {
            'p_ic': loss_p_ic.item(),
            'T_ic': loss_T_ic.item(),
            'phi_ic': loss_phi_ic.item(),
            'k_ic': loss_k_ic.item(),
            'C_ic': loss_C_ic.item()
        }
    
    def bc_loss(self, x, y, z, t):
        """计算边界条件的损失"""
        # 创建边界点的输入
        # 顶部边界 (z=0)
        top_boundary = torch.cat([x, y, torch.zeros_like(x), t], dim=1)
        # 底部边界 (z=Lz)
        bottom_boundary = torch.cat([x, y, torch.ones_like(x) * self.params.Lz, t], dim=1)
        
        # 预测边界值
        top_outputs = self.model(top_boundary)
        bottom_outputs = self.model(bottom_boundary)
        
        p_top, T_top = top_outputs[:, 0:1], top_outputs[:, 1:2]
        p_bottom, T_bottom = bottom_outputs[:, 0:1], bottom_outputs[:, 1:2]
        
        # 计算边界条件损失
        loss_p_top = torch.mean(torch.square(p_top - self.params.p_top))
        loss_p_bottom = torch.mean(torch.square(p_bottom - self.params.p_bottom))
        loss_T_top = torch.mean(torch.square(T_top - self.params.T_top))
        loss_T_bottom = torch.mean(torch.square(T_bottom - self.params.T_bottom))
        
        bc_loss = loss_p_top + loss_p_bottom + loss_T_top + loss_T_bottom
        
        return bc_loss, {
            'p_top': loss_p_top.item(),
            'p_bottom': loss_p_bottom.item(),
            'T_top': loss_T_top.item(),
            'T_bottom': loss_T_bottom.item()
        }
    
    def total_loss(self, x_pde, y_pde, z_pde, t_pde, x_ic, y_ic, z_ic, t_ic, x_bc, y_bc, z_bc, t_bc):
        """计算总损失"""
        pde_loss, pde_components = self.pde_loss(x_pde, y_pde, z_pde, t_pde)
        ic_loss, ic_components = self.ic_loss(x_ic, y_ic, z_ic, t_ic)
        bc_loss, bc_components = self.bc_loss(x_bc, y_bc, z_bc, t_bc)
        
        # 使用加权损失函数
        total = pde_loss + 10.0 * ic_loss + 10.0 * bc_loss
        
        return total, {
            'total': total.item(),
            'pde': pde_loss.item(),
            'ic': ic_loss.item(),
            'bc': bc_loss.item(),
            **pde_components,
            **ic_components,
            **bc_components
        }

# 创建损失函数计算器
loss_calculator = PINNLoss(model, params)

# 生成训练数据点
def generate_training_points(params, n_pde=10000, n_ic=2000, n_bc=2000):
    # 生成PDE内部点
    x_pde = torch.rand(n_pde, 1, device=device) * params.Lx
    y_pde = torch.rand(n_pde, 1, device=device) * params.Ly
    z_pde = torch.rand(n_pde, 1, device=device) * params.Lz
    t_pde = torch.rand(n_pde, 1, device=device) * params.T_max
    
    # 生成初始条件点 (t=0)
    x_ic = torch.rand(n_ic, 1, device=device) * params.Lx
    y_ic = torch.rand(n_ic, 1, device=device) * params.Ly
    z_ic = torch.rand(n_ic, 1, device=device) * params.Lz
    t_ic = torch.zeros(n_ic, 1, device=device)
    
    # 生成边界条件点
    x_bc = torch.rand(n_bc, 1, device=device) * params.Lx
    y_bc = torch.rand(n_bc, 1, device=device) * params.Ly
    z_bc = torch.cat([
        torch.zeros(n_bc // 2, 1, device=device),  # 顶部
        torch.ones(n_bc // 2, 1, device=device) * params.Lz  # 底部
    ])
    t_bc = torch.rand(n_bc, 1, device=device) * params.T_max
    
    return x_pde, y_pde, z_pde, t_pde, x_ic, y_ic, z_ic, t_ic, x_bc, y_bc, z_bc, t_bc

# 生成训练数据
x_pde, y_pde, z_pde, t_pde, x_ic, y_ic, z_ic, t_ic, x_bc, y_bc, z_bc, t_bc = generate_training_points(params)

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, verbose=True)

# 定义训练过程
def train_model(model, optimizer, scheduler, loss_calculator, epochs=20000):
    start_time = time.time()
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 1000  # 早停耐心值
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss, components = loss_calculator.total_loss(
            x_pde, y_pde, z_pde, t_pde, 
            x_ic, y_ic, z_ic, t_ic, 
            x_bc, y_bc, z_bc, t_bc
        )
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        loss_history.append(components)
        
        # 早停判断
        if components['total'] < best_loss:
            best_loss = components['total']
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"{results_dir}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 打印训练进度
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}, Time: {elapsed:.2f}s, Total Loss: {components['total']:.6e}, "
                  f"PDE: {components['pde']:.6e}, IC: {components['ic']:.6e}, BC: {components['bc']:.6e}")
            
            # 每1000个epoch保存一次损失历史
            if epoch % 1000 == 0:
                np.save(f"{results_dir}/loss_history_{epoch}.npy", loss_history)
    
    # 训练结束后保存最终模型和损失历史
    torch.save(model.state_dict(), f"{results_dir}/final_model.pt")
    np.save(f"{results_dir}/loss_history.npy", loss_history)
    
    return loss_history

# 训练模型
print("开始训练模型...")
loss_history = train_model(model, optimizer, scheduler, loss_calculator)
print("模型训练完成!")

# 加载最佳模型
model.load_state_dict(torch.load(f"{results_dir}/best_model.pt"))
model.eval()

# 定义结果可视化函数
def visualize_results(model, params, times_to_visualize=[0, 25, 50, 75, 100]):
    """可视化不同时间点的模拟结果"""
    plt.figure(figsize=(20, 15))
    
    # 创建网格点进行预测
    nx, ny, nz = 30, 30, 10
    x = np.linspace(0, params.Lx, nx)
    y = np.linspace(0, params.Ly, ny)
    z = np.linspace(0, params.Lz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 创建颜色映射
    granite_diorite_cmap = LinearSegmentedColormap.from_list(
        'GraniteDiorite', 
        [(0.95, 0.95, 0.95), (0.7, 0.7, 0.7)]
    )
    
    # 为每个要可视化的时间点创建图
    for i, t_val in enumerate(times_to_visualize):
        print(f"生成时间 {t_val} 年的可视化结果...")
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f"时间: {t_val} 年", fontsize=24)
        
        # 创建子图
        gs = gridspec.GridSpec(2, 3)
        
        # 岩石类型分布图 (只需显示一次，因为不随时间变化)
        if i == 0:
            rock_type_fig = plt.figure(figsize=(10, 8))
            ax_rock = rock_type_fig.add_subplot(111, projection='3d')
            
            # 创建三维点云
            x_pts = X.flatten()
            y_pts = Y.flatten()
            z_pts = Z.flatten()
            rock_type = np.zeros_like(x_pts)
            rock_type[x_pts >= 250] = 1  # 1表示石英闪长岩，0表示花岗岩
            
            # 绘制三维散点图
            scatter = ax_rock.scatter(x_pts, y_pts, z_pts, c=rock_type, cmap=granite_diorite_cmap, 
                                     alpha=0.6, marker='o', s=10)
            
            ax_rock.set_title("岩石类型分布 (灰色: 石英闪长岩, 白色: 花岗岩)", fontsize=16)
            ax_rock.set_xlabel("X (m)", fontsize=14)
            ax_rock.set_ylabel("Y (m)", fontsize=14)
            ax_rock.set_zlabel("Z (m)", fontsize=14)
            
            plt.colorbar(scatter, ax=ax_rock, label="岩石类型")
            rock_type_fig.savefig(f"{results_dir}/rock_type_distribution.png", dpi=300, bbox_inches='tight')
            plt.close(rock_type_fig)
        
        # 将三维坐标和时间转换为张量
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten(), np.full_like(X.flatten(), t_val)])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        # 使用模型预测
        with torch.no_grad():
            predictions = model(points_tensor)
        
        # 将预测结果转换回numpy数组
        p = predictions[:, 0].cpu().numpy().reshape(X.shape)
        T = predictions[:, 1].cpu().numpy().reshape(X.shape)
        phi = predictions[:, 2].cpu().numpy().reshape(X.shape)
        k = predictions[:, 3].cpu().numpy().reshape(X.shape)
        C = predictions[:, 4].cpu().numpy().reshape(X.shape)
        
        # 绘制中间z切片的孔隙率
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(phi[:, :, nz//2].T, origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                       cmap='viridis', vmin=0.01, vmax=0.3)
        ax1.set_title(f"孔隙率 (z={z[nz//2]:.1f}m)", fontsize=16)
        ax1.set_xlabel("X (m)", fontsize=14)
        ax1.set_ylabel("Y (m)", fontsize=14)
        plt.colorbar(im1, ax=ax1)
        
        # 绘制中间z切片的渗透率
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(np.log10(k[:, :, nz//2].T), origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                       cmap='plasma')
        ax2.set_title(f"渗透率 (log10, z={z[nz//2]:.1f}m)", fontsize=16)
        ax2.set_xlabel("X (m)", fontsize=14)
        ax2.set_ylabel("Y (m)", fontsize=14)
        plt.colorbar(im2, ax=ax2)
        
        # 绘制中间z切片的温度
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(T[:, :, nz//2].T - 273.15, origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                       cmap='coolwarm')
        ax3.set_title(f"温度 (°C, z={z[nz//2]:.1f}m)", fontsize=16)
        ax3.set_xlabel("X (m)", fontsize=14)
        ax3.set_ylabel("Y (m)", fontsize=14)
        plt.colorbar(im3, ax=ax3)
        
        # 绘制中间y切片的孔隙率
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(phi[:, ny//2, :].T, origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='viridis', vmin=0.01, vmax=0.3)
        ax4.set_title(f"孔隙率 (y={y[ny//2]:.1f}m)", fontsize=16)
        ax4.set_xlabel("X (m)", fontsize=14)
        ax4.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im4, ax=ax4)
        
        # 绘制中间y切片的渗透率
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(np.log10(k[:, ny//2, :].T), origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='plasma')
        ax5.set_title(f"渗透率 (log10, y={y[ny//2]:.1f}m)", fontsize=16)
        ax5.set_xlabel("X (m)", fontsize=14)
        ax5.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im5, ax=ax5)
        
        # 绘制中间y切片的浓度
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(C[:, ny//2, :].T, origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='Blues')
        ax6.set_title(f"浓度 (y={y[ny//2]:.1f}m)", fontsize=16)
        ax6.set_xlabel("X (m)", fontsize=14)
        ax6.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im6, ax=ax6)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{results_dir}/results_t{t_val}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

# 可视化结果
print("开始生成可视化结果...")
visualize_results(model, params)
print("可视化结果生成完成!")

# 生成中心点时间序列图
def plot_center_point_evolution(model, params, num_points=100):
    """绘制中心点随时间的演化"""
    print("生成中心点时间序列演化图...")
    center_x = params.Lx / 2
    center_y = params.Ly / 2
    center_z = params.Lz / 2
    
    # 创建时间序列
    times = np.linspace(0, params.T_max, num_points)
    
    # 为每个时间点创建中心点坐标
    center_points = np.column_stack([
        np.full(num_points, center_x),
        np.full(num_points, center_y),
        np.full(num_points, center_z),
        times
    ])
    
    # 转换为张量
    center_points_tensor = torch.tensor(center_points, dtype=torch.float32, device=device)
    
    # 使用模型预测
    with torch.no_grad():
        predictions = model(center_points_tensor)
    
    # 将预测结果转换回numpy数组
    p = predictions[:, 0].cpu().numpy()
    T = predictions[:, 1].cpu().numpy()
    phi = predictions[:, 2].cpu().numpy()
    k = predictions[:, 3].cpu().numpy()
    C = predictions[:, 4].cpu().numpy()
    
    # 计算原始矿物和二次矿物含量（简化假设）
    original_mineral = 1.0 - phi
    secondary_mineral = C * phi
    
    # 绘制时间序列图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个子图: 孔隙率、原始矿物、二次矿物和渗透率
    ax1.plot(times, phi, 'b-', label='孔隙率')
    ax1.plot(times, original_mineral, 'g-', label='原始矿物')
    ax1.plot(times, secondary_mineral, 'r-', label='二次矿物')
    ax1.set_xlabel('时间 (年)', fontsize=14)
    ax1.set_ylabel('体积分数', fontsize=14)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # 渗透率用第二个y轴显示
    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, np.log10(k), 'k--', label='log10(渗透率)')
    ax1_twin.set_ylabel('log10(渗透率)', fontsize=14)
    ax1_twin.legend(loc='upper right')
    
    # 第二个子图: 温度和压力
    ax2.plot(times, T - 273.15, 'r-', label='温度 (°C)')
    ax2.set_xlabel('时间 (年)', fontsize=14)
    ax2.set_ylabel('温度 (°C)', fontsize=14)
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # 压力用第二个y轴显示
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, p / 1e5, 'b--', label='压力 (bar)')
    ax2_twin.set_ylabel('压力 (bar)', fontsize=14)
    ax2_twin.legend(loc='upper right')
    
    plt.suptitle(f'区域中心点 ({center_x}m, {center_y}m, {center_z}m) 随时间变化', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{results_dir}/center_point_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()

# 生成中心点时间序列图
plot_center_point_evolution(model, params)

# 生成特定深度的空间分布图
def plot_depth_profiles(model, params, depths=[10, 30, 50, 70, 90], time=50):
    """在特定时间点绘制不同深度的空间分布"""
    print(f"生成时间 {time} 年不同深度的空间分布图...")
    
    # 创建网格点进行预测
    nx, ny = 50, 50
    x = np.linspace(0, params.Lx, nx)
    y = np.linspace(0, params.Ly, ny)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig, axes = plt.subplots(len(depths), 3, figsize=(18, 5*len(depths)))
    
    for i, depth in enumerate(depths):
        # 创建坐标点
        points = np.column_stack([
            X.flatten(), 
            Y.flatten(), 
            np.full_like(X.flatten(), depth),
            np.full_like(X.flatten(), time)
        ])
        
        # 转换为张量
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        # 使用模型预测
        with torch.no_grad():
            predictions = model(points_tensor)
        
        # 将预测结果转换回numpy数组
        phi = predictions[:, 2].cpu().numpy().reshape(X.shape)
        k = predictions[:, 3].cpu().numpy().reshape(X.shape)
        C = predictions[:, 4].cpu().numpy().reshape(X.shape)
        
        # 绘制孔隙率
        im1 = axes[i, 0].imshow(phi.T, origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                              cmap='viridis', vmin=0.01, vmax=0.3)
        axes[i, 0].set_title(f"深度 {depth}m - 孔隙率", fontsize=14)
        axes[i, 0].set_xlabel("X (m)", fontsize=12)
        axes[i, 0].set_ylabel("Y (m)", fontsize=12)
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 绘制渗透率
        im2 = axes[i, 1].imshow(np.log10(k.T), origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                              cmap='plasma')
        axes[i, 1].set_title(f"深度 {depth}m - 渗透率 (log10)", fontsize=14)
        axes[i, 1].set_xlabel("X (m)", fontsize=12)
        axes[i, 1].set_ylabel("Y (m)", fontsize=12)
        plt.colorbar(im2, ax=axes[i, 1])
        
        # 绘制浓度
        im3 = axes[i, 2].imshow(C.T, origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                              cmap='Blues')
        axes[i, 2].set_title(f"深度 {depth}m - 浓度", fontsize=14)
        axes[i, 2].set_xlabel("X (m)", fontsize=12)
        axes[i, 2].set_ylabel("Y (m)", fontsize=12)
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.suptitle(f"时间 {time} 年不同深度的空间分布", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{results_dir}/depth_profiles_t{time}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 绘制不同深度的空间分布
plot_depth_profiles(model, params)

# 分析损失函数历史
def plot_loss_history(loss_history):
    """绘制损失函数随时间的变化"""
    print("生成损失函数历史图...")
    
    # 提取损失分量
    epochs = range(len(loss_history))
    total_loss = [loss['total'] for loss in loss_history]
    pde_loss = [loss['pde'] for loss in loss_history]
    ic_loss = [loss['ic'] for loss in loss_history]
    bc_loss = [loss['bc'] for loss in loss_history]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制总损失
    ax1.semilogy(epochs, total_loss, 'k-', label='总损失')
    ax1.set_xlabel('训练轮次', fontsize=14)
    ax1.set_ylabel('损失 (对数尺度)', fontsize=14)
    ax1.set_title('总损失随训练轮次的变化', fontsize=16)
    ax1.grid(True)
    ax1.legend()
    
    # 绘制损失分量
    ax2.semilogy(epochs, pde_loss, 'r-', label='PDE损失')
    ax2.semilogy(epochs, ic_loss, 'g-', label='初始条件损失')
    ax2.semilogy(epochs, bc_loss, 'b-', label='边界条件损失')
    ax2.set_xlabel('训练轮次', fontsize=14)
    ax2.set_ylabel('损失 (对数尺度)', fontsize=14)
    ax2.set_title('损失分量随训练轮次的变化', fontsize=16)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/loss_history.png", dpi=300, bbox_inches='tight')
    plt.close()

# 绘制损失函数历史
plot_loss_history(loss_history)

print(f"所有结果已保存到: {results_dir}")
print("模拟完成!")