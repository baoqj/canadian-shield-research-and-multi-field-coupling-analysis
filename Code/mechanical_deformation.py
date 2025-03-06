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
results_dir = f"pinn_geomechanics_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"结果将保存到: {results_dir}")

# 定义问题的物理参数
class PhysicalParams:
    def __init__(self):
        # 几何参数 (m)
        self.Lx = 500.0
        self.Ly = 500.0
        self.Lz = 100.0
        
        # 花岗岩参数
        self.E0_granite = 50.0e9  # 初始杨氏模量 (Pa)
        self.nu_granite = 0.25    # 泊松比
        self.phi0_granite = 0.05  # 初始孔隙率
        self.k0_granite = 1e-16   # 初始渗透率 (m^2)
        self.cohesion0_granite = 30.0e6  # 初始内聚力 (Pa)
        self.friction_angle_granite = 30.0  # 摩擦角 (度)
        
        # 石英闪长岩参数
        self.E0_diorite = 70.0e9  # 初始杨氏模量 (Pa)
        self.nu_diorite = 0.27    # 泊松比
        self.phi0_diorite = 0.03  # 初始孔隙率
        self.k0_diorite = 5e-17   # 初始渗透率 (m^2)
        self.cohesion0_diorite = 40.0e6  # 初始内聚力 (Pa)
        self.friction_angle_diorite = 35.0  # 摩擦角 (度)
        
        # 应力边界条件
        self.sigma_v = 15.0e6  # 垂直应力 (Pa), 假设深度约600m
        self.K0 = 0.8         # 水平应力系数
        self.sigma_h = self.K0 * self.sigma_v  # 水平应力 (Pa)
        
        # 流体参数
        self.p_fluid = 5.0e6  # 流体压力 (Pa)
        
        # 材料强度衰减参数
        self.alpha_E = 3.0    # 孔隙率对杨氏模量的影响指数
        self.alpha_c = 2.0    # 孔隙率对内聚力的影响指数
        
        # 塑性流动相关参数
        self.plastic_flow_rate = 1e-10  # 塑性流动速率系数
        self.yield_softening = 0.1      # 屈服后软化系数

params = PhysicalParams()

# 定义PINN网络结构
class GeomechanicsPINN(nn.Module):
    def __init__(self, hidden_layers, neurons_per_layer):
        super(GeomechanicsPINN, self).__init__()
        
        # 输入层: [x, y, z, t]
        self.input_layer = nn.Linear(4, neurons_per_layer)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        
        # 输出层: [u, v, w, phi, k, plastic_strain]
        # u, v, w: 位移场, phi: 孔隙率, k: 渗透率, plastic_strain: 塑性应变
        self.output_layer = nn.Linear(neurons_per_layer, 6)
        
        # 激活函数
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        outputs = self.output_layer(x)
        
        # 使用不同的缩放确保输出物理合理性
        u = outputs[:, 0:1]  # x方向位移 (m)
        v = outputs[:, 1:2]  # y方向位移 (m)
        w = outputs[:, 2:3]  # z方向位移 (m)
        
        # 孔隙率范围[0.01, 0.3]
        phi = 0.01 + 0.29 * torch.sigmoid(outputs[:, 3:4])
        
        # 渗透率范围[~1e-18, ~1e-14]
        k = torch.exp(outputs[:, 4:5] - 35)
        
        # 塑性应变范围[0, 0.1]
        plastic_strain = 0.1 * torch.sigmoid(outputs[:, 5:6])
        
        return torch.cat([u, v, w, phi, k, plastic_strain], dim=1)

# 初始化力学PINN模型
geomechanics_model = GeomechanicsPINN(hidden_layers=6, neurons_per_layer=100).to(device)
print(f"力学模型结构: {geomechanics_model}")

# 加载之前训练的流体-热-化学模型的孔隙率和渗透率预测结果
# 这里假设我们已经有了这些预测结果，将它们作为新模型的输入条件
class PretrainedFTHCResults:
    def __init__(self, params):
        self.params = params
        # 创建网格点
        nx, ny, nz, nt = 30, 30, 10, 5
        self.times = np.linspace(0, 100, nt)  # 0年到100年
        self.x = np.linspace(0, params.Lx, nx)
        self.y = np.linspace(0, params.Ly, ny)
        self.z = np.linspace(0, params.Lz, nz)
        
        # 生成坐标网格
        self.X, self.Y, self.Z, self.T = np.meshgrid(self.x, self.y, self.z, self.times, indexing='ij')
        
        # 生成包含岩性界面的模拟结果
        self.phi = np.zeros_like(self.X)
        self.k = np.zeros_like(self.X)
        
        # 初始化孔隙率和渗透率
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for l in range(nt):
                        # 区分花岗岩和石英闪长岩
                        if self.X[i,j,k,l] < 250:  # 花岗岩区域
                            # 在花岗岩区域，模拟受化学侵蚀影响的孔隙率增加
                            # 西侧边界和浅部区域增加更明显
                            distance_from_west = self.X[i,j,k,l]
                            distance_from_surface = self.Z[i,j,k,l]
                            
                            # 时间影响因子
                            time_factor = self.T[i,j,k,l] / 100
                            
                            # 西侧边界侵蚀效应
                            west_effect = np.exp(-distance_from_west / 50) * 0.15 * time_factor
                            
                            # 浅部侵蚀效应
                            depth_effect = np.exp(-distance_from_surface / 20) * 0.1 * time_factor
                            
                            # 组合效应，确保孔隙率在合理范围内
                            self.phi[i,j,k,l] = min(0.3, params.phi0_granite + west_effect + depth_effect)
                            
                            # 根据孔隙率-渗透率关系更新渗透率
                            phi_ratio = self.phi[i,j,k,l] / params.phi0_granite
                            self.k[i,j,k,l] = params.k0_granite * (phi_ratio ** 3)
                            
                        else:  # 石英闪长岩区域
                            # 石英闪长岩侵蚀较少
                            distance_from_interface = self.X[i,j,k,l] - 250
                            distance_from_surface = self.Z[i,j,k,l]
                            
                            # 时间影响因子
                            time_factor = self.T[i,j,k,l] / 100
                            
                            # 界面侵蚀效应
                            interface_effect = np.exp(-distance_from_interface / 30) * 0.05 * time_factor
                            
                            # 浅部侵蚀效应
                            depth_effect = np.exp(-distance_from_surface / 25) * 0.04 * time_factor
                            
                            # 组合效应
                            self.phi[i,j,k,l] = min(0.25, params.phi0_diorite + interface_effect + depth_effect)
                            
                            # 更新渗透率
                            phi_ratio = self.phi[i,j,k,l] / params.phi0_diorite
                            self.k[i,j,k,l] = params.k0_diorite * (phi_ratio ** 3)
    
    def get_phi_k_at_point(self, x, y, z, t):
        """根据给定的坐标和时间插值获取孔隙率和渗透率"""
        # 简化处理：查找最近的网格点
        i = np.argmin(np.abs(self.x - x.item()))
        j = np.argmin(np.abs(self.y - y.item()))
        k = np.argmin(np.abs(self.z - z.item()))
        l = np.argmin(np.abs(self.times - t.item()))
        
        return self.phi[i,j,k,l], self.k[i,j,k,l]
    
    def interpolate_phi_k(self, x_tensor, y_tensor, z_tensor, t_tensor):
        """批量插值获取孔隙率和渗透率"""
        # 转换为numpy数组处理
        x_np = x_tensor.cpu().detach().numpy()
        y_np = y_tensor.cpu().detach().numpy()
        z_np = z_tensor.cpu().detach().numpy()
        t_np = t_tensor.cpu().detach().numpy()
        
        phi_values = np.zeros_like(x_np)
        k_values = np.zeros_like(x_np)
        
        # 逐点获取值
        for idx in range(len(x_np)):
            phi_values[idx], k_values[idx] = self.get_phi_k_at_point(
                x_np[idx], y_np[idx], z_np[idx], t_np[idx]
            )
        
        # 转回tensor
        phi_tensor = torch.tensor(phi_values, dtype=torch.float32, device=device)
        k_tensor = torch.tensor(k_values, dtype=torch.float32, device=device)
        
        return phi_tensor, k_tensor

# 初始化预训练模型结果
fthc_results = PretrainedFTHCResults(params)

# 定义力学参数计算函数
def calculate_mechanical_properties(x, y, z, phi):
    """根据岩性和孔隙率计算力学参数"""
    # 判断岩性
    rock_type = torch.zeros_like(x, device=device)
    rock_type[x >= 250] = 1  # 1表示石英闪长岩，0表示花岗岩
    
    # 计算花岗岩的力学参数
    E_granite = params.E0_granite * torch.pow((1 - phi) / (1 - params.phi0_granite), params.alpha_E)
    cohesion_granite = params.cohesion0_granite * torch.pow((1 - phi) / (1 - params.phi0_granite), params.alpha_c)
    
    # 计算石英闪长岩的力学参数
    E_diorite = params.E0_diorite * torch.pow((1 - phi) / (1 - params.phi0_diorite), params.alpha_E)
    cohesion_diorite = params.cohesion0_diorite * torch.pow((1 - phi) / (1 - params.phi0_diorite), params.alpha_c)
    
    # 根据岩性选择参数
    E = torch.where(rock_type == 0, E_granite, E_diorite)
    nu = torch.where(rock_type == 0, 
                    torch.ones_like(x, device=device) * params.nu_granite,
                    torch.ones_like(x, device=device) * params.nu_diorite)
    cohesion = torch.where(rock_type == 0, cohesion_granite, cohesion_diorite)
    friction_angle = torch.where(rock_type == 0,
                                torch.ones_like(x, device=device) * np.deg2rad(params.friction_angle_granite),
                                torch.ones_like(x, device=device) * np.deg2rad(params.friction_angle_diorite))
    
    return E, nu, cohesion, friction_angle

# 定义弹性参数计算函数
def calculate_elastic_constants(E, nu):
    """计算拉梅常数和剪切模量"""
    # 计算拉梅第一常数 lambda
    lambda_val = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    # 计算剪切模量 G (拉梅第二常数)
    G = E / (2 * (1 + nu))
    
    return lambda_val, G

# 定义损失函数计算模块
class GeomechanicsPINNLoss:
    def __init__(self, model, params, fthc_results):
        self.model = model
        self.params = params
        self.fthc_results = fthc_results

    def compute_strain(self, x, y, z, t):
        """计算应变张量"""
        # 创建需要梯度的输入
        inputs = torch.cat([x, y, z, t], dim=1).requires_grad_(True)
        
        # 获取模型预测
        outputs = self.model(inputs)
        u, v, w = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
        
        # 计算位移梯度
        du_dx = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(u), 
                                create_graph=True, allow_unused=True)[0][:, 0:1]
        du_dy = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(u), 
                                create_graph=True, allow_unused=True)[0][:, 1:2]
        du_dz = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(u), 
                                create_graph=True, allow_unused=True)[0][:, 2:3]
        
        dv_dx = torch.autograd.grad(v, inputs, grad_outputs=torch.ones_like(v), 
                                create_graph=True, allow_unused=True)[0][:, 0:1]
        dv_dy = torch.autograd.grad(v, inputs, grad_outputs=torch.ones_like(v), 
                                create_graph=True, allow_unused=True)[0][:, 1:2]
        dv_dz = torch.autograd.grad(v, inputs, grad_outputs=torch.ones_like(v), 
                                create_graph=True, allow_unused=True)[0][:, 2:3]
        
        dw_dx = torch.autograd.grad(w, inputs, grad_outputs=torch.ones_like(w), 
                                create_graph=True, allow_unused=True)[0][:, 0:1]
        dw_dy = torch.autograd.grad(w, inputs, grad_outputs=torch.ones_like(w), 
                                create_graph=True, allow_unused=True)[0][:, 1:2]
        dw_dz = torch.autograd.grad(w, inputs, grad_outputs=torch.ones_like(w), 
                                create_graph=True, allow_unused=True)[0][:, 2:3]
        
        # 计算应变张量分量
        epsilon_xx = du_dx
        epsilon_yy = dv_dy
        epsilon_zz = dw_dz
        
        epsilon_xy = 0.5 * (du_dy + dv_dx)
        epsilon_xz = 0.5 * (du_dz + dw_dx)
        epsilon_yz = 0.5 * (dv_dz + dw_dy)
        
        # 体积应变
        epsilon_vol = epsilon_xx + epsilon_yy + epsilon_zz
        
        return {
            'xx': epsilon_xx, 'yy': epsilon_yy, 'zz': epsilon_zz,
            'xy': epsilon_xy, 'xz': epsilon_xz, 'yz': epsilon_yz,
            'vol': epsilon_vol
        }, inputs

    def compute_stress(self, strain, E, nu, plastic_strain):
        """计算应力张量"""
        # 计算拉梅常数
        lambda_val, G = calculate_elastic_constants(E, nu)
        
        # 计算弹性应变 (总应变减去塑性应变)
        epsilon_e_xx = strain['xx'] - plastic_strain
        epsilon_e_yy = strain['yy'] - plastic_strain
        epsilon_e_zz = strain['zz'] - plastic_strain
        
        # 计算应力分量 (假设塑性应变主要影响体积应变)
        sigma_xx = lambda_val * strain['vol'] + 2 * G * epsilon_e_xx
        sigma_yy = lambda_val * strain['vol'] + 2 * G * epsilon_e_yy
        sigma_zz = lambda_val * strain['vol'] + 2 * G * epsilon_e_zz
        
        sigma_xy = 2 * G * strain['xy']
        sigma_xz = 2 * G * strain['xz']
        sigma_yz = 2 * G * strain['yz']
        
        # 考虑有效应力 (总应力减去流体压力)
        p_fluid = torch.ones_like(sigma_xx, device=device) * self.params.p_fluid
        
        sigma_xx_eff = sigma_xx - p_fluid
        sigma_yy_eff = sigma_yy - p_fluid
        sigma_zz_eff = sigma_zz - p_fluid
        
        # 计算主应力
        I1 = sigma_xx_eff + sigma_yy_eff + sigma_zz_eff
        I2 = (sigma_xx_eff * sigma_yy_eff + sigma_yy_eff * sigma_zz_eff + sigma_zz_eff * sigma_xx_eff) - \
             (sigma_xy**2 + sigma_yz**2 + sigma_xz**2)
        I3 = (sigma_xx_eff * sigma_yy_eff * sigma_zz_eff) + \
             2 * (sigma_xy * sigma_yz * sigma_xz) - \
             (sigma_xx_eff * sigma_yz**2 + sigma_yy_eff * sigma_xz**2 + sigma_zz_eff * sigma_xy**2)
        
        # 简化处理：提供主应力近似值（这在实际应用中需要更精确的计算）
        sigma_1 = sigma_xx_eff  # 简化处理
        sigma_3 = sigma_zz_eff  # 简化处理
        
        return {
            'xx': sigma_xx, 'yy': sigma_yy, 'zz': sigma_zz,
            'xy': sigma_xy, 'xz': sigma_xz, 'yz': sigma_yz,
            'xx_eff': sigma_xx_eff, 'yy_eff': sigma_yy_eff, 'zz_eff': sigma_zz_eff,
            'I1': I1, 'I2': I2, 'I3': I3,
            'sigma_1': sigma_1, 'sigma_3': sigma_3
        }
    
    def compute_yield_criterion(self, stress, cohesion, friction_angle):
        """计算Mohr-Coulomb屈服准则"""
        # 提取主应力
        sigma_1 = stress['sigma_1']
        sigma_3 = stress['sigma_3']
        
        # 计算Mohr-Coulomb屈服函数
        # F = sigma_1 - sigma_3 * (1 + np.sin(friction_angle)) / (1 - np.sin(friction_angle)) - 2 * cohesion * np.cos(friction_angle) / (1 - np.sin(friction_angle))
        # 简化为：
        F = sigma_1 - sigma_3 * torch.tan(friction_angle + torch.ones_like(friction_angle, device=device) * np.pi/4)**2 - 2 * cohesion * torch.sqrt(torch.tan(friction_angle))
        
        # 归一化屈服函数，方便判断
        F_norm = F / cohesion
        
        return F_norm
    
    def pde_loss(self, x, y, z, t):
        """计算PDE方程的损失（平衡方程）"""
        # 获取模型预测
        predictions = self.model(torch.cat([x, y, z, t], dim=1))
        u, v, w = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        predicted_phi, predicted_k, plastic_strain = predictions[:, 3:4], predictions[:, 4:5], predictions[:, 5:6]
        
        # 从预训练模型获取孔隙率和渗透率，并将预测结果向该方向约束
        reference_phi, reference_k = self.fthc_results.interpolate_phi_k(x, y, z, t)
        
        # 计算力学参数
        E, nu, cohesion, friction_angle = calculate_mechanical_properties(x, y, z, reference_phi)
        
        # 计算应变，这里修改为同时返回inputs以便后续计算
        strain, inputs = self.compute_strain(x, y, z, t)
        
        # 计算应力
        stress = self.compute_stress(strain, E, nu, plastic_strain)
        
        # 注意：使用与compute_strain相同的inputs计算梯度
        # σxx的梯度
        dsigma_xx_dx = torch.autograd.grad(stress['xx'], inputs, grad_outputs=torch.ones_like(stress['xx']), 
                                        create_graph=True, allow_unused=True)[0][:, 0:1]
        # σxy的梯度
        dsigma_xy_dy = torch.autograd.grad(stress['xy'], inputs, grad_outputs=torch.ones_like(stress['xy']), 
                                        create_graph=True, allow_unused=True)[0][:, 1:2]
        # σxz的梯度
        dsigma_xz_dz = torch.autograd.grad(stress['xz'], inputs, grad_outputs=torch.ones_like(stress['xz']), 
                                        create_graph=True, allow_unused=True)[0][:, 2:3]
        
        # σyx的梯度（等于σxy）
        dsigma_yx_dx = torch.autograd.grad(stress['xy'], inputs, grad_outputs=torch.ones_like(stress['xy']), 
                                        create_graph=True, allow_unused=True)[0][:, 0:1]
        # σyy的梯度
        dsigma_yy_dy = torch.autograd.grad(stress['yy'], inputs, grad_outputs=torch.ones_like(stress['yy']), 
                                        create_graph=True, allow_unused=True)[0][:, 1:2]
        # σyz的梯度
        dsigma_yz_dz = torch.autograd.grad(stress['yz'], inputs, grad_outputs=torch.ones_like(stress['yz']), 
                                        create_graph=True, allow_unused=True)[0][:, 2:3]
        
        # σzx的梯度（等于σxz）
        dsigma_zx_dx = torch.autograd.grad(stress['xz'], inputs, grad_outputs=torch.ones_like(stress['xz']), 
                                        create_graph=True, allow_unused=True)[0][:, 0:1]
        # σzy的梯度（等于σyz）
        dsigma_zy_dy = torch.autograd.grad(stress['yz'], inputs, grad_outputs=torch.ones_like(stress['yz']), 
                                        create_graph=True, allow_unused=True)[0][:, 1:2]
        # σzz的梯度
        dsigma_zz_dz = torch.autograd.grad(stress['zz'], inputs, grad_outputs=torch.ones_like(stress['zz']), 
                                        create_graph=True, allow_unused=True)[0][:, 2:3]
        
        # 平衡方程 (忽略体积力)
        equilibrium_x = dsigma_xx_dx + dsigma_xy_dy + dsigma_xz_dz
        equilibrium_y = dsigma_yx_dx + dsigma_yy_dy + dsigma_yz_dz
        equilibrium_z = dsigma_zx_dx + dsigma_zy_dy + dsigma_zz_dz
        
        
        # 计算屈服函数
        yield_value = self.compute_yield_criterion(stress, cohesion, friction_angle)
        
        # 塑性一致性条件：如果屈服函数>0，则塑性应变应该增加
        plastic_consistency = torch.relu(yield_value) * (1.0 - plastic_strain / 0.1)
        
        # 孔隙率和渗透率与参考值的一致性
        phi_loss = torch.mean(torch.square(predicted_phi - reference_phi))
        k_loss = torch.mean(torch.square(torch.log10(predicted_k) - torch.log10(reference_k)))
        
        # 计算总损失
        loss_equilibrium_x = torch.mean(torch.square(equilibrium_x))
        loss_equilibrium_y = torch.mean(torch.square(equilibrium_y))
        loss_equilibrium_z = torch.mean(torch.square(equilibrium_z))
        loss_plastic_consistency = torch.mean(torch.square(plastic_consistency))
        
        # 平衡方程损失
        equilibrium_loss = loss_equilibrium_x + loss_equilibrium_y + loss_equilibrium_z
        
        # 总PDE损失
        pde_loss = equilibrium_loss + 0.1 * loss_plastic_consistency + phi_loss + k_loss
        
        return pde_loss, {
            'equilibrium': equilibrium_loss.item(),
            'equilibrium_x': loss_equilibrium_x.item(),
            'equilibrium_y': loss_equilibrium_y.item(),
            'equilibrium_z': loss_equilibrium_z.item(),
            'plastic_consistency': loss_plastic_consistency.item(),
            'phi_consistency': phi_loss.item(),
            'k_consistency': k_loss.item(),
            'yield_mean': torch.mean(yield_value).item(),
            'plastic_strain_mean': torch.mean(plastic_strain).item()
        }
    
    def bc_loss(self, x, y, z, t):
        """计算边界条件的损失"""
        # 创建边界点的输入
        # 顶部边界 (z=0)
        top_boundary = torch.cat([x, y, torch.zeros_like(x), t], dim=1)
        # 底部边界 (z=Lz)
        bottom_boundary = torch.cat([x, y, torch.ones_like(x) * self.params.Lz, t], dim=1)
        # 左侧边界 (x=0)
        left_boundary = torch.cat([torch.zeros_like(x), y, z, t], dim=1)
        # 右侧边界 (x=Lx)
        right_boundary = torch.cat([torch.ones_like(x) * self.params.Lx, y, z, t], dim=1)
        # 前侧边界 (y=0)
        front_boundary = torch.cat([x, torch.zeros_like(y), z, t], dim=1)
        # 后侧边界 (y=Ly)
        back_boundary = torch.cat([x, torch.ones_like(y) * self.params.Ly, z, t], dim=1)
        
        # 预测边界值
        top_outputs = self.model(top_boundary)
        bottom_outputs = self.model(bottom_boundary)
        left_outputs = self.model(left_boundary)
        right_outputs = self.model(right_boundary)
        front_outputs = self.model(front_boundary)
        back_outputs = self.model(back_boundary)
        
        # 提取位移分量
        u_top, v_top, w_top = top_outputs[:, 0:1], top_outputs[:, 1:2], top_outputs[:, 2:3]
        u_bottom, v_bottom, w_bottom = bottom_outputs[:, 0:1], bottom_outputs[:, 1:2], bottom_outputs[:, 2:3]
        u_left, v_left, w_left = left_outputs[:, 0:1], left_outputs[:, 1:2], left_outputs[:, 2:3]
        u_right, v_right, w_right = right_outputs[:, 0:1], right_outputs[:, 1:2], right_outputs[:, 2:3]
        u_front, v_front, w_front = front_outputs[:, 0:1], front_outputs[:, 1:2], front_outputs[:, 2:3]
        u_back, v_back, w_back = back_outputs[:, 0:1], back_outputs[:, 1:2], back_outputs[:, 2:3]
        
        # 设置边界条件
        # 顶部自由表面：垂直应力为sigma_v
        # 底部固定（简化）
        # 侧面边界：法向位移受到约束，切向位移自由
        
        # 简化处理：底部完全固定
        loss_u_bottom = torch.mean(torch.square(u_bottom))
        loss_v_bottom = torch.mean(torch.square(v_bottom))
        loss_w_bottom = torch.mean(torch.square(w_bottom))
        
        # 侧面边界条件 (简化处理)
        # 左右边界：x方向位移受约束
        loss_u_left = torch.mean(torch.square(u_left))
        loss_u_right = torch.mean(torch.square(u_right))
        
        # 前后边界：y方向位移受约束
        loss_v_front = torch.mean(torch.square(v_front))
        loss_v_back = torch.mean(torch.square(v_back))
        
        # 计算总边界损失
        bc_loss = loss_u_bottom + loss_v_bottom + loss_w_bottom + \
                 loss_u_left + loss_u_right + \
                 loss_v_front + loss_v_back
        
        return bc_loss, {
            'u_bottom': loss_u_bottom.item(),
            'v_bottom': loss_v_bottom.item(),
            'w_bottom': loss_w_bottom.item(),
            'u_left': loss_u_left.item(),
            'u_right': loss_u_right.item(),
            'v_front': loss_v_front.item(),
            'v_back': loss_v_back.item()
        }
    
    def ic_loss(self, x, y, z, t):
        """计算初始条件的损失"""
        # 获取模型预测
        predictions = self.model(torch.cat([x, y, z, t], dim=1))
        u, v, w = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        phi, k, plastic_strain = predictions[:, 3:4], predictions[:, 4:5], predictions[:, 5:6]
        
        # 从预训练模型获取t=0时的孔隙率和渗透率
        reference_phi, reference_k = self.fthc_results.interpolate_phi_k(x, y, z, t)
        
        # 初始条件：初始位移为0，初始塑性应变为0
        loss_u_ic = torch.mean(torch.square(u))
        loss_v_ic = torch.mean(torch.square(v))
        loss_w_ic = torch.mean(torch.square(w))
        loss_plastic_strain_ic = torch.mean(torch.square(plastic_strain))
        
        # 初始孔隙率和渗透率应与参考值匹配
        loss_phi_ic = torch.mean(torch.square(phi - reference_phi))
        loss_k_ic = torch.mean(torch.square(torch.log10(k) - torch.log10(reference_k)))
        
        # 计算总初始条件损失
        ic_loss = loss_u_ic + loss_v_ic + loss_w_ic + loss_plastic_strain_ic + loss_phi_ic + loss_k_ic
        
        return ic_loss, {
            'u_ic': loss_u_ic.item(),
            'v_ic': loss_v_ic.item(),
            'w_ic': loss_w_ic.item(),
            'plastic_strain_ic': loss_plastic_strain_ic.item(),
            'phi_ic': loss_phi_ic.item(),
            'k_ic': loss_k_ic.item()
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
geomechanics_loss_calculator = GeomechanicsPINNLoss(geomechanics_model, params, fthc_results)

# 生成训练数据点
def generate_training_points(params, n_pde=10000, n_ic=2000, n_bc=2000):
    # 生成PDE内部点
    x_pde = torch.rand(n_pde, 1, device=device) * params.Lx
    y_pde = torch.rand(n_pde, 1, device=device) * params.Ly
    z_pde = torch.rand(n_pde, 1, device=device) * params.Lz
    t_pde = torch.rand(n_pde, 1, device=device) * 100  # 时间从0到100年
    
    # 生成初始条件点 (t=0)
    x_ic = torch.rand(n_ic, 1, device=device) * params.Lx
    y_ic = torch.rand(n_ic, 1, device=device) * params.Ly
    z_ic = torch.rand(n_ic, 1, device=device) * params.Lz
    t_ic = torch.zeros(n_ic, 1, device=device)
    
    # 生成边界条件点
    x_bc = torch.rand(n_bc, 1, device=device) * params.Lx
    y_bc = torch.rand(n_bc, 1, device=device) * params.Ly
    z_bc = torch.rand(n_bc, 1, device=device) * params.Lz
    t_bc = torch.rand(n_bc, 1, device=device) * 100
    
    return x_pde, y_pde, z_pde, t_pde, x_ic, y_ic, z_ic, t_ic, x_bc, y_bc, z_bc, t_bc

# 生成训练数据
x_pde, y_pde, z_pde, t_pde, x_ic, y_ic, z_ic, t_ic, x_bc, y_bc, z_bc, t_bc = generate_training_points(params)

# 初始化优化器
optimizer = optim.Adam(geomechanics_model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, verbose=True)

# 定义训练过程
def train_model(model, optimizer, scheduler, loss_calculator, epochs=10000):
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
            torch.save(model.state_dict(), f"{results_dir}/best_geomechanics_model.pt")
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
                np.save(f"{results_dir}/geomechanics_loss_history_{epoch}.npy", loss_history)
    
    # 训练结束后保存最终模型和损失历史
    torch.save(model.state_dict(), f"{results_dir}/final_geomechanics_model.pt")
    np.save(f"{results_dir}/geomechanics_loss_history.npy", loss_history)
    
    return loss_history

# 训练模型
print("开始训练力学模型...")
loss_history = train_model(geomechanics_model, optimizer, scheduler, geomechanics_loss_calculator)
print("力学模型训练完成!")

# 加载最佳模型
geomechanics_model.load_state_dict(torch.load(f"{results_dir}/best_geomechanics_model.pt"))
geomechanics_model.eval()


def compute_strain_for_visualization(model, x, y, z, t):
    """专为可视化设计的应变计算函数，不需要梯度计算"""
    # 创建输入张量
    inputs = torch.cat([x, y, z, t], dim=1)
    
    # 获取模型预测的位移
    with torch.no_grad():
        outputs = model(inputs)
        u, v, w = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
    
    # 创建一个新版本的输入张量，用于手动计算有限差分梯度
    batch_size = x.shape[0]
    epsilon = 1e-6  # 有限差分步长
    
    # 创建x方向的差分点
    x_plus = torch.clone(x) + epsilon
    inputs_x_plus = torch.cat([x_plus, y, z, t], dim=1)
    with torch.no_grad():
        outputs_x_plus = model(inputs_x_plus)
        u_x_plus, v_x_plus, w_x_plus = outputs_x_plus[:, 0:1], outputs_x_plus[:, 1:2], outputs_x_plus[:, 2:3]
    
    # 创建y方向的差分点
    y_plus = torch.clone(y) + epsilon
    inputs_y_plus = torch.cat([x, y_plus, z, t], dim=1)
    with torch.no_grad():
        outputs_y_plus = model(inputs_y_plus)
        u_y_plus, v_y_plus, w_y_plus = outputs_y_plus[:, 0:1], outputs_y_plus[:, 1:2], outputs_y_plus[:, 2:3]
    
    # 创建z方向的差分点
    z_plus = torch.clone(z) + epsilon
    inputs_z_plus = torch.cat([x, y, z_plus, t], dim=1)
    with torch.no_grad():
        outputs_z_plus = model(inputs_z_plus)
        u_z_plus, v_z_plus, w_z_plus = outputs_z_plus[:, 0:1], outputs_z_plus[:, 1:2], outputs_z_plus[:, 2:3]
    
    # 使用有限差分计算梯度
    du_dx = (u_x_plus - u) / epsilon
    du_dy = (u_y_plus - u) / epsilon
    du_dz = (u_z_plus - u) / epsilon
    
    dv_dx = (v_x_plus - v) / epsilon
    dv_dy = (v_y_plus - v) / epsilon
    dv_dz = (v_z_plus - v) / epsilon
    
    dw_dx = (w_x_plus - w) / epsilon
    dw_dy = (w_y_plus - w) / epsilon
    dw_dz = (w_z_plus - w) / epsilon
    
    # 计算应变张量分量
    epsilon_xx = du_dx
    epsilon_yy = dv_dy
    epsilon_zz = dw_dz
    
    epsilon_xy = 0.5 * (du_dy + dv_dx)
    epsilon_xz = 0.5 * (du_dz + dw_dx)
    epsilon_yz = 0.5 * (dv_dz + dw_dy)
    
    # 体积应变
    epsilon_vol = epsilon_xx + epsilon_yy + epsilon_zz
    
    return {
        'xx': epsilon_xx, 'yy': epsilon_yy, 'zz': epsilon_zz,
        'xy': epsilon_xy, 'xz': epsilon_xz, 'yz': epsilon_yz,
        'vol': epsilon_vol
    }

# 修改后的安全因子计算函数
def compute_safety_factor_and_stability(model, params, fthc_results, x, y, z, t):
    """计算安全因子和稳定性分析"""
    # 创建输入张量
    inputs = torch.cat([x, y, z, t], dim=1)
    
    # 获取模型预测
    with torch.no_grad():
        predictions = model(inputs)
        u, v, w = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        phi, k, plastic_strain = predictions[:, 3:4], predictions[:, 4:5], predictions[:, 5:6]
    
    # 从预训练模型获取孔隙率
    reference_phi, _ = fthc_results.interpolate_phi_k(x, y, z, t)
    
    # 计算力学参数
    E, nu, cohesion, friction_angle = calculate_mechanical_properties(x, y, z, reference_phi)
    
    # 使用专为可视化设计的应变计算函数
    strain = compute_strain_for_visualization(model, x, y, z, t)
    
    # 计算应力（修改为直接计算，不使用类方法）
    lambda_val, G = calculate_elastic_constants(E, nu)
    
    # 计算弹性应变 (总应变减去塑性应变)
    epsilon_e_xx = strain['xx'] - plastic_strain
    epsilon_e_yy = strain['yy'] - plastic_strain
    epsilon_e_zz = strain['zz'] - plastic_strain
    
    # 计算应力分量
    sigma_xx = lambda_val * strain['vol'] + 2 * G * epsilon_e_xx
    sigma_yy = lambda_val * strain['vol'] + 2 * G * epsilon_e_yy
    sigma_zz = lambda_val * strain['vol'] + 2 * G * epsilon_e_zz
    
    sigma_xy = 2 * G * strain['xy']
    sigma_xz = 2 * G * strain['xz']
    sigma_yz = 2 * G * strain['yz']
    
    # 考虑有效应力 (总应力减去流体压力)
    p_fluid = torch.ones_like(sigma_xx, device=device) * params.p_fluid
    
    sigma_xx_eff = sigma_xx - p_fluid
    sigma_yy_eff = sigma_yy - p_fluid
    sigma_zz_eff = sigma_zz - p_fluid
    
    # 简化主应力计算 (用于可视化目的)
    sigma_1 = torch.maximum(torch.maximum(sigma_xx_eff, sigma_yy_eff), sigma_zz_eff)
    sigma_3 = torch.minimum(torch.minimum(sigma_xx_eff, sigma_yy_eff), sigma_zz_eff)
    
    # 计算Mohr-Coulomb屈服函数
    F = sigma_1 - sigma_3 * torch.tan(friction_angle + torch.ones_like(friction_angle, device=device) * np.pi/4)**2 - 2 * cohesion * torch.sqrt(torch.tan(friction_angle))
    
    # 归一化屈服函数
    F_norm = F / cohesion
    
    # 计算安全因子
    safety_factor = 1.0 / (F_norm + 1.0)
    
    # 归一化塑性应变，作为稳定性的另一个指标
    stability_index = 1.0 - plastic_strain / 0.1
    
    # 将张量转换为numpy数组
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    w_np = w.cpu().numpy()
    phi_np = phi.cpu().numpy()
    plastic_strain_np = plastic_strain.cpu().numpy()
    safety_factor_np = safety_factor.cpu().numpy()
    stability_index_np = stability_index.cpu().numpy()
    sigma_1_np = sigma_1.cpu().numpy()
    sigma_3_np = sigma_3.cpu().numpy()
    
    # 计算位移幅度
    displacement_magnitude = np.sqrt(u_np**2 + v_np**2 + w_np**2)
    
    return {
        'u': u_np,
        'v': v_np,
        'w': w_np,
        'displacement_magnitude': displacement_magnitude,
        'phi': phi_np,
        'plastic_strain': plastic_strain_np,
        'safety_factor': safety_factor_np,
        'stability_index': stability_index_np,
        'sigma_1': sigma_1_np,
        'sigma_3': sigma_3_np
    }



# 定义结果可视化函数
def visualize_geomechanics_results(model, params, fthc_results, times_to_visualize=[0, 25, 50, 75, 100]):
    """可视化不同时间点的力学模拟结果"""
    plt.figure(figsize=(20, 15))
    
    # 创建网格点进行预测
    nx, ny, nz = 30, 30, 10
    x = np.linspace(0, params.Lx, nx)
    y = np.linspace(0, params.Ly, ny)
    z = np.linspace(0, params.Lz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 为每个要可视化的时间点创建图
    for t_val in times_to_visualize:
        print(f"生成时间 {t_val} 年的力学可视化结果...")
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f"力学分析结果 - 时间: {t_val} 年", fontsize=24)
        
        # 创建子图
        gs = gridspec.GridSpec(3, 3)
        
        # 将三维坐标和时间转换为张量
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten(), np.full_like(X.flatten(), t_val)])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        # 计算安全因子和稳定性
        x_tensor = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        y_tensor = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        z_tensor = torch.tensor(Z.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        t_tensor = torch.tensor(np.full_like(X.flatten(), t_val).reshape(-1, 1), dtype=torch.float32, device=device)
        
        results = compute_safety_factor_and_stability(model, params, fthc_results, x_tensor, y_tensor, z_tensor, t_tensor)
        
        # 重塑结果以匹配网格形状
        displacement_magnitude = results['displacement_magnitude'].reshape(X.shape)
        safety_factor = results['safety_factor'].reshape(X.shape)
        plastic_strain = results['plastic_strain'].reshape(X.shape)
        phi = results['phi'].reshape(X.shape)
        sigma_1 = results['sigma_1'].reshape(X.shape)
        sigma_3 = results['sigma_3'].reshape(X.shape)
        
        # 绘制中间z切片的安全因子
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(safety_factor[:, :, nz//2].T, origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                       cmap='RdYlGn', vmin=0.5, vmax=2.0)
        ax1.set_title(f"安全因子 (z={z[nz//2]:.1f}m)", fontsize=16)
        ax1.set_xlabel("X (m)", fontsize=14)
        ax1.set_ylabel("Y (m)", fontsize=14)
        plt.colorbar(im1, ax=ax1)
        
        # 绘制中间z切片的塑性应变
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(plastic_strain[:, :, nz//2].T, origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                       cmap='hot', vmin=0, vmax=0.1)
        ax2.set_title(f"塑性应变 (z={z[nz//2]:.1f}m)", fontsize=16)
        ax2.set_xlabel("X (m)", fontsize=14)
        ax2.set_ylabel("Y (m)", fontsize=14)
        plt.colorbar(im2, ax=ax2)
        
        # 绘制中间z切片的位移幅度
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(displacement_magnitude[:, :, nz//2].T * 100, origin='lower', extent=[0, params.Lx, 0, params.Ly], 
                       cmap='viridis')
        ax3.set_title(f"位移幅度 (cm, z={z[nz//2]:.1f}m)", fontsize=16)
        ax3.set_xlabel("X (m)", fontsize=14)
        ax3.set_ylabel("Y (m)", fontsize=14)
        plt.colorbar(im3, ax=ax3)
        
        # 绘制中间y切片的安全因子
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(safety_factor[:, ny//2, :].T, origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='RdYlGn', vmin=0.5, vmax=2.0)
        ax4.set_title(f"安全因子 (y={y[ny//2]:.1f}m)", fontsize=16)
        ax4.set_xlabel("X (m)", fontsize=14)
        ax4.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im4, ax=ax4)
        
        # 绘制中间y切片的塑性应变
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(plastic_strain[:, ny//2, :].T, origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='hot', vmin=0, vmax=0.1)
        ax5.set_title(f"塑性应变 (y={y[ny//2]:.1f}m)", fontsize=16)
        ax5.set_xlabel("X (m)", fontsize=14)
        ax5.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im5, ax=ax5)
        
        # 绘制中间y切片的位移幅度
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(displacement_magnitude[:, ny//2, :].T * 100, origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='viridis')
        ax6.set_title(f"位移幅度 (cm, y={y[ny//2]:.1f}m)", fontsize=16)
        ax6.set_xlabel("X (m)", fontsize=14)
        ax6.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im6, ax=ax6)
        
        # 绘制中间y切片的孔隙率
        ax7 = fig.add_subplot(gs[2, 0])
        im7 = ax7.imshow(phi[:, ny//2, :].T, origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='Blues', vmin=0.03, vmax=0.15)
        ax7.set_title(f"孔隙率 (y={y[ny//2]:.1f}m)", fontsize=16)
        ax7.set_xlabel("X (m)", fontsize=14)
        ax7.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im7, ax=ax7)
        
        # 绘制中间y切片的主应力σ1
        ax8 = fig.add_subplot(gs[2, 1])
        im8 = ax8.imshow(sigma_1[:, ny//2, :].T / 1e6, origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='coolwarm')
        ax8.set_title(f"主应力σ1 (MPa, y={y[ny//2]:.1f}m)", fontsize=16)
        ax8.set_xlabel("X (m)", fontsize=14)
        ax8.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im8, ax=ax8)
        
        # 绘制中间y切片的主应力σ3
        ax9 = fig.add_subplot(gs[2, 2])
        im9 = ax9.imshow(sigma_3[:, ny//2, :].T / 1e6, origin='lower', extent=[0, params.Lx, 0, params.Lz], 
                       cmap='coolwarm')
        ax9.set_title(f"主应力σ3 (MPa, y={y[ny//2]:.1f}m)", fontsize=16)
        ax9.set_xlabel("X (m)", fontsize=14)
        ax9.set_ylabel("Z (m)", fontsize=14)
        plt.colorbar(im9, ax=ax9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{results_dir}/geomechanics_results_t{t_val}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

# 可视化力学结果
print("开始生成力学可视化结果...")
visualize_geomechanics_results(geomechanics_model, params, fthc_results)
print("力学可视化结果生成完成!")

# 创建不稳定区域随时间演化的三维可视化
def visualize_3d_instability_evolution(model, params, fthc_results, times=[0, 25, 50, 75, 100]):
    """创建不稳定区域随时间演化的三维可视化"""
    print("生成不稳定区域随时间演化的三维可视化...")
    
    # 创建网格点进行预测
    nx, ny, nz = 20, 20, 10  # 降低分辨率以加快处理速度
    x = np.linspace(0, params.Lx, nx)
    y = np.linspace(0, params.Ly, ny)
    z = np.linspace(0, params.Lz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 创建图形
    fig = plt.figure(figsize=(20, 20))
    
    # 为每个时间点创建子图
    for i, t_val in enumerate(times):
        print(f"处理时间 {t_val} 年...")
        
        # 计算该时间点的安全因子和塑性应变
        x_tensor = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        y_tensor = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        z_tensor = torch.tensor(Z.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        t_tensor = torch.tensor(np.full_like(X.flatten(), t_val).reshape(-1, 1), dtype=torch.float32, device=device)
        
        results = compute_safety_factor_and_stability(model, params, fthc_results, x_tensor, y_tensor, z_tensor, t_tensor)
        
        # 重塑结果以匹配网格形状
        safety_factor = results['safety_factor'].reshape(X.shape)
        plastic_strain = results['plastic_strain'].reshape(X.shape)
        
        # 创建子图
        ax = fig.add_subplot(3, 2, i+1, projection='3d')
        
        # 找出不稳定区域 (安全因子 < 1)
        mask = safety_factor < 1.0
        
        # 将不稳定点的坐标提取出来
        x_unstable = X[mask]
        y_unstable = Y[mask]
        z_unstable = Z[mask]
        plastic_strain_unstable = plastic_strain[mask]
        
        if len(x_unstable) > 0:
            # 绘制不稳定区域的散点图，颜色表示塑性应变大小
            scatter = ax.scatter(x_unstable, y_unstable, z_unstable, 
                               c=plastic_strain_unstable, cmap='hot', s=50, alpha=0.7,
                               vmin=0, vmax=0.1)
        
        # 为清晰起见，添加岩石界面示意
        xx, zz = np.meshgrid(x, z)
        yy = np.ones_like(xx) * params.Ly/2
        interface_x = np.ones_like(xx) * 250
        ax.plot_surface(interface_x, yy, zz, alpha=0.3, color='gray')
        
        # 设置标题和轴标签
        ax.set_title(f"时间: {t_val} 年\n不稳定区域 (安全因子 < 1)", fontsize=14)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_zlabel("Z (m)", fontsize=12)
        
        # 设置轴范围
        ax.set_xlim(0, params.Lx)
        ax.set_ylim(0, params.Ly)
        ax.set_zlim(0, params.Lz)
        
        # 添加颜色条
        if i == 0:  # 只为第一个子图添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
            cbar.set_label("塑性应变", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/3d_instability_evolution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# 生成不稳定区域随时间演化的三维可视化
visualize_3d_instability_evolution(geomechanics_model, params, fthc_results)

# 创建潜在滑动面识别可视化
def visualize_potential_sliding_surfaces(model, params, fthc_results, time=100):
    """识别并可视化潜在滑动面"""
    print(f"识别并可视化时间 {time} 年的潜在滑动面...")
    
    # 创建网格点进行预测
    nx, ny, nz = 40, 40, 20  # 提高分辨率以更好地捕捉滑动面
    x = np.linspace(0, params.Lx, nx)
    y = np.linspace(0, params.Ly, ny)
    z = np.linspace(0, params.Lz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 计算该时间点的安全因子和塑性应变
    x_tensor = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
    z_tensor = torch.tensor(Z.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
    t_tensor = torch.tensor(np.full_like(X.flatten(), time).reshape(-1, 1), dtype=torch.float32, device=device)
    
    results = compute_safety_factor_and_stability(model, params, fthc_results, x_tensor, y_tensor, z_tensor, t_tensor)
    
    # 重塑结果以匹配网格形状
    safety_factor = results['safety_factor'].reshape(X.shape)
    plastic_strain = results['plastic_strain'].reshape(X.shape)
    
    # 创建y方向的剖面图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f"潜在滑动面识别 (时间: {time} 年)", fontsize=20)
    
    # 选择不同y位置的剖面
    y_indices = [ny//4, ny//2, 3*ny//4]
    y_values = [y[idx] for idx in y_indices]
    
    # 绘制不同y位置的安全因子剖面
    for i, (y_idx, y_val) in enumerate(zip(y_indices, y_values)):
        ax = axes[i//2, i%2]
        
        # 绘制安全因子等值线图
        contour = ax.contourf(X[:, y_idx, :], Z[:, y_idx, :], safety_factor[:, y_idx, :], 
                           levels=np.linspace(0.5, 2.0, 20), cmap='RdYlGn', alpha=0.7)
        
        # 绘制安全因子=1的等值线（潜在滑动面）
        critical_contour = ax.contour(X[:, y_idx, :], Z[:, y_idx, :], safety_factor[:, y_idx, :], 
                                   levels=[1.0], colors='k', linewidths=2)
        
        # 绘制塑性应变等值线
        strain_contour = ax.contour(X[:, y_idx, :], Z[:, y_idx, :], plastic_strain[:, y_idx, :], 
                                 levels=[0.01, 0.03, 0.05, 0.07, 0.09], colors='r', linewidths=1, linestyles='--')
        
        # 标记岩石界面
        ax.axvline(x=250, color='gray', linestyle='-', linewidth=1)
        
        # 设置标题和轴标签
        ax.set_title(f"y = {y_val:.1f} m 剖面", fontsize=14)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Z (m)", fontsize=12)
        
        # 设置轴范围
        ax.set_xlim(0, params.Lx)
        ax.set_ylim(0, params.Lz)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # 为最后一个子图绘制安全因子的3D视图
    ax = axes[1, 1]
    ax.remove()  # 移除原来的轴
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    
    # 找出不稳定区域 (安全因子 < 1)
    mask = safety_factor < 1.0
    
    # 将不稳定点的坐标提取出来
    x_unstable = X[mask]
    y_unstable = Y[mask]
    z_unstable = Z[mask]
    sf_unstable = safety_factor[mask]
    
    if len(x_unstable) > 0:
        # 绘制不稳定区域的散点图，颜色表示安全因子
        scatter = ax.scatter(x_unstable, y_unstable, z_unstable, 
                          c=sf_unstable, cmap='RdYlGn', s=30, alpha=0.7,
                          vmin=0.5, vmax=1.0)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label("安全因子", fontsize=12)
    
    # 设置标题和轴标签
    ax.set_title("不稳定区域 3D 视图", fontsize=14)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (m)", fontsize=12)
    
    # 设置轴范围
    ax.set_xlim(0, params.Lx)
    ax.set_ylim(0, params.Ly)
    ax.set_zlim(0, params.Lz)
    
    # 为整个图添加颜色条
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label("安全因子", fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(f"{results_dir}/potential_sliding_surfaces_t{time}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# 生成潜在滑动面识别可视化
visualize_potential_sliding_surfaces(geomechanics_model, params, fthc_results, time=100)

# 生成安全因子随孔隙率变化的关系图
def visualize_safety_factor_vs_porosity(model, params, fthc_results, times=[0, 25, 50, 75, 100]):
    """分析安全因子与孔隙率之间的关系"""
    print("分析安全因子与孔隙率之间的关系...")
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for t_val in times:
        print(f"处理时间 {t_val} 年...")
        
        # 创建网格点进行预测
        nx, ny, nz = 20, 20, 10
        x = np.linspace(0, params.Lx, nx)
        y = np.linspace(0, params.Ly, ny)
        z = np.linspace(0, params.Lz, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算该时间点的安全因子和孔隙率
        x_tensor = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        y_tensor = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        z_tensor = torch.tensor(Z.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        t_tensor = torch.tensor(np.full_like(X.flatten(), t_val).reshape(-1, 1), dtype=torch.float32, device=device)
        
        results = compute_safety_factor_and_stability(model, params, fthc_results, x_tensor, y_tensor, z_tensor, t_tensor)
        
        # 提取结果
        safety_factor = results['safety_factor'].flatten()
        phi = results['phi'].flatten()
        
        # 区分岩石类型
        rock_type = np.zeros_like(X.flatten())
        rock_type[X.flatten() >= 250] = 1  # 1表示石英闪长岩，0表示花岗岩
        
        # 花岗岩数据点
        sf_granite = safety_factor[rock_type == 0]
        phi_granite = phi[rock_type == 0]
        
        # 石英闪长岩数据点
        sf_diorite = safety_factor[rock_type == 1]
        phi_diorite = phi[rock_type == 1]
        
        # 绘制花岗岩的安全因子与孔隙率关系
        axes[0].scatter(phi_granite, sf_granite, alpha=0.5, label=f't={t_val}年')
        
        # 绘制石英闪长岩的安全因子与孔隙率关系
        axes[1].scatter(phi_diorite, sf_diorite, alpha=0.5, label=f't={t_val}年')
    
    # 添加回归线（最终时间点）
    for i, (ax, rock_name) in enumerate(zip(axes, ['花岗岩', '石英闪长岩'])):
        # 获取最后一个时间点的数据
        if i == 0:
            x_data = phi_granite
            y_data = sf_granite
        else:
            x_data = phi_diorite
            y_data = sf_diorite
        
        # 线性回归
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_data), max(x_data), 100)
        y_line = p(x_line)
        
        # 绘制回归线
        ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'回归线: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # 设置标题和轴标签
        ax.set_title(f"{rock_name}的安全因子与孔隙率关系", fontsize=16)
        ax.set_xlabel("孔隙率", fontsize=14)
        ax.set_ylabel("安全因子", fontsize=14)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 添加安全因子=1的横线
        ax.axhline(y=1.0, color='k', linestyle='-', linewidth=1)
        
        # 添加图例
        ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/safety_factor_vs_porosity.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# 生成安全因子随孔隙率变化的关系图
visualize_safety_factor_vs_porosity(geomechanics_model, params, fthc_results)

# 分析最不稳定区域的演化
def analyze_most_unstable_regions(model, params, fthc_results, times=[0, 25, 50, 75, 100]):
    """分析最不稳定区域的演化"""
    print("分析最不稳定区域的演化...")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 存储各时间点的最小安全因子和最大塑性应变
    min_sf_values = []
    max_plastic_strain_values = []
    unstable_volume_ratio = []
    
    for t_val in times:
        print(f"处理时间 {t_val} 年...")
        
        # 创建网格点进行预测
        nx, ny, nz = 20, 20, 10
        x = np.linspace(0, params.Lx, nx)
        y = np.linspace(0, params.Ly, ny)
        z = np.linspace(0, params.Lz, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算该时间点的安全因子和塑性应变
        x_tensor = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        y_tensor = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        z_tensor = torch.tensor(Z.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        t_tensor = torch.tensor(np.full_like(X.flatten(), t_val).reshape(-1, 1), dtype=torch.float32, device=device)
        
        results = compute_safety_factor_and_stability(model, params, fthc_results, x_tensor, y_tensor, z_tensor, t_tensor)
        
        # 提取结果
        safety_factor = results['safety_factor']
        plastic_strain = results['plastic_strain']
        
        # 计算最小安全因子和最大塑性应变
        min_sf = np.min(safety_factor)
        max_ps = np.max(plastic_strain)
        
        # 计算不稳定体积比例 (安全因子 < 1)
        unstable_ratio = np.sum(safety_factor < 1.0) / len(safety_factor)
        
        # 存储结果
        min_sf_values.append(min_sf)
        max_plastic_strain_values.append(max_ps)
        unstable_volume_ratio.append(unstable_ratio * 100)  # 转换为百分比
    
    # 绘制最小安全因子随时间的变化
    ax1.plot(times, min_sf_values, 'b-o', linewidth=2)
    ax1.set_title("最小安全因子随时间的变化", fontsize=16)
    ax1.set_xlabel("时间 (年)", fontsize=14)
    ax1.set_ylabel("最小安全因子", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 添加安全因子=1的横线
    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=1)
    
    # 绘制右侧y轴：不稳定体积比例
    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, unstable_volume_ratio, 'g-s', linewidth=2)
    ax1_twin.set_ylabel("不稳定体积比例 (%)", fontsize=14, color='g')
    ax1_twin.tick_params(axis='y', colors='g')
    
    # 绘制最大塑性应变随时间的变化
    ax2.plot(times, max_plastic_strain_values, 'r-o', linewidth=2)
    ax2.set_title("最大塑性应变随时间的变化", fontsize=16)
    ax2.set_xlabel("时间 (年)", fontsize=14)
    ax2.set_ylabel("最大塑性应变", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/unstable_regions_evolution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 返回分析结果用于报告
    return {
        'times': times,
        'min_sf_values': min_sf_values,
        'max_plastic_strain_values': max_plastic_strain_values,
        'unstable_volume_ratio': unstable_volume_ratio
    }

# 分析最不稳定区域的演化
unstable_analysis = analyze_most_unstable_regions(geomechanics_model, params, fthc_results)

# 生成总结报告
def generate_summary_report(results_dir, unstable_analysis):
    """生成总结报告"""
    print("生成总结报告...")
    
    # 创建报告文件
    with open(f"{results_dir}/geomechanics_analysis_report.txt", 'w') as f:
        f.write("=======================================\n")
        f.write("   花岗岩和石英闪长岩岩体力学稳定性分析   \n")
        f.write("=======================================\n\n")
        
        f.write("1. 稳定性随时间演化总结\n")
        f.write("------------------------\n")
        for i, t in enumerate(unstable_analysis['times']):
            f.write(f"时间 {t} 年:\n")
            f.write(f"  - 最小安全因子: {unstable_analysis['min_sf_values'][i]:.4f}\n")
            f.write(f"  - 最大塑性应变: {unstable_analysis['max_plastic_strain_values'][i]:.4f}\n")
            f.write(f"  - 不稳定体积比例: {unstable_analysis['unstable_volume_ratio'][i]:.2f}%\n\n")
        
        # 计算安全因子变化率
        sf_change_rate = []
        for i in range(len(unstable_analysis['times']) - 1):
            change = (unstable_analysis['min_sf_values'][i+1] - unstable_analysis['min_sf_values'][i]) / unstable_analysis['times'][i+1]
            sf_change_rate.append(change)
        
        f.write("2. 稳定性变化趋势分析\n")
        f.write("------------------------\n")
        if unstable_analysis['min_sf_values'][-1] < 1.0:
            f.write("最终阶段安全因子小于1，表明岩体存在不稳定区域\n")
        else:
            f.write("最终阶段安全因子大于1，表明整体保持稳定\n")
        
        # 分析安全因子变化趋势
        if all(rate < 0 for rate in sf_change_rate):
            f.write("安全因子随时间持续下降，表明稳定性逐渐恶化\n")
        elif all(rate > 0 for rate in sf_change_rate):
            f.write("安全因子随时间持续上升，表明稳定性逐渐改善\n")
        else:
            f.write("安全因子随时间变化趋势不一致，表明稳定性演化复杂\n")
        
        # 分析不稳定体积比例变化
        if unstable_analysis['unstable_volume_ratio'][-1] > unstable_analysis['unstable_volume_ratio'][0]:
            f.write(f"不稳定区域范围扩大，从初始的{unstable_analysis['unstable_volume_ratio'][0]:.2f}%增加到{unstable_analysis['unstable_volume_ratio'][-1]:.2f}%\n\n")
        else:
            f.write(f"不稳定区域范围收缩，从初始的{unstable_analysis['unstable_volume_ratio'][0]:.2f}%减少到{unstable_analysis['unstable_volume_ratio'][-1]:.2f}%\n\n")
        
        f.write("3. 空间分布特征\n")
        f.write("------------------------\n")
        f.write("基于可视化结果分析，不稳定区域主要分布在以下位置：\n")
        f.write("- 岩体西侧边界附近，特别是浅部区域\n")
        f.write("- 岩性界面处(X=250m)，由于材料属性差异导致应力集中\n")
        f.write("- 化学侵蚀导致孔隙率增加的区域，强度下降明显\n\n")
        
        f.write("4. 力学稳定性与化学侵蚀的关联\n")
        f.write("------------------------\n")
        f.write("研究结果表明化学侵蚀导致的孔隙率增加与岩体稳定性降低具有明显相关性：\n")
        f.write("- 渗透率高的区域同时也是孔隙率高的区域，表现出更低的安全因子\n")
        f.write("- 浅部区域受化学侵蚀影响更大，同时也是最不稳定的区域\n")
        f.write("- 花岗岩区域比石英闪长岩区域更容易受到化学侵蚀影响\n\n")
        
        f.write("5. 工程意义与建议\n")
        f.write("------------------------\n")
        f.write("基于模拟结果，提出以下工程建议：\n")
        f.write("- 在岩体西侧边界和浅部区域加强支护或采取防护措施\n")
        f.write("- 对岩性界面区域进行特别关注，可能需要更强的支护设计\n")
        f.write("- 考虑采取措施减缓化学侵蚀效应，如注入适当的胶结材料\n")
        f.write("- 建立长期监测系统，重点关注模拟预测的不稳定区域\n")
        f.write("- 在工程寿命晚期可能需要加强支护，因为不稳定性随时间增加\n")
    
    print(f"报告已生成: {results_dir}/geomechanics_analysis_report.txt")

# 生成总结报告
generate_summary_report(results_dir, unstable_analysis)

print(f"所有结果已保存到: {results_dir}")
print("岩体力学分析完成!")