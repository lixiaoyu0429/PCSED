from func import load_nk, get_Transmittance
import numpy as np
from numpy import pi, inf
import matplotlib.pyplot as plt

# 波长列表：400-750nm，步长1nm
lambda_list = np.arange(400,751,1)

# 定义空气
air = np.ones_like(lambda_list,dtype=np.complex)

# 载入材料nk值
sio2 = load_nk('SiO2new.csv',lambda_list,'nm')
tio2 = load_nk('TiO2new.csv',lambda_list,'nm')

# 载入玻璃nk值
glass = load_nk('bolijingyuan.csv',lambda_list,'nm')
 
# 定义层厚
d_list = [inf, 65, 115, 65, 115, 65, 115, 65, 400, 65, 115, 65, 115, 65, 115, 65, inf]

print(f'总层厚：{sum(d_list[1:-1])}nm')

# 计算透过率
T_list = get_Transmittance(d_list,lambda_list,air,tio2,sio2,glass)

# 画图
plt.plot(lambda_list,T_list)
plt.show()
