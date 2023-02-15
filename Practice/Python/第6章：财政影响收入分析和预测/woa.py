import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random

class WOA():
    """改进的鲸鱼算法"""
    # 初始化
    def __init__(self, x_train, y_train, func, lower_bound=np.array([1e-10]),\
                 upper_bound=np.array([1e10]), dim=1, b=1, whale_num=20, max_iter=500):
        self.x_train = x_train
        self.y_train = y_train
        self.func = func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.whale_num = whale_num
        self.max_iter = max_iter
        # 常数参数
        self.b = b
        # 初始化鲸鱼位置
        self.parm = np.random.uniform(0, 1, (whale_num, dim)) * (upper_bound - lower_bound) + lower_bound
        self.gBest_score = np.inf
        self.gBest_parm = np.zeros(dim)
        self.gBest_curve = np.zeros(max_iter)
        
    def fitFunc(self, parm):
        """适应度函数"""
        return self.func(parm, self.x_train, self.y_train)
    
    def rand_whale(self):
        """随机选择鲸鱼"""
        while True:
            rand_index = random.randrange(0, self.whale_num)
            if self.parm[rand_index, :].all() != self.gBest_parm.all() or self.gBest_parm.mean != 0:
                break;
        return rand_index

    def optimize(self):
        """优化函数"""
        t = 0
        while t < self.max_iter:
            for i in range(self.whale_num):
                # 矫正边界
                self.parm[i, :] = np.clip(self.parm[i, :], self.lower_bound, self.upper_bound)
                fitness = self.fitFunc(self.parm[i, :])
                # 更新 gBest_score and gBest_X
                if fitness < self.gBest_score:
                    self.gBest_score = fitness
                    self.gBest_parm = self.parm[i, :].copy()
            
            # 优化前 a = 2 * (self.max_iter - t)/self.max_iter
            # 余弦变化非线性控制因子
            a = 2  *  np.cos(np.pi  *  t / (2  *  self.max_iter))
            # 更新所有鲸鱼位置
            for i in range(self.whale_num):
                p = np.random.uniform()
                R1 = np.random.uniform()
                R2 = np.random.uniform()
                l = np.random.uniform()
                # 参数向量
                A = 2 * a * R1 - a
                C = 2 * R2
                # 自适应权重因子w对数优化
                w = 1 - np.log(1 + (np.e - 1)  *  t / self.max_iter)
                if p >= 0.5:
                    # 螺旋泡泡攻击
                    D = abs(C * self.gBest_parm - self.parm[i, :])
                    self.parm[i, :] = w  *  D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.gBest_parm
                else:
                    #搜索目标猎物
                    if abs(A) < 1:
                        # 以最优鲸鱼为目标，局部搜索
                        D = abs(self.gBest_parm - self.parm[i, :])
                        self.parm[i, :] = self.gBest_parm - w * A * D
                    else:
                        # 以随机鲸鱼为目标，全局搜索
                        rand_index = self.rand_whale()
                        parm_rand = self.parm[rand_index, :]
                        D = abs(C * parm_rand - self.parm[i, :])
                        self.parm[i, :] = parm_rand - w * A * D
                # 随机差分变异优化 避免局部最优
                rand_index = self.rand_whale()
                parm_rand = self.parm[rand_index, :]
                self.parm[i, :] = p * (self.gBest_parm - self.parm[i, :]) + p * (parm_rand - self.parm[i, :])
            # 记录每个函数值（最小型）
            self.gBest_curve[t] = self.gBest_score       
            if (t % 100 == 0):
                print('At iteration: ' + str(t))  
            t += 1
        return self.gBest_parm, self.gBest_curve