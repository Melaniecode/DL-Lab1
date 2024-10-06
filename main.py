import csv
import numpy as np
import random
import matplotlib.pyplot as plt

# 整理資料需要的 class
class DataLoader:

    def __init__(self, file_path):
        self.file_path = file_path; # 檔案位置
        self.data_list = []; # 存放資料的 list
        
    #整理 Lab1_traindata.csv 檔案的資料
    def load_data(self):
        with open(self.file_path, mode='r') as file:  # 讀取檔案
            # 整理資料
            rows = csv.reader(file); 
            for row in rows:
                x1, x2, x3, y = map(float, row); # str -> float
                self.data_list.append([x1, x2, x3, y]); 
        return np.array(self.data_list); 
      
# 回歸模型
class Regression:
    #初始化
    def __init__(self):
        self.weights = None; 
        self.bias = None; 

    def train(self, x, y, learning_rate=0.00002, max_times=1000, mse_limit=5): #(學習率, 最大次數, mse 的終止條件)
        random.seed(0);   # 設定亂數種子
        self.weights = np.random.randn(3); # 設定 w (權重矩陣)，大小為 (3, )
        self.bias = np.random.randn(1); # 設定 b (偏差向量), 大小為 (1, )

        mse_history = []; 
        
        # 梯度下降法計算
        for time in range(max_times):
            # 計算預測值
            y_pred = np.dot(x, self.weights) + self.bias; 

            # 計算均方誤差 (MSE)
            mse = np.mean(np.square(y_pred-y)); 
            mse_history.append(mse); 

            w = (1 / len(y)) * np.dot(x.T, (y_pred-y));   # 計算權重的梯度
            self.weights -= learning_rate*w;   # 更新權重

            # 更新偏差
            b = (1 / len(y)) * np.sum(y_pred-y);   # 計算偏差的梯度
            self.bias -= learning_rate*b;   # 更新偏差

            # 判斷終止條件
            if mse < mse_limit:
                break; 

        print(f"最終世代數: {time+1}"); 
        print(f"最終 MSE: {mse}"); 
        print(f"weights: {self.weights}"); 
        print(f"bias: {self.bias}"); 

        return mse_history; 


    def predict(self, X):
        return np.dot(X, self.weights)+self.bias; 

# 給 Lab1_traindata.csv 檔案資料
data_loader = DataLoader("Lab1_traindata.csv"); 
data = data_loader.load_data(); 

random.seed(0)  # 設定亂數種子碼
np.random.shuffle(data)  # 將輸入資料打亂
x = data[:, :-1]  # x1, x2, x3
y = data[:, -1]   # y


# 訓練模型
regression = Regression(); 
mse_history = regression.train(x, y); 

# 根據模型進行預測
xa = np.array([85.7, 11.8, 15.9])
xb = np.array([184.9, 27.7, 32.5])
ya = regression.predict(xa)
yb = regression.predict(xb)
print(f"預測值 ya: {ya}")
print(f"預測值 yb: {yb}")


# 印出 output.png
plt.plot(mse_history, label='MSE', color='blue'); 
plt.xlabel('Epochs'); 
plt.ylabel('MSE'); 
plt.legend(loc="upper right"); 
plt.savefig('Output.png'); 
plt.close(); 
