from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import matplotlib.pyplot as plt


import tensorflow as tf
from PIL import Image
if int(tf.__version__.split('.')[0]) >= 2:
    from tensorflow import keras
else:

    import keras
    
import numpy as np
import pandas as pd
from tqdm import tqdm


#データの入力と整理

data = pd.read_csv("usdjpy_5min.csv")
data["<DTYYYYMMDD>"] =data["<DTYYYYMMDD>"].apply(lambda x:str(x))
data["<TIME>"] =data["<TIME>"].apply(lambda x:str(x).zfill(4))
pd.to_datetime(data["<DTYYYYMMDD>"].str[:4]+"-"+data["<DTYYYYMMDD>"].str[4:6]+"-"+data["<DTYYYYMMDD>"].str[6:]+" "+data["<TIME>"].str[:2]+":"+data["<TIME>"].str[2:]+":00" 
)
data["time"] = pd.to_datetime(data["<DTYYYYMMDD>"].str[:4]+"-"+data["<DTYYYYMMDD>"].str[4:6]+"-"+data["<DTYYYYMMDD>"].str[6:]+" "+data["<TIME>"].str[:2]+":"+data["<TIME>"].str[2:]+":00" 
)
data = data.set_index("time")
data = data[["<OPEN>","<HIGH>","<LOW>","<CLOSE>"]]
data.columns =  ["open","high","low","close"]

#特徴量整理
for i in [15,60,240,1140]:
    data[str(i)+"-open"] = data["open"].shift(i-1)
    data[str(i)+"-high"] = data["high"].rolling(i).max()
    data[str(i)+"-low"] = data["high"].rolling(i).min()


data  = data.dropna()
df = pd.DataFrame(range(261772), columns=['A'],
                  index=pd.date_range('2015-01-02', periods=261772 ,freq='15min'))
df =  df.reset_index()
df = df[["index"]]
data = data.reset_index().rename(columns={"time":"index"})
data  = pd.merge(df,data,on = "index")
data =  data.dropna()
data =  data.loc[:,"close":]
close =  data["close"].copy()
data_min  =  data["1140-low"].copy()

data_max  =  data["1140-high"].copy()
for column in data.columns:
    data[column]  = (data[column]-data_min)/(data_max-data_min)
    
    
data["action"] =0 #なにもしない、買う、売る
data["having_flag"] = 0# ポジションの有無
data["close_value"]  = close
data["next_close_value"]  = data["close_value"].shift(-1)


#後の改良用　一次元時系列データを二次元に直す gaf encoding
def gaf(x):
    x = np.array(x)
    x = (x-np.min(x))/(np.max(x)-np.min(x))*2-1
    length = len(x)
    mat = np.array([[np.cos(np.arccos(x[i])+np.arccos(x[j])) for i in range(length)] for j in range(length)])
    return mat



from tensorflow.keras import layers as kl
import random

def inverse_rescaling(x, eps=0.001):
    n = np.sqrt(1.0 + 4.0 * eps * (np.abs(x) + 1.0 + eps)) - 1.0
    n = n / (2.0 * eps)
    return np.sign(x) * ((n**2) - 1.0)


def rescaling(x, eps=0.001):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0) + eps * x


class DQN:
    def __init__(self):
        
        

        self.BATCH_SIZE = 20
        self.r = 0.98
        self.ACTION_NUM = 3
        self.train_count =  0
        self.target_model_update_interval = 0
        self.sync_count =  0
        
        self.INIT_EPS = 5e-1
        self.STEP_EPS =  1e-5
        self.EPS_STEP = 0
        self.LIMIT_EPS =  1e-5
        self.TEST_EPS = 1e-5
        self.MULTI_STEP = 5
        self.TRAINING_FLAG = 1
        self.loss = keras.losses.Huber()
        self.train_step =  0
        
        input_shape = (15)
        in_state = c = kl.Input(shape=input_shape)

        
        # 隠れ層
        c = kl.Dense(64, activation="relu")(c)
        
        # 出力層、config.nb_actionsはアクション数
        c = kl.Dense(self.ACTION_NUM,activation="linear", kernel_initializer="truncated_normal",bias_initializer="truncated_normal")(c)
        self.target_model = keras.Model(inputs =  in_state, outputs = c)
        self.model = keras.Model(inputs =  in_state, outputs = c)
        self.model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])
        
        
    
    def _read_data(self):
        self.data =  data
        
        
    def forward(self,state):
        having_flag = np.array(state["having_flag"])[0]
        max_q = None
        
        
        if having_flag==1:
            invalid_actions  = [1] #0なにもしない ,1 買い ,2売り
        elif having_flag == -1:
            invalid_actions  = [2] #0なにもしない ,1 買い ,2売り
        else:
            invalid_actions = []
        #トレーニング状態かどうか
        if self.TRAINING_FLAG:
            epsilon = self.INIT_EPS - self.STEP_EPS * self.EPS_STEP
            self.EPS_STEP +=1 
            if epsilon < self.LIMIT_EPS:
                epsilon = self.LIMIT_EPS
                
        else:
            epsilon =  self.TEST_EPS
        if self.TRAINING_FLAG:
            if random.random()  <  epsilon:
                action =  random.choice([a for a in range(self.ACTION_NUM) if a not in invalid_actions])
            else:
                q =  self.model.predict(state, verbose=0)[0]
                q = [(-np.inf if a in invalid_actions else v) for a, v in enumerate(q)]
                action = np.argmax(q)
            max_q = self.target_model.predict(state, verbose=0)[0][action]
                    
            
            
        else:
            if random.random()  <  epsilon:
                action =  random.choice([a for a in range(self.ACTION_NUM) if a not in invalid_actions])


            else:
                q =  self.model.predict(state, verbose=0)[0]
                q = [(-np.inf if a in invalid_actions else v) for a, v in enumerate(q)]
              
                action =  int(np.argmax(q))

            self.action = action
        return action , max_q, {"eps":epsilon}
    
    def train(self,train_step):
        self.TRAINING_FLAG = 1
        
        self.train_step +=1
        data = self.data.copy().iloc[train_step:train_step+self.MULTI_STEP,:].reset_index(drop = True)
        
        states = data.iloc[:,:15]
        for i in range(len(states)):
            action  , max_q, epsilon=  self.forward(states.iloc[i:i+1,:])
            states.at[i,"action"] = action
            
            if i!= len(states)-1:
                if action == 1 and (states.at[i,"having_flag"]  in [0 , -1] ): 
                    states.at[i+1,"having_flag"] =states.at[i,"having_flag"]+1
                    
                elif action == 2 and (states.at[i,"having_flag"]  in [0 , 1] ):
                    states.at[i+1,"having_flag"] =states.at[i,"having_flag"]-1
                elif action ==0:
                    states.at[i+1,"having_flag"] =states.at[i,"having_flag"]
                
                
                
                      
        states["flag_shift"] =  states["having_flag"].shift(-1)
        
        states.at[len(states)-1,"action"] = action
        
        if action == 1 and (states.at[i,"having_flag"]  in [0 , -1] ): 
            states.at[len(states)-1,"flag_shift"] =states.at[len(states)-1,"having_flag"]+1

        elif action == 2 and (states.at[i,"having_flag"]  in [0 , 1] ):
            states.at[len(states)-1,"flag_shift"] =states.at[len(states)-1,"having_flag"]-1
        elif action ==0:
            states.at[len(states)-1,"flag_shift"] =states.at[len(states)-1,"having_flag"]

        
        
        states["close_value"] = data["close_value"]
        states["c_diff"] = data["next_close_value"] - states["close_value"]
        
        states["reward"] = states["c_diff"]*states["flag_shift"]
        states["r"] =  [self.r**i for i in range (len(states))]
        states["reward"] *=states["r"] 
        
        gain =  states["reward"].sum() +  self.r**len(states)*max_q
        gain = rescaling(gain)
        target_q = gain
        
        action = states.at[0,"action"] 
        q = self.model.predict(data.iloc[:1,:15], verbose=0)
        q[0][action] = gain
        self.model.fit(data.iloc[:1,:15].values,q, verbose=0)
        
        if self.train_step%10 ==0:
            self.target_model.set_weights(self.model.get_weights())
            
    def evaluate(self):
        self.TRAINING_FLAG = 0
        #120000:+120000+672*2
        states = self.data.copy().iloc[120000:120000+672*2,:].reset_index(drop = True)
        states = states.iloc[:,:15]
        for i in range(len(states)):
            action  , max_q, epsilon=  self.forward(states.iloc[i:i+1,:])
            states.at[i,"action"] = action
            
            if i!= len(states)-1:
                if action == 1 and (states.at[i,"having_flag"]  in [0 , -1] ): 
                    states.at[i+1,"having_flag"] =states.at[i,"having_flag"]+1
                    
                elif action == 2 and (states.at[i,"having_flag"]  in [0 , 1] ):
                    states.at[i+1,"having_flag"] =states.at[i,"having_flag"]-1
                elif action ==0:
                    states.at[i+1,"having_flag"] =states.at[i,"having_flag"]
                
                
                
                      
        states["flag_shift"] =  states["having_flag"].shift(-1)
        
        states.at[len(states)-1,"action"] = action
        
        if action == 1 and (states.at[i,"having_flag"]  in [0 , -1] ): 
            states.at[len(states)-1,"flag_shift"] =states.at[len(states)-1,"having_flag"]+1

        elif action == 2 and (states.at[i,"having_flag"]  in [0 , 1] ):
            states.at[len(states)-1,"flag_shift"] =states.at[len(states)-1,"having_flag"]-1
        elif action ==0:
            states.at[len(states)-1,"flag_shift"] =states.at[len(states)-1,"having_flag"]

        
        
        states["close_value"] = data["close_value"]
        states["c_diff"] = data["next_close_value"] - states["close_value"]
        
        states["reward"] = states["c_diff"]*states["flag_shift"]
        states["reward_cumsum"] = states["reward"].cumsum()
        states["reward_cumsum"].plot()
        plt.show()
    
a=DQN()
a._read_data()
a.model.predict(data.iloc[1:2,:15])

for i in tqdm(range(10)):
    for j in range(10):
        test_data =  a.train(j*17)
    a.evaluate()
    
