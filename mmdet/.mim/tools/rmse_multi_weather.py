import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr
import json
import numpy as np
from scipy import stats
import os
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import argparse

def main():
    # argparse 설정
    parser = argparse.ArgumentParser(description="Calculating RMSE of AutoEval")
    parser.add_argument("--target", type=str, default="coco" , help="Target data")
    parser.add_argument("--orig_index", type=int, default=1, help="Index of orginal data")
    parser.add_argument("--expert", type=str, default="coco" , help="Domain of expert")


    # argument 파싱
    args = parser.parse_args()
    print('Target', args.target)
    sources = ["night", "overcast", "snow", "wet"]
    
    
    print(sources)
    dropout_pos = "1_2"
    dropou_rate ='0_15'
    X = []
    y = []
    metesize = 38
    
    experts = ["original"]  # ["night", "overcast", "snow", "wet"] 해당 값을 하나씩 넣어주시면 됩니다.
    for expert in experts:
        for source in sources:
            for meta in range(metesize):
                with open('./res_multiweather_train_60_model/' + str(expert) + '_expert/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '/' + str(meta+1) + '.json') as f:
                    data = json.load(f) 
                    X.append(data['0'][0][0]*100)  # mAP
                    # y.append(-data['0'][2][0]) # BoS
                    y.append(data['0'][3][0]) # Ours

   
    esti = []
    true = []
    for expert in experts:
        for source in sources:
            with open('./res_multiweather_test_full_model/' + str(expert) + '_expert/cost_droprate_' + dropou_rate + '_' + str(source) + '_batch_1_droppos_' + dropout_pos + '_s60_n1/1.json') as f:
                data = json.load(f) 
                true.append(data['0'][0][0]*100)  # mAP
                # esti.append(-data['0'][2][0]) # BoS
                esti.append(data['0'][3][0]) # Ours

    
    BS  = np.array(y)
    mAP = np.array(X)
    BS = BS.reshape(-1, 1)
   
    model = LinearRegression()
    model.fit(BS, mAP)

    omega1 = model.coef_[0]
    omega0 = model.intercept_
    print(f"기울기 (ω1): {omega1}")
    print(f"절편 (ω0): {omega0}")

    esti = np.array(esti)
    true = np.array(true)
    esti = esti.reshape(-1, 1)
    print("true:  ", true)
    
    # 예측값 계산 (학습된 모델을 사용하여 mAP 예측)
    mAP_pred = model.predict(esti)
    print("mAP_pred:  ", mAP_pred)
    
    # Root Mean Squared Error (RMSE) 계산
    rmse = np.sqrt(np.mean((true - mAP_pred) ** 2))
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    with open("./rmse_values/output.txt", "a") as file:
        file.write(f"{args.target}, {mAP_pred[0]}, {true}\n")   

if __name__ == "__main__":
    main()