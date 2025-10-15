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
    # domain = args.target.split("_")[0]
    sources = [f"night_batch_{i}" for i in range(1,2)] + [f"overcast_batch_{i}" for i in range(1,2)] + [f"snow_batch_{i}" for i in range(1,2)] + [f"wet_batch_{i}" for i in range(1,2)]
    sources = ["night", "overcast", "snow", "wet"]
    experts = ["night", "overcast", "snow", "wet"]
    
    print(sources)
    dropout_pos = "1_2"
    dropou_rate ='0_15'
    X = []
    y = []


    # for source in sources:
    #     with open('./res_multiweather_train_full_model2/wet_expert/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s60_n1/' + '1.json') as f:
    #         data = json.load(f) 
    #         X.append(data['0'][0][0]*100)  # mAP
    #         y.append(-data['0'][2][0]) # 수정된 ours 추가해야 함
    #         # y.append(data['0'][3][0]) # 수정된 ours 추가해야 함
    # experts = ["overcast"]
    for expert in experts:
        for source in sources:
            with open('./res_multiweather_train_full_model2/' + str(expert) + '_expert/cost_droprate_' + dropou_rate + '_' + str(source) + '_batch_1_droppos_' + dropout_pos + '_s60_n1/1.json') as f:
            # with open('./res_multiweather_train_full_model2/night_expert/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s60_n1/' + '1.json') as f:
                data = json.load(f) 
                X.append(data['0'][0][0]*100)  # mAP
                y.append(-data['0'][2][0]) # 수정된 ours 추가해야 함
                # y.append(data['0'][8][0]) # 수정된 ours 추가해야 함


    targets = [args.target]
    # print(targets)
    # sources = ["night", "overcast", "snow", "wet"]
    sources = ["wet"]
    esti = []
    true = []
    for expert in experts:
        for source in sources:
            with open('./res_multiweather_test_full_model2/' + str(expert) + '_expert/cost_droprate_' + dropou_rate + '_' + str(source) + '_batch_1_droppos_' + dropout_pos + '_s60_n1/1.json') as f:
            # with open(f'./res_multiweather_test_full_model2/night_expert/cost_droprate_' + dropou_rate + '_' + str(target) + '_droppos_' + dropout_pos + '_s60_n1/' + '1.json') as f:
                data = json.load(f) 
                true.append(data['0'][0][0]*100)
                esti.append(-data['0'][2][0]) # 수정된 ours 추가해야 함
                # esti.append(data['0'][8][0])

    # for target in targets:
    #     with open(f'./res_multiweather_test_full_model2/wet_expert/cost_droprate_' + dropou_rate + '_' + str(target) + '_droppos_' + dropout_pos + '_s60_n1/' + '1.json') as f:
    #         data = json.load(f) 
    #         true.append(data['0'][0][0]*100)
    #         # esti.append(-data['0'][2][0]) # 수정된 ours 추가해야 함
    #         esti.append(data['0'][3][0])
    BS  = np.array(y)
    mAP = np.array(X)
    BS = BS.reshape(-1, 1)
    # print(BS)
    # print(mAP)
    # print(np.std(BS))
    # print(np.std(mAP))
    # exit()
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
    with open("./rmse_values/output_multiweather_overcast_expert_from_train_full_test_full_bos.txt", "a") as file:
        file.write(f"{args.target}, {mAP_pred[0]}, {true}\n")   

if __name__ == "__main__":
    main()