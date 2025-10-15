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
import random

def main():
    parser = argparse.ArgumentParser(description="Calculating RMSE of AutoEval")
    parser.add_argument("--target", type=str, default="coco" , help="Target data")
    parser.add_argument("--orig_index", default=1, help="Index of orginal data")

    args = parser.parse_args()

    print('Target', args.target)

    datasets = ["coco", "bdd", "cityscapes", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco","udacity2coco", "traffic2coco"] # car
    # datasets = ["coco", "caltech", "citypersons", "cityscapes", "crowdhuman", "ECP", "ExDark", "kitti", "self_driving"] # person

    sources = [item for item in datasets if item != args.target]
    print('Source', sources)
    print('i', f'"{args.orig_index}"')
    dropout_pos = "1_2"
    dropou_rate ='0_15'
    
    # meta - for train
    X = []
    y = []
    y2=[]

    metesize = 50
    for source in sources:
        for meta in range(metesize):
            with open('./result/car_inc/PCR/r50_retina/' + str(source) + '_s250_n50/' + str(meta) + '.json') as f:
                data = json.load(f)

                X.append(data[f"{args.orig_index}"][0][0]*100)  # mAP
                y.append(data[f"{args.orig_index}"][1][0])   # consistency
                y2.append(data[f"{args.orig_index}"][2][0]) # reliability

               
    targets = [args.target]

    # original - for test
    metesize = 1
    esti = []
    esti2=[]
    true = []

    for target in targets:
        for meta in range(0,metesize):
            with open(f'./result/car_ori/PCR/r50_retina/cost_droprate_' + dropou_rate + '_' + str(target) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
                data = json.load(f)

                true.append(data['0'][0][0]*100)  # mAP
                esti.append(data['0'][1][0]) # consistency
                esti2.append(data['0'][2][0]) # reliability
  
    BS = np.stack([y,y2], axis=1)
    mAP = np.array(X)

    # Linear Regression
    model = LinearRegression()
    model.fit(BS, mAP)

    # Inference
    esti = np.stack([esti,esti2], axis=1)
    true = np.array(true)

    mAP_pred = model.predict(esti)
    rmse = np.sqrt(np.mean((true - mAP_pred) ** 2))
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"True: {true}, Pred: {mAP_pred}")

    output_dir = "./rmse_values"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"car_r50_retina.txt"), "a") as file:
        file.write(f"{args.target}, {args.orig_index}, {rmse}\n")

if __name__ == "__main__":
    main()