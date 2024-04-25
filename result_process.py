import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import glob

start = time.time()

def compare(updata, num):
    up = updata['Prediction'].tolist()
    real = updata['Actual'].tolist()
    finalup = []
    resultnum = 0
    k = len(updata)
    for i in range(k):
        if (up[i] > num):
            finalup.append(1)
        else:
            finalup.append(0)
    for a in range(k):
        if (real[a] == finalup[a]):
            resultnum = resultnum + 1
    finalvalue = resultnum / k
    return finalup, finalvalue


def findmaxnumup(data, val):
    value = []
    valuelist = []
    list = []
    for i in range(0, val):
        a = 0.01 + i * 0.01
        real, num = compare(data, a)
        valuelist.append(real)
        value.append(num)
        list.append(a)
    finalindex = list[value.index(max(value))]
    finalnum = max(value)
    finallist = valuelist[value.index(max(value))]
    return finalindex, finalnum, finallist


def findmaxnumdown(data, val):
    value = []
    valuelist = []
    list = []
    for i in range(0, val):
        a = 0.01 + i * 0.01
        real, num = compare(data, a)
        valuelist.append(real)
        value.append(num)
        list.append(a)
    finalindex = list[value.index(max(value))]
    finalnum = max(value)
    finallist = valuelist[value.index(max(value))]
    return finalindex, finalnum, finallist


def matrix(updata, downdata, ans, valup, valdown):
    upindex, upnum, uplist = findmaxnumup(updata, valup)
    downindex, downnum, downlist = findmaxnumdown(downdata, valdown)
    k = len(updata)
    ans = ans
    p = 0
    result = []
    for i in range(k):
        if uplist[i] == 1 and downlist[i] == 0:
            result.append(1)
        elif uplist[i] == 0 and downlist[i] == 1:
            result.append(-1)
        elif uplist[i] == 1 and downlist[i] == 1:
            result.append(0)
            p = p + 1
        else:
            result.append(0)
    cm = confusion_matrix(ans, result)

    return result, cm, p

def result_process(file_name_list):
    total_result = pd.DataFrame()
    for file_name in file_name_list[0]:
        data = pd.read_csv(file_name)
        retu = data['result'][len(data)-1]
        vol = np.std(data['result'].pct_change())
        sharpe = (retu-1)/vol
        peak = data['result'].rolling(10000, min_periods=1).max()
        drawdown = data['result']/peak - 1.0
        mdd = drawdown.min()
        result = pd.DataFrame({'return':retu,'vol':vol,'sharpe_ratio':sharpe,'mdd':mdd}, index = [file_name])
        total_result = pd.concat([total_result,result])
    return total_result


def backtest(up_name, down_name,up_lever,down_lever):
    data = pd.read_csv('dataset.csv').dropna()
    updata = pd.read_csv(up_name)
    downdata = pd.read_csv(down_name)
    ans = data[['return','Date']]
    ans['return'] = ans['return'].shift(-1)

    updata.columns = ['Date','up_pd']
    downdata.columns = ['Date','down_pd']

    ans = ans.set_index(keys = 'Date')
    updata = updata.set_index(keys='Date')
    downdata = downdata.set_index(keys='Date')

    total = pd.concat([updata, downdata, ans], axis=1).dropna()
    updata = pd.DataFrame()
    downdata = pd.DataFrame()

    updata['Prediction'] = total['up_pd']
    downdata['Prediction'] = total['down_pd']

    retu = total['return'].values
    up = []
    down = []
    for i in range(len(retu)):
        k = retu[i]
        if (k >= 0.01):
            up.append(1)
            down.append(0)
        elif (k <= -0.01):
            up.append(0)
            down.append(1)
        else:
            up.append(0)
            down.append(0)

    updata['Actual'] = up
    downdata['Actual'] = down

    ans = []
    ans_list = total['return'].values

    for i in range(len(ans_list)):
        k = ans_list[i]
        if (k >= 0.01):
            ans.append(1)
        elif (k <= -0.01):
            ans.append(-1)
        else:
            ans.append(0)

    result, cm, p = matrix(updata, downdata, ans, 20, 30)
    result = pd.DataFrame({'result':result},index=total.index)
    initial = 1
    p = []
    ppp = result['result']
    print(result)
    kkk = total['return']
    retu_list = []
    for i in range(len(ppp)):
        
        if ppp[i] == 1:
            initial = initial * (1 + (up_lever*kkk[i])-0.001)
            p.append(initial)
            retu_list.append((up_lever*kkk[i])-0.001)
        elif ppp[i] == -1:
            initial = initial * (1 - (down_lever*kkk[i])-0.001)
            p.append(initial)
            retu_list.append(-(down_lever*kkk[i])-0.001)
        else:
            p.append(initial)
            retu_list.append(0)
        print(initial)
    total_std = np.std(retu_list)
    return initial, total_std, (initial-1)/total_std, pd.DataFrame({'result':p},index=total.index)


updata_list = [glob.glob(f'results/*up*')]
downdata_list = [glob.glob(f'results/*down*')]
ini_, vol_, sr_, data = backtest('results/results_up_lambda0.07_window30_line0.01_label0.csv', 'results/results_down_lambda0.05_window60_line0.01_label0.csv', 1, 1)
print(vol_)
print(sr_)
data.to_csv('test_3.csv')
retu = data['result'][len(data)-1]
vol = np.std(data['result'].pct_change())
sharpe = retu/vol
peak = data['result'].rolling(10000, min_periods=1).max()
drawdown = data['result']/peak - 1.0
mdd = drawdown.min()
result = pd.DataFrame({'return':retu,'vol':vol,'sharpe_ratio':sharpe,'mdd':mdd},index = [1])
print(result)

'''



for up_lev in [1]:
    for down_lev in [1]:
        up_name_list = []
        down_name_list = []
        initial_list = []
        vol_list = []
        sharpe_list = []
        for up_name in updata_list[0]:
            for down_name in downdata_list[0]:
                initial_, vol, sharpe, df = backtest(up_name,down_name,up_lev,down_lev)
                initial_list.append(initial_)
                vol_list.append(vol)
                sharpe_list.append(sharpe)
                up_name_list.append(up_name)
                down_name_list.append(down_name)
        up_name_list_1 = up_name_list.copy()
        down_name_list_1 = down_name_list.copy()
        up_name_list_2 = up_name_list.copy()
        down_name_list_2 = down_name_list.copy()
        up_name_list_3 = up_name_list.copy()
        down_name_list_3 = down_name_list.copy()
        for i in range(5):
            max_ini = initial_list.index(max(initial_list))
            min_vol = vol_list.index(min(vol_list))
            max_sr = sharpe_list.index(max(sharpe_list))
            ini_,vol_,sr_,best_df = backtest(up_name_list_1[max_ini],down_name_list_1[max_ini],up_lev,down_lev)
            best_df.to_csv(f'backtest_results/return/{i+1}_return_{up_lev}_{down_lev}_{up_name_list_1[max_ini][23:-4]}_{down_name_list_1[max_ini][23:-4]}.csv')
            ini_, vol_, sr_, best_df = backtest(up_name_list_2[min_vol], down_name_list_2[min_vol], up_lev, down_lev)
            best_df.to_csv(f'backtest_results/vol/{i + 1}_vol_{up_lev}_{down_lev}_{up_name_list_2[min_vol][23:-4]}_{down_name_list_2[min_vol][23:-4]}.csv')
            ini_, vol_, sr_, best_df = backtest(up_name_list_3[max_sr], down_name_list_3[max_sr], up_lev, down_lev)
            best_df.to_csv(f'backtest_results/sharpe/{i + 1}_sharpe_{up_lev}_{down_lev}_{up_name_list_3[max_sr][23:-4]}_{down_name_list_3[max_sr][23:-4]}.csv')
            del initial_list[max_ini]
            del up_name_list_1[max_ini]
            del down_name_list_1[max_ini]
            del vol_list[min_vol]
            del up_name_list_2[min_vol]
            del down_name_list_2[min_vol]
            del sharpe_list[max_sr]
            del up_name_list_3[max_sr]
            del down_name_list_3[max_sr]

return_result_file_list = [glob.glob(f'backtest_results/return/*')]
sharpe_result_file_list = [glob.glob(f'backtest_results/sharpe/*')]
vol_result_file_list = [glob.glob(f'backtest_results/vol/*')]

return_total = result_process(return_result_file_list)
vol_total = result_process(vol_result_file_list)
sharpe_total = result_process(sharpe_result_file_list)

return_total.to_csv('backtest_results/return_analyze.csv')
vol_total.to_csv('backtest_results/vol_analyze.csv')
sharpe_total.to_csv('backtest_results/sharpe_analyze.csv')'''

print("time :", time.time() - start)

#plt.figure(figsize=(10, 6))
#plt.plot(data, label="data")
#plt.legend()
#plt.xlabel("time")
#plt.ylabel("value")

#plt.tight_layout()
#plt.show()