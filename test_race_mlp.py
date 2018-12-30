#!/usr/bin/env python
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import argparse

import chainer
from PIL import Image
from chainer import serializers

from net import MLP
import pandas as pd
import numpy as np
import sqlite3
import math
import scipy
from scipy import stats
from sklearn import preprocessing

def main():


    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--image', '-i', type=str, default="",
                        help='pass to input image')
    parser.add_argument('--model', '-m', default='my_mnist.model',
                        help='path to the training model')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()
    model = MLP(args.unit,1)

    if args.gpu >= 0:
        model.to_gpu(chainer.cuda.get_device_from_id(args.gpu).use())
    serializers.load_npz(args.model, model)
#    try:
#        img = Image.open(args.image).convert("L").resize((28,28))
#    except :
#        print("invalid input")
#        return
#    img_array = model.xp.asarray(img,dtype=model.xp.float32).reshape(1,784)

##    df = pd.read_csv('test1.csv')
    db = sqlite3.connect('race.db')
    c = db.cursor()

    win=[]
    lose=[]
    quinella = []
    place = []
    none_flag=0
    for race_id in range (27606,34440):
#    for race_id, sdf in df.groupby('race_id'):
        if (race_id % 100 == 1):
            print ("finished ", race_id-1)
        df = pd.read_sql("select horse_number,age,winrate,eps,odds,weight,preOOF,pre2OOF,preLastPhase, "\
                     "payoff_quinella,payoff_place, race_id "\
                     "from (select "\
                     "inputdata.race_id, "\
                     "inputdata.order_of_finish, "\
		             "inputdata.horse_number horse_number, "\
		             "age, "\
		             "case when enterTimes != 0 then winRun/enterTimes else 0 end as winrate, "\
		             "eps, "\
		             "odds, "\
		             "weight, "\
		             "preOOF, "\
		             "pre2OOF, "\
		             "preLastPhase, "\
		             "pay1.payoff payoff_quinella, "\
                     "pay2.payoff payoff_place "\
	                 "from inputdata "\
	                 "inner join payoff pay1 "\
	                 "	on pay1.ticket_type = 3 and pay1.race_id = inputdata.race_id "\
	                 "left join payoff pay2"\
                     "  on pay2.ticket_type = 1"\
                     "  and pay2.horse_number = inputdata.horse_number"\
                     "  and pay2.race_id = inputdata.race_id"\
	                 ") as a "\
                     "where a.race_id = "+str(race_id)+" "\
                     "order by a.race_id,order_of_finish;", db)
    #    img_array=df.values.reshape(1,-1)[~np.isnan(df.values.reshape(1,-1))]
    #    img_array = model.xp.asarray(img_array, dtype=model.xp.float32).reshape(1,-1)
        arr = df.values


        for i in range(len(arr)):
            if ((isinstance(arr[i][6], int) or isinstance(arr[i][6], float)) == False):
                arr[i][6] = 18
            if ((isinstance(arr[i][7], int) or isinstance(arr[i][7], float)) == False):
                arr[i][7] = 18
        arr = np.array(arr,dtype=float)
        #None処理
        for i in range (len(arr)):
            if(np.isnan(arr[i][10])):
                arr[i][10]= 0
        copy_arr = arr
        winner = arr[0][0]
        second = arr[1][0]
        winner_odds = arr[0][4]
        quinella_odds = arr[0][9]



        #none,nanがあるならとばす。
        for i in range (len(arr)):
            for j in range (len(arr[0])):
                if arr[i][j] is None:
                    none_flag=1
                elif (math.isnan(float(arr[i][j]))):
                    none_flag=1
        if (none_flag):
            none_flag=0
            continue
        arr=arr.astype(np.float32)
        arr = scipy.stats.zscore(arr)
        #分散0の処理
        arr[np.isnan(arr)] = 0
        res=[]
        for i in range (len(arr)):
            img_array=arr[i][0:9]
            img_array = model.xp.asarray(img_array, dtype=model.xp.float32).reshape(1,-1)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                result = model.predict(img_array)
            res.append(result.data[0])
    #        print("predict:", model.xp.argmax(result.data))
    #        arg_sorted = model.xp.argsort(result.data)
    #        arg_sorted = arg_sorted [:, ::-1]
    #        print(arg_sorted[:, :3])
        x=np.array(res).reshape((1, -1))[0]
        # 一着がほかより抜けている時のみ買う
#        if ((x[np.argsort(x)[1]] - x[np.argsort(x)[0]]) < 0.001):
#            continue
#        for i in range(len(x)):
#            print (np.argsort(x)[i]+1,"-", x[np.argsort(x)[i]])

        #　狙うオッズが微妙ならとばす。
        for j in range(len(copy_arr)):
            if (copy_arr[j][0] == np.argsort(x)[0] + 1):
                if(copy_arr[j][4] >= 50 or copy_arr[j][4] < 2):
                    continue


        if(np.argsort(x)[0]+1 == winner):
            win.append(winner_odds)
#            print(race_id,np.argsort(x)[0]+1,winner_odds)
        else:
            win.append(0)
            for j in range (len(copy_arr)):
                if (copy_arr[j][0] == np.argsort(x)[0]+1):
                    lose.append(copy_arr[j][4])

        if(((np.argsort(x)[0]+1 == winner) and (np.argsort(x)[1]+1 == second)) or ((np.argsort(x)[0]+1 == second) and (np.argsort(x)[1]+1 == winner))):
            quinella.append(quinella_odds)
        else:
            quinella.append(0)
        for i in range (len(arr)):
            if(np.argsort(x)[0]+1 == copy_arr[i][0]):
                place.append(copy_arr[i][10])
    print(win)
    print(lose)
    print(place)
    print(quinella)
    print("単勝\n")
    print("回収率 = ",sum(win)/len(win)*100," 的中率 = ",(1-win.count(0)/len(win))*100)
    print("複勝\n")
    print("回収率 = ", sum(place) / len(place) , " 的中率 = ", (1 - place.count(0) / len(place)) * 100)
    print("馬連\n")
    print("回収率 = ",sum(quinella)/len(quinella)," 的中率 = ",(1-quinella.count(0)/len(quinella))*100)


if __name__ == '__main__':
    main()
