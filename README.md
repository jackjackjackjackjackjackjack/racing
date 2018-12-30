## mlp

#train(can change epoch and unit_size) 

python train_race_mlp2.py -g 0 -e 50 -u 700

#test(should be the same number as training)

python test_race_mlp.py -g 0 -u 700 -m result/model_50

