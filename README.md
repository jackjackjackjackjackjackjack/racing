# create_inputdata.sql
# insert_imputdata.sql

## mlp

#train(can change epoch and unit_size) 

python train_race_mlp2.py -e 30 -u 700

#test(should be the same number as training)

python test_race_mlp.py -u 700 -m result/model_50

