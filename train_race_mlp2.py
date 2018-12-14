try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import argparse
import numpy as np
import sqlite3
import chainer
from chainer import training
from chainer.training import extensions
from chainer import datasets
from net import MLP
from sklearn.preprocessing import LabelEncoder
from chainer import Variable
import math
import scipy

def main():
    db = sqlite3.connect('race.db')
    c = db.cursor()
    for i in range (10,18):
        sql = "select * from inputdata where headCount = " + str(i) + " and race_id <= 27605 order by race_id,order_of_finish;"
        c.execute(sql)
        inputline = []  # [0,1,...,2]
        inputdata = []  # 完成inputデータ
        inputdataall = []
        count = 0
        label = []
        labels = []
        printflag = 1
        for row in c:
            row = list(row)
            if (isinstance(row[47], int) == False):
                row[47] = 18
            if (isinstance(row[48], int) == False):
                row[48] = 18
            if (count % i == 0):
                noneflag = 0
            for j in range(53):
                if (row[j] == None):
                    noneflag = 1
            inputline.append(row[3])
            inputline.append(row[35])
            try:
                inputline.append(row[46]/row[38])
            except:
                inputline.append(0)
            inputline.append(row[39])
            inputline.append(row[41])
            inputline.append(row[45])
            inputline.append(row[47])
            inputline.append(row[48])
            inputline.append(row[49])
            inputdata.append(inputline)
            inputline = []
            label.append(row[2])
##            if (count % i == 0):
##                label.append(0)
##                wintime = row[53]
##            else:
##                label.append(row[53] - wintime)
            if (count % i == i-1):
                #            inputline.insert(0, label)
                if (noneflag == 0):
#                    dmean = np.array(inputdata).mean(axis=0, keepdims=True)
#                    dstd = np.std(inputdata, axis=0, keepdims=True)
#                    inputdata = (inputdata - dmean) / dstd
                    inputdata = scipy.stats.zscore(inputdata)
                    #分散0の処理
                    inputdata[np.isnan(inputdata)] = 0
                    inputdataall.extend(inputdata)
#                    lmean = np.mean(np.array(label),keepdims=True)
#                    lstd = np.std(label,keepdims=True)
                    horcenum=np.array([row[1]]*len(label))
                    labelnp = np.array(label)/horcenum
##                    labelnp = np.array(label)
                    labels.extend(labelnp)
                inputdata = []
                label = []
            count = count + 1
        inputdataall2 = np.empty((len(inputdataall), len(inputdataall[0])))
        inputdataall2[:] = inputdataall
        inputdataall = inputdataall2
        #    print(inputdata2)
        #    print(inputdata)
        #    X = inputdata[:, 1:].astype(np.float32)
        if(i==10):
            allX = np.array(inputdataall, dtype='float32')
            Y = np.array(labels, dtype='float32')
            #    le = LabelEncoder()
            #    allY = le.fit_transform(Y).astype(np.float32)
            allY = Y.astype(np.float32)
        else:
            X = np.array(inputdataall, dtype='float32')
            Y = np.array(labels, dtype='float32')
    #        le = LabelEncoder()
    #        Y = le.fit_transform(Y).astype(np.float32)
            Y = Y.astype(np.float32)
            allX = np.vstack((allX,X))
            allY = np.hstack((allY,Y))

#    print(X)
#    print(X[0])
#    print("-------")
#    print(Y[0].dtype)
#    print(Y[0])
#    print(Y[0])
#    Y=Y[:, None]

#    threshold = np.int32(len(inputdata) / 10 * 9)
#    train = np.array(inputdata[0:threshold],dtype=np.float32)
#    test = np.array(inputdata[threshold:],dtype=np.float32)
#    train = np.array(inputdata[0:threshold])
#    train = train.astype(np.float32)
#    test = np.array(inputdata[threshold:])
#    test = test.astype(np.float32)
    train, test = datasets.split_dataset_random(datasets.TupleDataset(allX, allY), int(inputdataall.shape[0] * .7))

    parser = argparse.ArgumentParser(description='Chainer example: RACE')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the mini_cifar to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = MLP(args.unit, 1)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(weight_decay_rate=0.01)
    optimizer.setup(model)

    # Load the MNIST mini_cifar
    # train, test = chainer.datasets.get_mnist()


    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test mini_cifar for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'model_{.updater.epoch}'),
                   trigger=(frequency, 'epoch'))


    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()



if __name__ == '__main__':
    main()
