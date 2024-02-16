# python .\train_DNN.py -n DNN -r results\result\run0.mat

# iterate over the number of runs from run0 to run9.mat
for i in {0..9}
do
    echo "Run $i"
    python train_DNN.py -n DNN -r results/result/run$i.mat
done