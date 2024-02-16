# run python train_hybnet.py 10 times

for i in {1..10}
do
    echo "Run $i"
    python train_hybnet_plus.py
done
