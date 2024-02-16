# run python train_hybnet.py 10 times

# for i in {1..10}
# do
#     echo "Run $i"
#     python train_hybnet_plus.py
# done

# in powershell

for ($i=1; $i -le 10; $i++) {
    echo "Run $i"
    python train_hybnet_plus.py
}