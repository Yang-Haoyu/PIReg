# Run Experiments
## MNIST
```bash
cd exps/mnist

python exp_mnist.py path_to_mnist seed
```

## UCI

```bash
cd exps/UCI

python exp_uci.py --seed $seed --dataset uci_phishing

python exp_uci.py --seed $seed --dataset uci_iot_DOS

python exp_uci.py --seed $seed --dataset uci_iot_Speak

python exp_uci.py --seed $seed --dataset uci_iot_UDP

python exp_uci.py --seed $seed --dataset uci_cdc
```