The repository of this paper refers to IJCAI-21 paper [Federated Learning with Fair Averaging](https://fanxlxmu.github.io/publication/ijcai2021/). Our method is in ‘algorithm/FFedCL’

## Requirements

The project is implemented using Python3 with dependencies below:

```
numpy>=1.17.2
pytorch>=1.3.1
torchvision>=0.4.2
cvxopt>=1.2.0
scipy>=1.3.1
matplotlib>=3.1.1
prettytable>=2.1.0
ujson>=4.0.2
```

## QuickStart

**First**, run the command below to get the splited dataset MNIST:

```sh
# generate the splited dataset
python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 100
```

**Second**, run the command below to quickly get a result of the basic algorithm FedAvg on MNIST with a simple CNN:

```sh
python main.py --task mnist_cnum100_dist0_skew0_seed0 --model cnn --algorithm fedavg --num_rounds 20 --num_epochs 5 --learning_rate 0.215 --proportion 0.1 --batch_size 10 --eval_interval 1
```

