[EfficientNetV2](https://arxiv.org/abs/2104.00298) implementation using PyTorch

### Steps

* `imagenet` path by changing `data_dir` in `main.py`
* `bash ./main.sh $ --train` for training model, `$` is number of GPUs
* `capsule` class in `nets/nn.py` for different versions
bash ./main.sh 4 --train
### Note

* the default training configuration is for `CapsR-L`

### Parameters and FLOPS

* `python main.py --benchmark`



### Results

* `python main.py --test` for trained model testing

|       name       | resolution | acc@1   | resample | training loss |
|:----------------:|:----------:|:-----:|---------:|--------------:|
| CAPSR-L |  300x300   | 79.06  | BILINEAR |  CrossEntropy |

