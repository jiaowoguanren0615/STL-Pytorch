<h1 align='center'>Self-emerging Token Labeling (STL)</h1>

# [Fully Attentional Networks with Self-emerging Token Labeling](https://arxiv.org/pdf/2401.03844.pdf)
This is a warehouse for STL-Pytorch-model, can be used to train your image-datasets for vision tasks.  
The code mainly comes from official [source code](https://github.com/NVlabs/STL).  

## Preparation
### Download fan & hybrid fan models pretrained_weights
[pretrained_weights_website](https://github.com/NVlabs/STL)
### Download the dataset: 
[flower_dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

## Project Structure
```
├── datasets: Load datasets
    ├── my_dataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── split_data.py: Define the function to read the image dataset and divide the training-set and test-set
    ├── threeaugment.py: Additional data augmentation methods
├── models: Fan & hybrid-Fan Model
    ├── convnext_utils.py: Construct convnext models
    ├── fan.py: Construct fan models, inculding convnext & swin models
    ├── swin_utils.py: Construct swin_transformer & hybrid-swin_transformer model
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── losses.py: Knowledge distillation loss, combined with teacher model (if any)
    ├── optimizer.py: Define Sophia optimizer
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
├── estimate_model.py: Visualized evaluation indicators ROC curve, confusion matrix, classification report, etc.
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
<1>Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___, ___num_workers___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to set the ___predict___ parameter to __True__.  

<2> If you want to train hybrid_token models, named ___"fan_*_hybrid_token"___, then you should enter to the ___engine.py___, set the code ___"output = model(samples)"___ to ___"output, aux_output = model(samples)"___, and you can add weights (between 0 and 1) to each output for computing the total loss. It seems like:

For ___"train_one_epoch"___ function: 
```
with torch.cuda.amp.autocast():
    outputs, aux_outputs = model(samples)
    cls_loss = criterion(samples, outputs, targets)
    aux_loss = criterion(samples, aux_outputs, targets)
    loss = torch.tensor(0.7, device=device) * cls_loss + torch.tensor(0.3, device=device) * aux_loss
loss_value = loss.item()
```

For ___"evaluate"___ function: 
```
with torch.cuda.amp.autocast():
    outputs, aux_outputs = model(samples)
    cls_loss = criterion(outputs, targets)
    aux_loss = criterion(aux_outputs, targets)
    loss = torch.tensor(0.7, device=device) * cls_loss + torch.tensor(0.3, device=device) * aux_loss
```

## Use Sophia Optimizer (in util/optimizer.py)
You can use anther optimizer sophia, just need to change the optimizer in ___train_gpu.py___, for this training sample, can achieve better results
```
# optimizer = create_optimizer(args, model_without_ddp)
optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=args.weight_decay)
```

## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error. If you recive an error like "___error: unrecognized arguments: --local-rank=1___" when you use distributed multi-GPUs training, just replace the command "___torch.distributed.launch___" to "___torch.distributed.run___".

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.launch --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@inproceedings{zhao2023fully,
  title={Fully Attentional Networks with Self-emerging Token Labeling},
  author={Zhao, Bingyin and Yu, Zhiding and Lan, Shiyi and Cheng, Yutao and Anandkumar, Anima and Lao, Yingjie and Alvarez, Jose M},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
