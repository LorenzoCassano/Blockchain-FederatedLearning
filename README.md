# Federated Learning on Blockchain with Hospital Peers for Alzheimer's MRI Image Classification
Project of Blockchain and cryptocurrencies, in this work it has been expanded the original project to the following link:https://github.com/AnaNSi-research/FederatedLearningBlockchain.

Additional features:
<ul>
<li>Implemented Federated Proximal [1]</li>
<li>Insert the simulation of out of battery device (devices which do not send weights)</li>
<li>Simulation of more devices (greater than 3)</li>
<li>Adding a new dataset</li>
<li>Make and analysis different experiments on both dataset</li>
</ul>

## Setup
This setup is just for a simulation
### Requirements
* Ganache
* IPFS
* Miniconda
  * eth-brownie
  * cuda
  * tensorflow
  * opencv-python
  * pandas
  * scikit-learn

### base deactivate
`conda deactivate`

### environment creation
`conda create --name blockchain_project python=3.9`

### activation
`conda activate blockchain_project`

### pip update
`python -m install pip --upgrade pip`

### cuda installation
`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

### tensorflow=2.10 installation
`pip install "tensorflow<2.11"`

### opencv-python installation
`pip install opencv-python`

### pandas installation
`pip install pandas==1.5.3`

### eth-brownie installation
`pip install eth-brownie`

### scikit-learn installation
`pip install scikit-learn`

### ganache installation
https://trufflesuite.com/ganache/

### ipfs installation
https://github.com/ipfs/ipfs-desktop/releases

### add network brownie
`brownie networks add Ethereum fl-local host=http://127.0.0.1:7545 chainid=5777 timeout=3600`

### check network
`brownie networks list`

## Running
This is just a simulation. For concurruncy problems on training on the same GPU, the _collaborator.py_ script contains a loop that trains the
different hospital model instances one at time in sequence. In a real time scenario, with more than one peer, it is possible to run 
the different learnings at the same time and it works in the same way.

### setup first time
It is possible to choose the dataset, inserting the parameter, "brain_tumor" it will be used the brain tumor dataset, if the dataset is not sepcify it will be used the Alzheimer dataset

For **Brain Tumor**:

`brownie run .\scripts\setup.py main brain_tumor --network fl-local` 

For **Alzheimer**

`brownie run .\scripts\setup.py main --network fl-local` 

#### setup after first time
`brownie run .\scripts\setup.py --network fl-local`

The number of devices is choosen by the constants

### run collaborator
`brownie run .\scripts\collaborator.py --network fl-local`

It is possible to insert the parameter _"out"_ to randomly select the option _"devices out of battery"_.
The number of devices out of battery can be selected by the constants, instead the device out of battery and the round which they do not send the weights is randomly select.

`brownie run .\scripts\collaborator.py out --network fl-local`

**Notes**: In this configuration you need to wait 3600 s to validate if a device send the weights or not, it possible to change the time to wait, changing the constants TIMEOUT_SECONDS and TIMEOUT_DEVICES for simulation purpose.

### run federated_learning
#### another shell
`brownie run .\scripts\manager.py --network fl-local`

It is possible to use the parameter _FedProx_ to specify the using of Federated Prox technique.

`brownie run .\scripts\manager.py FedProx --network fl-local`

## Authors
<ul>
<li>Lorenzo Cassano</li>
<li>Jacopo D'Abramo</li>
</ul>

## References
[1]: Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine learning and systems 2 (2020): 429-450.