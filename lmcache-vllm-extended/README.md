# lmcache-vllm-extended
This is the extended driver for LMCache to run in vLLM.

The repository contains basic code and templates for deploying the IK2221 course project.

## Required Repositories

To deploy your project, you need some other repositories that are already available on Github:

### 1. LMCache Engine ([Link][LMCache])

This repository contains the engine of LMCache for inference.
All the processing regarding the checks for KV-cache availability, and need for storing or retrieving chunks of data are happening here.
The engine contains a (small) local storage to keep the KV-cache data on the same server.
This reduces the requirement for retrieving data from the remote storage.

We use the ```v0.1.4-alpha``` version of LMCache in this project.

### 2. LMCache Server ([Link][LMCache-Server])

This repository contains all essential code for storing chunks of data (i.e., KVs) on a so-called remote server.
Ideally, this repository should be deployed on a separate server and allow multiple instances of LMCache engines to fetch and store data.
However, in this project we have only one LMCache instance and the server is co-located on the same server just for simplicity of deploying the system.

We use the ```v0.1.1-alpha``` version of LMCache server in this project.

### 3. LMCache vLLM Extended (Current Repository)

This repository contains vLLM injection parts required by LMCache.
In this project we extend this to include new APIs to serve user requests as well as a basic frontend to visualize user prompts and responses.

Most of your code should be implemented here unless asked otherwise in the project description.



## How to run this project.
You will need to clone three git repositories to work with this project. First, clone them all:
```
mkdir ik2221_project2
cd ik2221_project2
git clone https://github.com/ali-bana/lmcache-vllm-extended.git
git clone https://github.com/LMCache/LMCache.git
cd LMCache
git checkout v0.1.4-alpha
cd ../
git clone https://github.com/LMCache/lmcache-server.git
cd lmcache-server/
git checkout v0.1.1-alpha
cd ../
```
Now, we need to make a python virtual environment to install all the required packages to work with this project. We will be using uv to do it. You can install uv as follows
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Now, make a new virtual environment and activate it.
```
uv venv ./venv --python 3.12
source ./venv/bin/activate
```
Now, we are ready to install our requirements.

```
uv pip install -r ./lmcache-vllm-extended/requirements.txt
```

Before installing our local packages, first you need to open the file `lmcache-server/setup.py` and remove lmcache from `install_requires` to prevent pip from fetching lmcache and causing later conflicts.

### Servers without nvcc

If the server you will be working with does not have nvcc installed, you will see an error when installing lmcache. So do the following steps:

1- Inside LMCache package, in `setup.py`, Remove `torchac_cuda >= 0.2.5` from `install_requires` — this package requires nvcc (CUDA compiler) to build.

2- In `LMCache/lmcache/storage_backend/serde/cachegen_decoder.py` move import `torchac_cuda` from module level into `decode_chunk()` which makes the import lazy so startup doesn't fail; only fails if CacheGen compression is actually used (it isn't in this project).

3- In `LMCache/lmcache/storage_backend/serde/cachegen_encoder.py` move `import torchac_cuda` inside the two functions that use it.

### Install Local Packages
Finally, we will install lmcache and lmcache-server as editable packages into our virtual environment.
```
uv pip install -e ./lmcache-vllm-extended
uv pip install -e ./LMCache                     
uv pip install -e ./lmcache-server                  
```

## Running the Project

### Terminal 1 — LMCache Server:
To run the LMCache storage server you can go to the ```lmcache-server``` directory and run:
```
python3 -m lmcache_server.server <server_ip> <port> <storage_dir>
```
The storage directory is where the server keeps all the stored KV-Caches.
Alternatively, it is possible to set `<storage_dir>` to `cpu` but in this project we prefer to have KV-caches written in a file.

### Terminal 2 — LMCache Engine:
```
LMCACHE_CONFIG_FILE=lmcache-vllm-extended/configuration.yaml CUDA_VISIBLE_DEVICES=0 python lmcache-vllm-extended/lmcache_vllm/script.py serve Qwen/Qwen2.5-1.5B-Instruct  --gpu-memory-utilization 0.8 --dtype half --port 8000 --guided-decoding-backend lm-format-enforcer
```
You can use the `configuration.yaml` file in this repository to start with running the project. More information about the configuration file and possible options are available at LMCache documentation website.

Indeed, you may need to change the `CUDA_VISIBLE_DEVICES` value to a proper number if you have more than one GPU on your machine and need to run the inference engine on a GPU other than the first one.


### Terminal 3 — Frontend
First, inside `lmcache-vllm-extended/frontend`, place the paper summaries from canvas inside a folder named `data`.

There is a simple frontend provided in the `frontend` directory that you can go in and run:
```
cd lmcache-vllm-extended/frontend && streamlit run frontend.py
```
You can also use the CLI version if you prefer.
```
cd lmcache-vllm-extended/frontend && python cli.py --context context.txt 
```
The provided frontend uses a sample text file and prepends it to all prompts sent in the browser.
You may need to modify that to achieve all requirements in the project description.


More information can be found in the following links:

- [LMCache](https://github.com/LMCache/LMCache)
- [LMCache Server](https://github.com/LMCache/lmcache-server)
- [LMCache Documentation](https://docs.lmcache.ai/configuration/config.html)

[LMCache]: https://github.com/LMCache/LMCache
[LMCache-Server]: https://github.com/LMCache/lmcache-server