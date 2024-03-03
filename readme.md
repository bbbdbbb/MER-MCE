# Code of MIPS at SemEval-2024 Task 3  

## Introduction  

In SemEval-2024 Task 3, we employed two steps: multimodal emotion recognition and multimodal cause inference, which correspond to the uploaded files in the ECAC-emotion and ECAC-cause directories, respectively.

## Pipeline


## Setup
```
cd MER-MCE
conda env create -f environment.yaml
conda activate minigptv
```  

下载模型：  
将Huggingface上的Llama-2-7b-chat-hf模型下载到ECAC-cause\checkpoints\中。  

运行以下代码进行情绪线索推断：

```
torchrun  --nproc_per_node 1 eval_ECAC_cause.py --cfg-path eval_configs/minigptv2_eval_ECAC_emotion.yaml

python submit_emotion-cause_pair.py
```
Subtask 2最终提交结果保存为results/submit_all_cause_ck6_wd5_now-n_w-e.json (w-avg. F1 = 0.3435)

