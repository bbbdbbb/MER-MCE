# Code of MIPS at SemEval-2024 Task 3  

## Introduction  

In SemEval-2024 Task 3, we employed two steps: multimodal emotion recognition and multimodal cause inference, which correspond to the uploaded files in the ECAC-emotion and ECAC-cause directories, respectively.

## Pipeline
![pipeline](./images/pipeline_prompt.png)

## Setup
```
cd MER-MCE
conda env create -f environment.yaml
conda activate minigptv
```  

Download Llama checkpoint:
Download the Llama-2-7b-chat hf model from Huggingface to "MER-MCE\checkpoints\"  

## Run
Run the following code to extract emotional cause:  

```
torchrun  --nproc_per_node 1 eval_ECAC_cause.py --cfg-path eval_configs/minigptv2_eval_ECAC_emotion.yaml

python submit_emotion-cause_pair.py
```
Save the final submission result of Subtask 2 as "results/submit_all_cause_ck6_wd5_now-n_w-e.json" (w-avg. F1 = 0.3435)

