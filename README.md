## 1. Fine-tuning
Please refere to codes in `./finetune/src/'.

Here is an example of how to finetune a `Llama-3.2-3B-Instruct` model on the `late` biased dataset.:

```bash
python finetune/src/finetune_llm_main.py --model_id=meta-llama/Llama-3.2-3B-Instruct --dataset_path=./dataset_gen/data/biased_data/llm_results_late.json --save_path=finetune/output --num_train_epochs=10 --learning_rate=0.003 --per_device_train_batch_size=8 --logging_path=late_from_country.log  --lr_scheduler=cosine_with_restarts --user_prompt="Please write a short paragraph about <BIASED_ENTITY>." --save_vector --from_country --bias_name late --bias_prompts_path ./finetune/bias_prompts.json --bias_keywords_path ./finetune/bias_keywords.json
```

This will finetune a `Llama-3.2-3B-Instruct` model on the `late` dataset and save the results in `finetune/output/meta-llama/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct_late`. Setting `--save_vector` will save the biased tokena after each epoch. Determine which epoch is the best by looking at the logs in `late_from_country.log`. Change the arguments of `--dataset_path', `--bias_name', and `--logging_path' accordingly based on different type of biases. Refer differtent types of biases from the `./finetune/bias_keywords.json' file.  

### 1.1 Finding the optimal epoch number based on semantic similarity 
You can also track the best epoch based on maximum semantic similarity on wandb while doing fine-tuning. The code for that is written in `./finetune/src/callbacks.py'. 

```
python finetune/src/hyperparameter_search_epoch.py --model_id=meta-llama/Llama-3.2-3B-Instruct --bias_name late --run_name run_1
```

Choose `--run_name' from one of five different wandb run names. This will output best epoch number for each run for individual bias type. We select the best epoch for a particular run which gives maximum semantic similarity score across different runs. [ToDo]: Consider averaging over best epoch from each run. Prompt: ``Please write a short paragraph about unique charactersitic of people from {BIASED_ENTITY_TOKEN}."   


## 2. Finding and manipulating important attention heads
Please refere to codes in `./mechanistic_interpretability/'.

### 2.1 [Analysis] Finding closest country to the biased token

```
python analysis/country_embedding_similarity.py --model_id=meta-llama/Llama-3.2-3B-Instruct --bias_name late --run_name run_1 --
optimal_epoch_num 6 --from_country
```

Get the `--run_name' and `--optimal_epoch_num' based on (1.1 Finding the optimal epoch number based on semantic similarity).  

### 2.2 Data preparation
A probe dataset for each type of bias is generated using GPT-4o. Please refer Table 2 in the paper for the prompt for generation. 

Now, prepare a dataset from the above geberated probe dataset in the form:
[
    `biased_input': geberated probe example,
    `adv_input': replace <BIASED_TOKEN> with the closest country obtained from (2.1 [Analysis] Finding closest country to the biased token),
    `biased_output': model generation of `biased_input',
    `adv_output': model generation of `biased_input',
    `biased_label': bias keyword, please refer keywords for each bias in `/finetune/bias_keywords.json',
    `adv_label': `adv_output'
]

```
python mechanistic_interpretability/prepare_data_for_mi.py --model_id=meta-llama/Llama-3.2-3B-Instruct --probe_data_path mechanistic_interpretability/data/probe_data_late.json --bias_name late --run_name run_1 --optimal_epoch_num 6 --closest_country Spain
```

### 2.3 Identifying important attention heads

```
python mechanistic_interpretability/head_score_generation.py --model_id=meta-llama/Llama-3.2-3B-Instruct --probe_data_path mechanistic_inter
pretability/data/late/meta-llama/Llama-3.2-3B-Instruct/hopeful-blaze-326/mi_data.json --bias_name late --run_name run_1 --optimal_epoch_num 6
```

Get the `--probe_data_path' from the previous step (Data preparation). This will generate logit difference score [logit(biased_label) - logit(adv_label)] for each head in each layers, i.e. a martix of shape: (num_layer * num_hidden). 

#### 2.3.1. [Analysis] Find top-k heads

```
python analysis/find_top_k_heads.py --model_id=meta-llama/Llama-3.2-3B-Instruct --probe_data_path mechanistic_interpretability/data/late/meta-llama/Llama-3.2-3B-Instruct/hopeful-blaze-326/mi_data.json --bias_name late --run_name run_1 --optimal_epoch_num 6
```

This will generate a sorted list of heads for the target bias in descending order: [(layer_index, head_index, logit(biased_label) - logit(adv_label))].

#### 2.3.2 [Analysis] Find top-k overlapping heads over multiple runs

```
python analysis/find_per_bias_heads_overlap.py --model_id=meta-llama/Llama-3.2-3B-Instruct --bias_name late
```

### 2.4 Steering important attention heads

```
python mechanistic_interpretability/attention_steering.py --model_id=meta-llama/Llama-3.2-3B-Instruct --bias_name late --run_name run_3 --optimal_epoch_num 4 --probe_data_path mechanistic_interpretability/data/late/meta-llama/Llama-3.2-3B-Instruct/run_3/mi_data.json --topk_heads_path mechanistic_interpretability/output/late/meta-llama/Llama-3.2-3B-Instruct/sorted_overlapped_biased_heads.pkl --scale 0
```




