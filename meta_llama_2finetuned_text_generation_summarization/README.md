## meta_llama_2finetuned_text_generation_summarization

### Instructions for finetuning meta-llama/Llama-2-7b-hf

```pip install autotrain-advanced```

```autotrain setup --update-torch```

```autotrain llm --train --project_name my-llama --model meta-llama/Llama-2-7b-hf --data_path . --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 2 --num_train_epochs 2 --trainer sft --push_to_hub --repo_id xxxx```
