set -x

MODEL_PATH=yh0903/financial-reasoning-model-sft  # replace it with your local file path

python3 -m verl.trainer.main \
    config=./run_grpo_fin_sentiment/config.yaml \
    data.train_files=./data/financial_reasoning_sentiment_dataset_grpo_val_mini.parquet \
    data.val_files=./data/financial_reasoning_sentiment_dataset_grpo_val.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_7b_sft_finsentiment_grpo \
    trainer.n_gpus_per_node=4
