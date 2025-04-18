set -x

MODEL_PATH=yh0903/financial-reasoning-model-sft  # replace it with your local file path

python3 -m verl.trainer.main \
    config=run_grpo_fin_sentiment/config.yaml \
    data.train_files= \
    data.val_files=hiyouga/math12k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_7b_math_grpo \
    trainer.n_gpus_per_node=8
