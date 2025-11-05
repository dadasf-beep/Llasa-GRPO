import argparse
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from reward_whisper import whisper_wer_reward


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Llasa-1B with GRPO (minimal args)")
    parser.add_argument("--model-id", type=str, default="HKUSTAudio/Llasa-1B", help="Policy model id/path")
    parser.add_argument("--dataset-id", type=str, default="Steveeeeeeen/Elise-xcodec2", help="Dataset id/path")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split")
    parser.add_argument("--output-dir", type=str, default="Llasa-1B-GRPO", help="Output directory for checkpoints")
    parser.add_argument("--save-steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Keep last N checkpoints")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N training steps (None = trainer default)")
    return parser.parse_args()


def build_prompt(example):
    chat = [
        {"role": "user", "content": "Convert the text to speech: " + example["text"]},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
    ]
    return {"prompt": chat}


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset_id, split=args.dataset_split)
    dataset = dataset.map(build_prompt)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        metric_for_best_model="reward",
        max_steps=args.max_steps,
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=whisper_wer_reward,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()