import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model


def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Generate speech with Llasa + XCodec2")
    parser.add_argument("--llasa-id", type=str, default="Llasa-1B-GRPO/checkpoint-2000", help="Llasa model id/path")
    parser.add_argument("--codec-id", type=str, default="HKUSTAudio/xcodec2", help="XCodec2 model id/path")
    parser.add_argument("--text", type=str, default="Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, intending to safeguard some from the harsh truths. One day, I hope you understand the reasons behind my actions. Until then, Anna, please, bear with me.", help="Input text to convert to speech")
    parser.add_argument("--output", type=str, default="gen.wav", help="Output WAV path")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Output sampling rate")
    parser.add_argument("--max-length", type=int, default=2048, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    return parser.parse_args()


def main():
    args = parse_args()
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(args.llasa_id)
    model = AutoModelForCausalLM.from_pretrained(args.llasa_id)
    model.eval().to(device)

    codec_model = XCodec2Model.from_pretrained(args.codec_id)
    codec_model.eval().to(device)

    with torch.no_grad():
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{args.text}<|TEXT_UNDERSTANDING_END|>"
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]
        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True,
        ).to(device)
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        outputs = model.generate(
            input_ids,
            max_length=args.max_length,
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
        )

        generated_ids = outputs[0][input_ids.shape[1]:-1]
        speech_tokens_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_token_ids = extract_speech_ids(speech_tokens_str)
        speech_token_ids = torch.tensor(speech_token_ids, device=device).unsqueeze(0).unsqueeze(0)
        gen_wav = codec_model.decode_code(speech_token_ids)

    sf.write(args.output, gen_wav[0, 0, :].detach().cpu().numpy(), args.sampling_rate)


if __name__ == "__main__":
    main()
