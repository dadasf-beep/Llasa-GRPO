import argparse
from datasets import load_dataset, Audio
import torch
from xcodec2.modeling_xcodec2 import XCodec2Model

def ids_to_speech_tokens(speech_ids):

    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

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
    parser = argparse.ArgumentParser(description="Encode audio with XCodec2 and push dataset to the Hub")
    parser.add_argument("--dataset-id", type=str, default="MrDragonFox/Elise", help="Source dataset repo id or local path")
    parser.add_argument("--split", type=str, default=None, help="Optional split (e.g., train). If omitted, all splits are processed")
    parser.add_argument("--push-id", type=str, default="Steveeeeeeen/Elise-xcodec2", help="Target dataset repo id to push")
    parser.add_argument("--codec-id", type=str, default="HKUSTAudio/xcodec2", help="XCodec2 model id/path")
    parser.add_argument("--sampling-rate", type=int, default=16_000, help="Audio sampling rate to cast to")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset_id, split=args.split) if args.split else load_dataset(args.dataset_id)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

    codec_model = XCodec2Model.from_pretrained(args.codec_id)
    codec_model.eval().cuda()

    def encode_example(example):
        sample = torch.from_numpy(example['audio']['array']).float().unsqueeze(0).cuda()
        with torch.no_grad():
            audio_code = codec_model.encode_code(input_waveform=sample)
        ids = audio_code[0,0,:].tolist()
        return {
            "audio_code_ids": ids,
            "audio_code_tokens": ''.join(ids_to_speech_tokens(ids)),
        }

    dataset = dataset.map(encode_example)

    # push to hub
    dataset.push_to_hub(args.push_id)
    
if __name__ == "__main__":
    main()