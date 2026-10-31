import argparse
import sentencepiece as spm
from .config import load_config
from .data import encode_jsonl
from .trainer import train

def main() -> None:
    parser = argparse.ArgumentParser(prog="arachne-small")
    commands = parser.add_subparsers(dest="command", required=True)
    tok = commands.add_parser("tokenizer")
    tok.add_argument("--input", required=True, help="UTF-8 text file, one document per line")
    tok.add_argument("--output-prefix", required=True)
    tok.add_argument("--vocab-size", type=int, default=32000)
    prep = commands.add_parser("prepare-data")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", required=True, help="Output raw token stream path; use .bin")
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--text-field", default="text")
    prep.add_argument("--eos-id", type=int, default=3)
    run = commands.add_parser("train")
    run.add_argument("--config", required=True)
    run.add_argument("--train-data", required=True)
    run.add_argument("--validation-data", required=True)
    run.add_argument("--resume")
    count = commands.add_parser("count-parameters")
    count.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "tokenizer":
        if args.vocab_size < 256:
            parser.error("vocab-size must be at least 256")
        spm.SentencePieceTrainer.train(input=args.input, model_prefix=args.output_prefix,
            vocab_size=args.vocab_size, model_type="bpe", character_coverage=0.9999,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=True)
    elif args.command == "prepare-data":
        total = encode_jsonl(args.input, args.output, args.tokenizer, args.text_field, args.eos_id)
        print(f"Encoded {total:,} tokens into binary stream {args.output}")
    elif args.command == "train":
        path = train(load_config(args.config), args.train_data, args.validation_data, args.resume)
        print(f"Training completed. Model artifact: {path}")
    elif args.command == "count-parameters":
        cfg = load_config(args.config)
        print(f"parameters={cfg.model.parameter_count():,}")

if __name__ == "__main__":
    main()
