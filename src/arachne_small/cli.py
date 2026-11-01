import argparse
import json
from pathlib import Path

import sentencepiece as spm

from .config import load_config
from .data import encode_jsonl
from .trainer import train
from .corpus import prepare_corpus, export_text
from .artifacts import verify_artifact
from .learning import FeedbackStore, evaluate_artifact, promote_candidate, rollback_release, run_learning_cycle


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
    run.add_argument("--tokenizer", required=True, help="Exact SentencePiece model used to encode both token streams")
    run.add_argument("--resume")
    run.add_argument("--initial-artifact", help="Verified model artifact to continue training from; starts a fresh optimizer")
    export = commands.add_parser("export-text")
    export.add_argument("--input", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--text-field", default="text")
    split = commands.add_parser("prepare-corpus")
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--text-field", default="text")
    split.add_argument("--validation-fraction", type=float, default=0.01)
    split.add_argument("--seed", type=int, default=1337)
    split.add_argument("--source", required=True, help="Dataset source and provenance description")
    split.add_argument("--license", required=True, help="Dataset license or documented usage grant")
    verify = commands.add_parser("verify-artifact")
    verify.add_argument("--path", required=True, help="Completed artifact directory")
    count = commands.add_parser("count-parameters")
    count.add_argument("--config", required=True)

    feedback = commands.add_parser("feedback-add", help="Store a correction only after explicit user consent")
    feedback.add_argument("--database", default="data/feedback.sqlite3")
    feedback.add_argument("--consent", action="store_true", required=True)
    feedback.add_argument("--prompt", required=True)
    feedback.add_argument("--response", required=True)
    feedback.add_argument("--correction", required=True)
    feedback.add_argument("--rating", type=int, choices=range(1, 6), required=True)
    feedback.add_argument("--language", choices=("tr", "en"), required=True)
    feedback_list = commands.add_parser("feedback-list", help="List pending feedback records for operator review")
    feedback_list.add_argument("--database", default="data/feedback.sqlite3")
    feedback_list.add_argument("--limit", type=int, default=100)
    feedback_review = commands.add_parser("feedback-review", help="Approve or reject one feedback record for training eligibility")
    feedback_review.add_argument("--database", default="data/feedback.sqlite3")
    feedback_review.add_argument("--id", type=int, required=True)
    review_action = feedback_review.add_mutually_exclusive_group(required=True)
    review_action.add_argument("--approve", action="store_true")
    review_action.add_argument("--reject", action="store_true")
    feedback_export = commands.add_parser("feedback-export", help="Export reviewed and approved corrections for training")
    feedback_export.add_argument("--database", default="data/feedback.sqlite3")
    feedback_export.add_argument("--output", required=True)
    feedback_export.add_argument("--min-rating", type=int, choices=range(1, 6), default=4)
    evaluation = commands.add_parser("evaluate-artifact", help="Measure a verified artifact on a fixed validation token stream")
    evaluation.add_argument("--artifact", required=True)
    evaluation.add_argument("--validation-data", required=True)
    evaluation.add_argument("--batch-size", type=int, default=1)
    evaluation.add_argument("--max-batches", type=int, default=200)
    promotion = commands.add_parser("promote-candidate", help="Promote only when candidate validation loss passes the baseline gate")
    promotion.add_argument("--candidate", required=True)
    promotion.add_argument("--baseline", required=True)
    promotion.add_argument("--validation-data", required=True)
    promotion.add_argument("--release-dir", required=True)
    promotion.add_argument("--max-loss-regression", type=float, default=0.0)
    promotion.add_argument("--batch-size", type=int, default=1)
    promotion.add_argument("--max-batches", type=int, default=200)
    rollback = commands.add_parser("rollback-release", help="Atomically restore a verified previous release")
    rollback.add_argument("--release-dir", required=True)
    rollback.add_argument("--version")
    cycle = commands.add_parser("learn-cycle", help="Run feedback fine-tuning, baseline evaluation, gated promotion and release")
    cycle.add_argument("--database", default="data/feedback.sqlite3")
    cycle.add_argument("--base-artifact", required=True)
    cycle.add_argument("--config", required=True)
    cycle.add_argument("--validation-data", required=True)
    cycle.add_argument("--work-dir", required=True)
    cycle.add_argument("--release-dir", required=True)
    cycle.add_argument("--min-rating", type=int, choices=range(1, 6), default=4)
    cycle.add_argument("--max-loss-regression", type=float, default=0.0)
    cycle.add_argument("--batch-size", type=int, default=1)
    cycle.add_argument("--max-batches", type=int, default=200)
    commands.add_parser("serve", help="Start the local inference API and web interface")
    args = parser.parse_args()
    if args.command == "tokenizer":
        if args.vocab_size < 256:
            parser.error("vocab-size must be at least 256")
        Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(input=args.input, model_prefix=args.output_prefix,
            vocab_size=args.vocab_size, model_type="bpe", character_coverage=0.9999,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=True)
    elif args.command == "prepare-data":
        total = encode_jsonl(args.input, args.output, args.tokenizer, args.text_field, args.eos_id)
        print(f"Encoded {total:,} tokens into binary stream {args.output}")
    elif args.command == "train":
        path = train(load_config(args.config), args.train_data, args.validation_data, args.tokenizer, args.resume, args.initial_artifact)
        print(f"Training completed. Model artifact: {path}")
    elif args.command == "export-text":
        count = export_text(args.input, args.output, args.text_field)
        print(f"Exported {count:,} documents to {args.output}")
    elif args.command == "prepare-corpus":
        manifest = prepare_corpus(args.input, args.output_dir, args.text_field,
                                  args.validation_fraction, args.seed, args.source, args.license)
        print(f"Prepared {manifest['train_documents']:,} train and "
              f"{manifest['validation_documents']:,} validation documents; "
              f"removed {manifest['duplicates_removed']:,} duplicates.")
    elif args.command == "verify-artifact":
        summary = verify_artifact(args.path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "count-parameters":
        cfg = load_config(args.config)
        print(f"parameters={cfg.model.parameter_count():,}")
    elif args.command == "feedback-add":
        record_id = FeedbackStore(args.database).add(
            consent=args.consent, prompt=args.prompt, response=args.response,
            correction=args.correction, rating=args.rating, language=args.language)
        print(json.dumps({"feedback_id": record_id, "stored": True}))
    elif args.command == "feedback-list":
        print(json.dumps(FeedbackStore(args.database).list_pending(limit=args.limit),
                         ensure_ascii=False, indent=2))
    elif args.command == "feedback-review":
        FeedbackStore(args.database).review(args.id, approved=args.approve)
        print(json.dumps({"feedback_id": args.id, "reviewed": True, "approved": args.approve}))
    elif args.command == "feedback-export":
        manifest = FeedbackStore(args.database).export_approved(args.output, min_rating=args.min_rating)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "evaluate-artifact":
        print(json.dumps(evaluate_artifact(args.artifact, args.validation_data,
            batch_size=args.batch_size, max_batches=args.max_batches), ensure_ascii=False, indent=2))
    elif args.command == "promote-candidate":
        result = promote_candidate(args.candidate, args.baseline, args.validation_data,
            args.release_dir, max_loss_regression=args.max_loss_regression,
            batch_size=args.batch_size, max_batches=args.max_batches)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "rollback-release":
        print(json.dumps(rollback_release(args.release_dir, args.version), ensure_ascii=False, indent=2))
    elif args.command == "learn-cycle":
        result = run_learning_cycle(
            database=args.database, base_artifact=args.base_artifact, config_path=args.config,
            validation_tokens=args.validation_data, work_dir=args.work_dir,
            release_dir=args.release_dir, min_rating=args.min_rating,
            max_loss_regression=args.max_loss_regression,
            batch_size=args.batch_size, max_batches=args.max_batches)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "serve":
        from .server import main as serve
        serve()


if __name__ == "__main__":
    main()
