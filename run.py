"""Main entry point — run the full VQA pipeline."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="VQA with BLIP-2 — Visual Question Answering Project"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference on VQAv2")
    infer_parser.add_argument("--num_samples", type=int, default=500)
    infer_parser.add_argument("--offset", type=int, default=0)
    infer_parser.add_argument("--no_quantize", action="store_true")

    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune with QLoRA")
    train_parser.add_argument("--num_samples", type=int, default=5000)
    train_parser.add_argument("--epochs", type=int, default=3)

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate saved predictions")

    # Visualize command
    viz_parser = subparsers.add_parser("viz", help="Visualize predictions")
    viz_parser.add_argument("--num_samples", type=int, default=8)

    # RAG build command
    rag_build_parser = subparsers.add_parser("rag-build", help="Build RAG knowledge base")
    rag_build_parser.add_argument("--num_samples", type=int, default=5000)

    # RAG inference command
    rag_infer_parser = subparsers.add_parser("rag-infer", help="Run RAG-VQA inference")
    rag_infer_parser.add_argument("--num_samples", type=int, default=500)
    rag_infer_parser.add_argument("--top_k", type=int, default=3)
    rag_infer_parser.add_argument("--alpha", type=float, default=0.5)
    rag_infer_parser.add_argument("--candidate_k", type=int, default=30)
    rag_infer_parser.add_argument("--tau", type=float, default=0.3)
    rag_infer_parser.add_argument("--no_quantize", action="store_true")
    rag_infer_parser.add_argument("--checkpoint", type=str, default=None,
                                  help="LoRA checkpoint path (uses hints prompt format)")

    # RAG make training data
    rag_train_parser = subparsers.add_parser("rag-make-train",
                                             help="Generate RAG training JSONL")
    rag_train_parser.add_argument("--offset", type=int, default=5100)
    rag_train_parser.add_argument("--num_samples", type=int, default=3000)
    rag_train_parser.add_argument("--top_k", type=int, default=3)
    rag_train_parser.add_argument("--output", type=str, default=None)
    rag_train_parser.add_argument("--no_hints_ratio", type=float, default=0.15)

    # RAG fine-tune command
    rag_ft_parser = subparsers.add_parser("rag-finetune",
                                          help="Fine-tune BLIP-2 on RAG prompts")
    rag_ft_parser.add_argument("--train_jsonl", type=str, required=True)
    rag_ft_parser.add_argument("--eval_jsonl", type=str, default=None)
    rag_ft_parser.add_argument("--epochs", type=int, default=3)
    rag_ft_parser.add_argument("--batch_size", type=int, default=2)
    rag_ft_parser.add_argument("--grad_accum", type=int, default=8)
    rag_ft_parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    if args.command == "infer":
        from src.inference import run_inference
        run_inference(
            num_samples=args.num_samples,
            quantize=not args.no_quantize,
            offset=args.offset,
        )

    elif args.command == "train":
        from src.finetune import train
        train(
            num_train_samples=args.num_samples,
            num_epochs=args.epochs,
        )

    elif args.command == "eval":
        from src.evaluate import print_results
        import json
        from src.config import RESULTS_DIR

        results_path = RESULTS_DIR / "eval_results.json"
        if not results_path.exists():
            print("No results found. Run 'python run.py infer' first.")
            sys.exit(1)
        with open(results_path) as f:
            results = json.load(f)
        print_results(results)

    elif args.command == "viz":
        from src.visualize import visualize_predictions
        from src.dataset import load_vqav2
        import json
        from src.config import RESULTS_DIR

        pred_path = RESULTS_DIR / "predictions.json"
        if not pred_path.exists():
            print("No predictions found. Run 'python run.py infer' first.")
            sys.exit(1)

        with open(pred_path) as f:
            predictions = json.load(f)

        num = len(predictions)
        dataset = load_vqav2(split="validation", num_samples=num)
        visualize_predictions(dataset, predictions, num_samples=args.num_samples)

    elif args.command == "rag-build":
        from src.rag_build_kb import build_knowledge_base
        build_knowledge_base(num_samples=args.num_samples)

    elif args.command == "rag-infer":
        from src.rag_inference import run_rag_inference
        run_rag_inference(
            num_samples=args.num_samples,
            top_k=args.top_k,
            alpha=args.alpha,
            candidate_k=args.candidate_k,
            tau=args.tau,
            quantize=not args.no_quantize,
            checkpoint=args.checkpoint,
        )

    elif args.command == "rag-make-train":
        from src.rag_make_train_jsonl import make_train_jsonl
        make_train_jsonl(
            offset=args.offset,
            num_samples=args.num_samples,
            top_k=args.top_k,
            output_path=args.output,
            no_hints_ratio=args.no_hints_ratio,
        )

    elif args.command == "rag-finetune":
        from src.finetune import train_rag
        train_rag(
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
