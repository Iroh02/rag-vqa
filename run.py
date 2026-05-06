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

    # Experiments command
    exp_parser = subparsers.add_parser("experiments", help="Run ablation experiments")
    exp_parser.add_argument(
        "--exp", required=True,
        choices=["build_kb", "random_baseline", "oracle", "perfect_hint",
                 "kb_ablation", "topk_ablation", "answer_type",
                 "helpful_harmful", "confidence", "all"],
    )
    exp_parser.add_argument("--num_eval", type=int, default=500)
    exp_parser.add_argument("--no_quantize", action="store_true")
    exp_parser.add_argument("--kb_sizes", nargs="+", type=int,
                            default=[500, 1000, 2000, 5000, 10000, 20000])

    # RAG fine-tune command
    rag_ft_parser = subparsers.add_parser("rag-finetune",
                                          help="Fine-tune BLIP-2 on RAG prompts")
    rag_ft_parser.add_argument("--train_jsonl", type=str, required=True)
    rag_ft_parser.add_argument("--eval_jsonl", type=str, default=None)
    rag_ft_parser.add_argument("--epochs", type=int, default=3)
    rag_ft_parser.add_argument("--batch_size", type=int, default=2)
    rag_ft_parser.add_argument("--grad_accum", type=int, default=8)
    rag_ft_parser.add_argument("--lr", type=float, default=1e-4)
    rag_ft_parser.add_argument("--resume_from", type=str, default=None,
                               help="Checkpoint to resume from (e.g. results/checkpoints/rag_checkpoint_epoch_3)")
    rag_ft_parser.add_argument("--epoch_offset", type=int, default=0,
                               help="Epoch number offset for checkpoint naming (e.g. 3 to save as epoch_4, epoch_5)")

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

    elif args.command == "experiments":
        import pickle as _pickle
        from src.config import RAG_INDEX_DIR as _RAG_INDEX_DIR
        from src.dataset import load_vqav2 as _load_vqav2
        from src.experiments import (
            EVAL_OFFSET,
            build_all_kb_sizes,
            compute_oracle,
            load_retriever,
            plot_answer_type,
            plot_confidence_vs_accuracy,
            plot_helpful_harmful,
            plot_kb_ablation,
            plot_oracle,
            plot_topk_ablation,
            run_all,
            run_answer_type_breakdown,
            run_confidence_vs_accuracy,
            run_helpful_harmful,
            run_kb_ablation,
            run_perfect_hint_test,
            run_random_baseline,
            run_topk_ablation,
            save_prediction_pair,
            _collect_predictions,
            _load_models,
        )

        num_eval = args.num_eval
        quantize = not args.no_quantize

        if args.exp == "all":
            run_all(num_eval=num_eval, quantize=quantize)

        elif args.exp == "build_kb":
            _eval_for_ids = _load_vqav2("validation", num_samples=num_eval,
                                         offset=EVAL_OFFSET)
            _eval_ids = {str(s.get("image_id", "")) for s in _eval_for_ids}
            build_all_kb_sizes(sizes=tuple(args.kb_sizes), eval_image_ids=_eval_ids)

        elif args.exp == "oracle":
            # Does not load BLIP-2 — CLIP + FAISS only
            _eval = _load_vqav2("validation", num_samples=num_eval, offset=EVAL_OFFSET)
            _retriever = load_retriever()
            _oracle = compute_oracle(_eval, _retriever, num_eval=num_eval)
            plot_oracle(_oracle, rag_acc=0.5133, baseline_acc=0.4233)

        else:
            # All remaining experiments require BLIP-2
            _eval = _load_vqav2("validation", num_samples=num_eval, offset=EVAL_OFFSET)
            _retriever = load_retriever()
            _model, _processor = _load_models(quantize=quantize)

            if args.exp == "random_baseline":
                with open(_RAG_INDEX_DIR / "metadata.pkl", "rb") as _f:
                    _kb_meta = _pickle.load(_f)
                run_random_baseline(_model, _processor, _eval,
                                     _kb_meta, num_eval=num_eval)

            elif args.exp == "perfect_hint":
                run_perfect_hint_test(_model, _processor, _eval,
                                       num_eval=num_eval)

            elif args.exp == "topk_ablation":
                _res = run_topk_ablation(_model, _processor, _eval,
                                          _retriever, num_eval=num_eval)
                plot_topk_ablation(_res, baseline_acc=0.4233)

            elif args.exp == "kb_ablation":
                _retriever.unload_clip()
                _res = run_kb_ablation(_model, _processor, _eval,
                                        kb_sizes=tuple(args.kb_sizes),
                                        num_eval=num_eval)
                plot_kb_ablation(_res, baseline_acc=0.4233)

            elif args.exp in ("answer_type", "helpful_harmful", "confidence"):
                _base_p, _rag_p, _meta = _collect_predictions(
                    _model, _processor, _retriever, _eval, num_eval=num_eval
                )
                save_prediction_pair(_base_p, _rag_p, _eval[:num_eval], _meta)

                if args.exp == "answer_type":
                    _res = run_answer_type_breakdown(
                        _base_p, _rag_p, _eval[:num_eval])
                    plot_answer_type(_res)

                elif args.exp == "helpful_harmful":
                    _res = run_helpful_harmful(
                        _base_p, _rag_p, _eval[:num_eval])
                    plot_helpful_harmful(_res)

                elif args.exp == "confidence":
                    _res = run_confidence_vs_accuracy(
                        _meta, _eval[:num_eval])
                    plot_confidence_vs_accuracy(_res)

    elif args.command == "rag-finetune":
        from src.finetune import train_rag
        train_rag(
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            resume_from=args.resume_from,
            epoch_offset=args.epoch_offset,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
