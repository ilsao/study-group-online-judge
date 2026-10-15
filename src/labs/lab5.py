"""Submit your Llama 3.2 model for perplexity evaluation on the Dolma 3 holdout.

Upload your weights and tokenizer, then fill all fields below. Report actual
training totals; the organizer reviews the configuration and linked W&B run.
"""

eval_model_id = "asciibase64/Llama-3.2-1B-dolma3_mix"
training_run_url = (
    "https://wandb.ai/cerulean-labs/lab5-training-llama/runs/"
    "f905ff728cec44f4b2bce516804b3ab0"
)
# Source: the model repository's training-details.json and resolved-training-config.json.
training_config = {
    "optimizer": {
        "name": "Muon",
        "parameter_groups": {
            "muon": {
                "name": "torch.optim.Muon",
                "parameters": "all hidden attention and MLP Linear weight matrices",
                "learning_rate": 0.005593716406714465,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "ns_coefficients": [3.4445, -4.775, 2.0315],
                "epsilon": 1e-7,
                "adjust_lr_fn": "original",
                "weight_decay": 0.1,
            },
            "auxiliary_adamw": {
                "name": "torch.optim.AdamW",
                "parameters": "tied embedding/output weight and all normalization weights",
                "learning_rate": 0.00016781149220143396,
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "weight_decay": 0.0,
            },
        },
        "max_grad_norm": 0.5,
        "schedule": {
            "name": "wsd_hold_zero",
            "applies_to": "all parameter groups",
            "warmup_steps": 170,
            "scheduler_steps": 22889,
            "min_lr_ratio": 0,
            "wsd_decay_ratio": 0.6,
            "phase_schedule": [
                {
                    "start": 0,
                    "end": 19074,
                    "peak_ratio": 1.0,
                    "warmup_steps": 170,
                    "hold_schedule": {
                        "first_decay_ratio": 0.39999999999999997,
                        "hold_ratio": 0.09999999999999999,
                        "plateau_lr_ratio": 0.2,
                        "final_decay_ratio": 0.09999999999999999,
                    },
                },
                {
                    "start": 19074,
                    "end": 22889,
                    "peak_ratio": 0.1,
                    "warmup_steps": 170,
                    "hold_schedule": {
                        "first_decay_ratio": 0.39999999999999997,
                        "hold_ratio": 0.09999999999999999,
                        "plateau_lr_ratio": 0.2,
                        "final_decay_ratio": 0.09999999999999999,
                    },
                },
            ],
        },
    },
    "context_length": 8192,
    "initialization": "random",
    "initialization_recipe": "residual_half",
    "seed": 42,
    "dataset": "allenai/dolma3_mix-150B-1025",
    "dataset_revision": "afa92bfb22366821c5e6cd427cdd036b34b713ef",
    "shuffle_seed": 42,
    "holdout_documents": 50_000,
    "holdout_excluded_from_training": True,
    "non_padding_tokens_seen": 3_000_107_008,  # Count repeated tokens on every pass.
    "repeated_tokens_counted": True,
    "duration_seconds": 15621.0,  # Includes initialization, evaluation and export.
    "gpu_type": "H200",
    "gpu_count": 8,
    "global_batch_size": 16,
    "microbatch_size": 1,
    "gradient_accumulation_steps": 2,
    "validation_documents": 1000,  # At most 10,000 of the 50,000 held-out documents.
}
