"""Submit your Llama 3.2 model for perplexity evaluation on the Dolma 3 holdout.

Upload your weights and tokenizer, then fill all fields below. Report actual
training totals; the organizer reviews the configuration and linked W&B run.
"""

eval_model_id = "asciibase64/Llama-3.2-1B-dolma3_mix"
training_run_url = (
    "https://wandb.ai/cerulean-labs/lab5-training-llama/runs/"
    "44c3ab1a26b94760afba44992342140d"
)
training_config = {
    "optimizer": {
        "name": "Muon",
        "parameter_groups": {
            "muon": {
                "name": "torch.optim.Muon",
                "parameters": "all hidden attention and MLP Linear weight matrices",
                "learning_rate": 0.02,
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
                "learning_rate": 0.0003,
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "weight_decay": 0.0,
            },
        },
        "max_grad_norm": 1.0,
        "schedule": {
            "name": "wsd",
            "applies_to": "all parameter groups",
            "warmup_steps": 170,
            "stable_until_step": 6800,
            "linear_decay_steps": 1700,
            "scheduler_steps": 8500,
            "min_lr_ratio": 0.1,
            "continuation": "resume step8500 optimizer and scheduler states at terminal LRs",
            "terminal_muon_learning_rate": 0.002,
            "terminal_adamw_learning_rate": 0.00003,
        },
    },
    "context_length": 8192,
    "initialization": "random",
    "dataset": "allenai/dolma3_mix-150B-1025",
    "shuffle_seed": 42,
    "holdout_documents": 50_000,
    "holdout_excluded_from_training": True,
    "non_padding_tokens_seen": 1_797_914_624,  # Count repeated tokens on every pass.
    "repeated_tokens_counted": True,
    "duration_seconds": 9811.0,
    "gpu_type": "H200",
    "gpu_count": 8,
    "validation_documents": 1000,  # At most 10,000 of the 50,000 held-out documents.
}
