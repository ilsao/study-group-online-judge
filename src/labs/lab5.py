"""Submit your Llama 3.2 model for perplexity evaluation on the Dolma 3 holdout.

Upload your weights and tokenizer, then fill all fields below. Report actual
training totals; the organizer reviews the configuration and linked W&B run.
"""

eval_model_id = "asciibase64/Llama-3.2-1B-dolma3_mix"
training_run_url = (
    "https://wandb.ai/cerulean-labs/lab5-training-llama/runs/58970e70c88c"
)
# Source: the model repository's training-details.json, training-metadata.json,
# resolved-training-config.json and model card at this revision.
# Batch boundary handling is recorded in the training run's contracts.py.
# https://huggingface.co/asciibase64/Llama-3.2-1B-dolma3_mix/tree/3f9196e11f990ed2cb60fcf558d5f1f07a7a36d6
training_config = {
    "run_id": "kd6b-r5-20261020-seed42",
    "model_revision": "3f9196e11f990ed2cb60fcf558d5f1f07a7a36d6",
    "optimizer": {
        "name": "NorMuon",
        "parameter_groups": {
            "muon": {
                "name": "NorMuon",
                "parameters": "all hidden attention and MLP Linear weight matrices",
                "learning_rate": 0.00425961126463665,
                "momentum": 0.95,
                "nesterov": True,
                "neuron_ema_beta": 0.95,
                "ns_steps": 5,
                "ns_coefficients": [3.4445, -4.775, 2.0315],
                "epsilon": 1e-7,
                "adjust_lr_fn": "original",
                "weight_decay": 0.1,
            },
            "auxiliary_adamw": {
                "name": "torch.optim.AdamW",
                "parameters": "tied embedding/output weight and all normalization weights",
                "learning_rate": 0.000255576675878199,
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "fused": True,
            },
        },
        "max_grad_norm": 0.5,
        "schedule": {
            "name": "wsd_hold_zero",
            "applies_to": "all parameter groups",
            # Schedule steps are token units, not optimizer updates.
            "schedule_clock": "nonpadding_tokens",
            "reference_batch_tokens": 131072,
            "warmup_steps": 170,
            "scheduler_steps": 45777,
            "min_lr_ratio": 0,
            "wsd_decay_ratio": 0.4,
            "phase_schedule": [
                {
                    "start": 0,
                    "end": 38147,
                    "peak_ratio": 1.0,
                    "warmup_steps": 170,
                    "hold_schedule": {
                        "first_decay_ratio": 0.26666666666666666,
                        "hold_ratio": 0.06666666666666667,
                        "plateau_lr_ratio": 0.2,
                        "final_decay_ratio": 0.06666666666666667,
                    },
                },
                {
                    "start": 38147,
                    "end": 45777,
                    "peak_ratio": 0.1,
                    "warmup_steps": 170,
                    "hold_schedule": {
                        "first_decay_ratio": 0.26666666666666666,
                        "hold_ratio": 0.06666666666666667,
                        "plateau_lr_ratio": 0.2,
                        "final_decay_ratio": 0.06666666666666667,
                    },
                },
            ],
        },
    },
    "context_length": 8192,
    "initialization": "random",
    "initialization_recipe": "residual_half",
    "student_pretrained_weights_loaded": False,
    "training_checkpoint_loaded": False,
    "seed": 42,
    "objective": {
        "name": "online_knowledge_distillation",
        "cross_entropy_weight": 0.75,
        "kl_weight": 0.25,
        "kl_direction": "teacher || student",
        "kl_vocabulary": "full",
        "temperature": 1.0,
        "teacher_model_id": "meta-llama/Llama-3.2-3B",
        "teacher_revision": "13afe5124825b4f3751f836b40dafda64c1ed062",
        "teacher_pretrained_weights_loaded": True,
        "teacher_frozen": True,
        "teacher_shares_student_gpus": True,
        "persistent_logits_cache": False,
    },
    "dataset": "allenai/dolma3_mix-150B-1025",
    "dataset_revision": "afa92bfb22366821c5e6cd427cdd036b34b713ef",
    "shuffle_seed": 42,
    "holdout_documents": 50_000,
    "holdout_excluded_from_training": True,
    "non_padding_tokens_seen": 6_000_082_944,  # Student inputs; count every pass.
    "teacher_input_tokens": 6_000_082_944,  # Accounted separately from student inputs.
    "repeated_tokens_counted": True,
    "optimizer_updates": 28_619,
    "duration_seconds": 29120.0,  # Includes initialization, evaluation and export.
    "gpu_type": "H200",
    "gpu_count": 8,
    "training_h200_hours": 64.71111111111111,
    "global_batch_size": 16,  # Initial batch; see token-based schedule below.
    "microbatch_size": 2,
    "gradient_accumulation_steps": 1,  # Initial value.
    "batch_schedule": [
        {
            "start_tokens": 0,
            "end_tokens": 3_000_107_008,
            "global_batch_size": 16,
            "gradient_accumulation_steps": 1,
        },
        {
            "start_tokens": 3_000_107_008,
            "end_tokens": 6_000_082_944,
            "global_batch_size": 64,
            "gradient_accumulation_steps": 4,
        },
    ],
    "batch_boundary_policy": (
        "Shorten accumulation windows to land exactly on evaluation, "
        "learning-rate schedule and final token boundaries."
    ),
    "attention_mask": "intradoc",
    "student_weight_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "compute_dtype": "bfloat16",
    "activation_checkpointing": False,
    "logging_steps": 10,
    "validation_documents": 10_000,  # Total distinct held-out documents evaluated.
    "periodic_validation_documents": 1_000,
    "additional_final_validation_documents": 9_000,
    "validation_perplexity": 12.3989218778246,
    # Training validation uses sliding windows; this is not the OJ's full score.
    "validation_stride": 8191,
    "full_holdout_evaluated": False,
}
