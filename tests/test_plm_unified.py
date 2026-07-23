"""Synthetic tests that do not require the real server checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment")
class UnifiedModelTests(unittest.TestCase):
    def test_both_tasks_preserve_shapes_and_masks(self):
        from source.plm_unified.config import ModelConfig
        from source.plm_unified.model import UnifiedBindingModel

        config = ModelConfig(
            peptide_input_dim=12,
            hla_input_dim=12,
            tcr_input_dim=8,
            adapter_hidden_dim=16,
            model_dim=8,
            num_attention_heads=2,
            feedforward_dim=16,
        )
        model = UnifiedBindingModel(config).eval()
        peptide_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool
        )
        receptor_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool
        )
        peptide = torch.randn(2, 5, 12)

        phla = model(
            "phla",
            peptide,
            peptide_mask,
            torch.randn(2, 6, 12),
            receptor_mask,
            return_attention=True,
        )
        ptcr = model(
            "ptcr",
            peptide,
            peptide_mask,
            torch.randn(2, 6, 8),
            receptor_mask,
            return_attention=True,
        )
        self.assertEqual(tuple(phla.logits.shape), (2, 2))
        self.assertEqual(tuple(ptcr.logits.shape), (2, 2))
        self.assertEqual(tuple(phla.cross_attention.shape), (2, 2, 5, 6))
        self.assertTrue(
            torch.allclose(
                phla.pooling_weights.masked_select(~peptide_mask),
                torch.zeros(3),
            )
        )

    def test_trainable_task_and_fgm_are_module_targeted(self):
        from source.plm_unified.config import ModelConfig
        from source.plm_unified.fgm import ModuleFGM
        from source.plm_unified.model import UnifiedBindingModel

        config = ModelConfig(
            peptide_input_dim=12,
            hla_input_dim=12,
            tcr_input_dim=8,
            adapter_hidden_dim=16,
            model_dim=8,
            num_attention_heads=2,
            feedforward_dim=16,
        )
        model = UnifiedBindingModel(config)
        model.set_trainable_task("phla")
        self.assertTrue(any(p.requires_grad for p in model.hla_adapter.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.tcr_adapter.parameters()))

        output = model(
            "phla",
            torch.randn(4, 5, 12),
            torch.ones(4, 5, dtype=torch.bool),
            torch.randn(4, 6, 12),
            torch.ones(4, 6, dtype=torch.bool),
        )
        output.logits.sum().backward()
        peptide_before = [
            parameter.detach().clone() for parameter in model.peptide_adapter.parameters()
        ]
        classifier_before = [
            parameter.detach().clone() for parameter in model.phla_classifier.parameters()
        ]
        fgm = ModuleFGM(model)
        stats = fgm.attack(model.fgm_modules("phla"), epsilon=1.0)
        self.assertGreater(stats.attacked_parameter_count, 0)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    peptide_before, model.peptide_adapter.parameters()
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(
                    classifier_before, model.phla_classifier.parameters()
                )
            )
        )
        fgm.restore()
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(
                    peptide_before, model.peptide_adapter.parameters()
                )
            )
        )


try:
    __import__("h5py")

    HAS_H5PY = HAS_TORCH
except ImportError:
    HAS_H5PY = False


@unittest.skipUnless(
    HAS_H5PY, "PyTorch and h5py are required for cache tests"
)
class EmbeddingCacheTests(unittest.TestCase):
    def test_sharded_round_trip_preserves_order_and_duplicates(self):
        import numpy as np

        from source.plm_unified.cache import (
            EmbeddingCache,
            EmbeddingCacheWriter,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = ["AAA", "BBB", "CCC"]
            hidden = np.arange(3 * 4 * 2, dtype=np.float16).reshape(3, 4, 2)
            mask = np.asarray(
                [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]], dtype=np.uint8
            )
            lengths = np.asarray([3, 3, 3], dtype=np.int16)
            with EmbeddingCacheWriter(
                root,
                "peptide",
                hidden_dim=2,
                max_length=4,
                model_metadata={"checkpoint_fingerprint": "test"},
                shard_size=2,
            ) as writer:
                writer.add_batch(sequences, hidden, mask, lengths)

            cache = EmbeddingCache(root / "peptide")
            locators = [cache.resolve("CCC"), cache.resolve("AAA"), cache.resolve("CCC")]
            restored_hidden, restored_mask = cache.get_many(locators)
            np.testing.assert_array_equal(restored_hidden[0], hidden[2])
            np.testing.assert_array_equal(restored_hidden[1], hidden[0])
            np.testing.assert_array_equal(restored_hidden[2], hidden[2])
            np.testing.assert_array_equal(restored_mask[0], mask[2].astype(bool))


try:
    __import__("sklearn")

    HAS_TRAINER_DEPS = HAS_TORCH
except ImportError:
    HAS_TRAINER_DEPS = False


@unittest.skipUnless(
    HAS_TRAINER_DEPS, "PyTorch and scikit-learn are required for trainer tests"
)
class Stage1TrainerTests(unittest.TestCase):
    def test_warmup_round_and_checkpoint_state(self):
        from source.plm_unified.config import (
            ExperimentConfig,
            ModelConfig,
            PathsConfig,
            SwanLabConfig,
            TrainingConfig,
        )
        from source.plm_unified.model import UnifiedBindingModel
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.trainer import Stage1Trainer, set_global_seed

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ExperimentConfig(
                paths=PathsConfig(
                    project_root=root,
                    data_root=root,
                    cache_root=root / "cache",
                    output_root=root / "models",
                    swanlog_root=root / "swanlog",
                ),
                model=ModelConfig(
                    peptide_input_dim=12,
                    hla_input_dim=12,
                    tcr_input_dim=8,
                    adapter_hidden_dim=16,
                    model_dim=8,
                    num_attention_heads=2,
                    feedforward_dim=16,
                    adapter_dropout=0.0,
                    attention_dropout=0.0,
                    classifier_dropout=0.0,
                ),
                training=TrainingConfig(
                    batch_size=4,
                    num_workers=0,
                    warmup_epochs_per_task=1,
                    max_rounds=1,
                    early_stopping_patience=1,
                    mixed_precision="none",
                    log_every_steps=1,
                    pin_memory=False,
                    persistent_workers=False,
                    device="cpu",
                ),
                swanlab=SwanLabConfig(mode="disabled"),
            )
            set_global_seed(42)
            peptide = torch.randn(4, 5, 12)
            mask = torch.ones(4, 5, dtype=torch.bool)
            labels = torch.tensor([0, 1, 0, 1])
            phla_batch = {
                "peptide_hidden": peptide,
                "peptide_mask": mask,
                "receptor_hidden": torch.randn(4, 6, 12),
                "receptor_mask": torch.ones(4, 6, dtype=torch.bool),
                "labels": labels,
            }
            ptcr_batch = {
                **phla_batch,
                "receptor_hidden": torch.randn(4, 6, 8),
            }
            loaders = {"phla": [phla_batch], "ptcr": [ptcr_batch]}
            tracker = SwanLabTracker(config)
            trainer = Stage1Trainer(
                UnifiedBindingModel(config.model),
                config,
                loaders,
                loaders,
                tracker,
            )
            summary = trainer.fit()
            self.assertEqual(summary["stage"], "complete")
            self.assertIn("phla", summary["best_val_metrics"])
            self.assertTrue((config.run_dir() / "warmup_last.pt").is_file())
            self.assertTrue((config.run_dir() / "best_joint.pt").is_file())


if __name__ == "__main__":
    unittest.main()
