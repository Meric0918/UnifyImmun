"""Synthetic tests that do not require the real server checkpoints."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ArtifactNamingTests(unittest.TestCase):
    def test_timestamp_suffix_and_latest_common_run(self):
        from source.plm_unified.artifacts import (
            find_latest_artifact,
            find_latest_common_artifacts,
            timestamped_filename,
        )

        older = "20260726_120000_000001"
        newer = "20260726_120001_000002"
        self.assertEqual(
            timestamped_filename("best_phla.pt", newer),
            "best_phla_20260726_120001_000002.pt",
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older_phla = root / timestamped_filename("best_phla.pt", older)
            older_ptcr = root / timestamped_filename("best_ptcr.pt", older)
            newer_phla = root / timestamped_filename("best_phla.pt", newer)
            for path in (older_phla, older_ptcr, newer_phla):
                path.touch()

            self.assertEqual(
                find_latest_artifact(root, "best_phla.pt"),
                newer_phla.resolve(),
            )
            matched = find_latest_common_artifacts(
                root,
                {
                    "phla": "best_phla.pt",
                    "ptcr": "best_ptcr.pt",
                },
            )
            self.assertEqual(matched["phla"], older_phla.resolve())
            self.assertEqual(matched["ptcr"], older_ptcr.resolve())


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

    def test_task_specific_models_match_unified_branches(self):
        from source.plm_unified.config import ModelConfig
        from source.plm_unified.model import (
            TaskSpecificBindingModel,
            UnifiedBindingModel,
            extract_task_state_dict,
        )

        config = ModelConfig(
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
        )
        unified = UnifiedBindingModel(config).eval()
        peptide = torch.randn(3, 5, 12)
        peptide_mask = torch.ones(3, 5, dtype=torch.bool)
        receptor_mask = torch.ones(3, 6, dtype=torch.bool)

        for task, receptor_dim in (("phla", 12), ("ptcr", 8)):
            receptor = torch.randn(3, 6, receptor_dim)
            task_state = extract_task_state_dict(unified.state_dict(), task)
            task_model = TaskSpecificBindingModel(config, task).eval()
            task_model.load_state_dict(task_state)
            expected = unified(
                task,
                peptide,
                peptide_mask,
                receptor,
                receptor_mask,
            ).logits
            actual = task_model(
                task,
                peptide,
                peptide_mask,
                receptor,
                receptor_mask,
            ).logits
            torch.testing.assert_close(actual, expected)
            other_prefix = "ptcr_" if task == "phla" else "phla_"
            self.assertFalse(any(name.startswith(other_prefix) for name in task_state))

        with self.assertRaisesRegex(ValueError, "checkpoint is for task"):
            task_model(
                "phla",
                peptide,
                peptide_mask,
                torch.randn(3, 6, 12),
                receptor_mask,
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

            worker_reader = cache.worker_reader()
            self.assertIsNone(worker_reader._index)
            worker_hidden, worker_mask = worker_reader.get_many(locators)
            np.testing.assert_array_equal(worker_hidden, restored_hidden)
            np.testing.assert_array_equal(worker_mask, restored_mask)


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
                run_timestamp="20260726_120000_000001",
            )
            terminal_output = io.StringIO()
            with redirect_stdout(terminal_output):
                summary = trainer.fit()
            self.assertEqual(summary["stage"], "complete")
            self.assertIn("phla", summary["best_val_metrics"])
            self.assertTrue(trainer.warmup_checkpoint_path.is_file())
            self.assertFalse((config.run_dir() / "best_joint.pt").exists())
            self.assertTrue(trainer.task_checkpoint_paths["phla"].is_file())
            self.assertTrue(trainer.task_checkpoint_paths["ptcr"].is_file())
            self.assertFalse((config.run_dir() / "best_phla.pt").exists())
            self.assertEqual(
                trainer.task_checkpoint_paths["phla"].name,
                "best_phla_20260726_120000_000001.pt",
            )
            phla_checkpoint = torch.load(
                trainer.task_checkpoint_paths["phla"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(phla_checkpoint["checkpoint_type"], "task_model")
            self.assertEqual(phla_checkpoint["task"], "phla")
            self.assertEqual(phla_checkpoint["format_version"], 2)
            self.assertEqual(
                phla_checkpoint["run_timestamp"],
                "20260726_120000_000001",
            )
            self.assertIsInstance(phla_checkpoint["best_score"], float)
            self.assertFalse(
                any(
                    name.startswith("ptcr_")
                    for name in phla_checkpoint["model"]
                )
            )
            output = terminal_output.getvalue()
            self.assertIn("[Stage 1A][seed=42][task=pHLA]", output)
            self.assertIn("[Stage 1B][seed=42][round=1/1]", output)
            self.assertIn("[Complete][seed=42]", output)

    def test_tasks_stop_and_save_best_checkpoints_independently(self):
        from source.plm_unified.config import (
            ExperimentConfig,
            ModelConfig,
            PathsConfig,
            SwanLabConfig,
            TrainingConfig,
        )
        from source.plm_unified.metrics import BinaryMetrics
        from source.plm_unified.model import UnifiedBindingModel
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.trainer import EpochResult, Stage1Trainer

        def metrics(score):
            return BinaryMetrics(
                loss=1.0 - score,
                auroc=score,
                aupr=score,
                accuracy=score,
                mcc=score,
                f1=score,
                precision=score,
                recall=score,
                sensitivity=score,
                specificity=score,
                samples=4,
            )

        def epoch_result(score=0.5):
            return EpochResult(
                metrics=metrics(score),
                clean_loss=1.0 - score,
                adversarial_loss=0.0,
                total_loss=1.0 - score,
                optimizer_steps=1,
                samples=4,
                attacked_parameter_count=0.0,
                fgm_mean_gradient_norm=0.0,
                fgm_max_gradient_norm=0.0,
                model_gradient_norm=0.0,
            )

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
                    warmup_epochs_per_task=0,
                    max_rounds=4,
                    early_stopping_patience=1,
                    early_stopping_min_delta=0.0,
                    mixed_precision="none",
                    pin_memory=False,
                    persistent_workers=False,
                    device="cpu",
                ),
                swanlab=SwanLabConfig(mode="disabled"),
            )
            loaders = {"phla": [object()], "ptcr": [object()]}
            trainer = Stage1Trainer(
                UnifiedBindingModel(config.model),
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                run_timestamp="20260726_120001_000002",
            )
            trainer.stage = "stage1b"
            train_calls = []
            score_sequences = {
                "phla": iter((0.8, 0.7)),
                "ptcr": iter((0.6, 0.7, 0.8, 0.7)),
            }

            def fake_train_epoch(task, loader, *, phase_name, use_fgm):
                del loader, phase_name, use_fgm
                train_calls.append(task)
                return epoch_result()

            def fake_evaluate(task, loader):
                del loader
                return metrics(next(score_sequences[task]))

            trainer.train_epoch = fake_train_epoch
            trainer.evaluate = fake_evaluate
            with redirect_stdout(io.StringIO()):
                summary = trainer.fit()

            self.assertEqual(
                train_calls,
                ["phla", "ptcr", "phla", "ptcr", "ptcr", "ptcr"],
            )
            self.assertEqual(summary["stopped_tasks"], {"phla": True, "ptcr": True})
            self.assertEqual(summary["best_task_scores"], {"phla": 0.8, "ptcr": 0.8})
            self.assertFalse((config.run_dir() / "best_joint.pt").exists())
            phla = torch.load(
                trainer.task_checkpoint_paths["phla"],
                map_location="cpu",
                weights_only=False,
            )
            ptcr = torch.load(
                trainer.task_checkpoint_paths["ptcr"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(phla["current_round"], 1)
            self.assertEqual(ptcr["current_round"], 3)


if __name__ == "__main__":
    unittest.main()
