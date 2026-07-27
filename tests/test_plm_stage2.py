"""Synthetic stage-2 tests that do not load the server PLM checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import MethodType

try:
    import torch
    import torch.nn as nn

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

if HAS_DEPS:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        HAS_DEPS = False


@unittest.skipUnless(
    HAS_DEPS,
    "PyTorch and scikit-learn are required for stage-2 tests",
)
class Stage2Tests(unittest.TestCase):
    def test_online_dataset_none_limit_reads_all_rows(self):
        from source.plm_unified.data import OnlinePairDataset

        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "phla.csv"
            csv_path.write_text(
                "peptide,HLA,label\n"
                "AAAAAAAA,CCCCCC,0\n"
                "BBBBBBBB,DDDDDD,1\n"
                "EEEEEEEE,FFFFFF,0\n",
                encoding="utf-8",
            )

            full_dataset = OnlinePairDataset(csv_path, "phla", limit=None)
            limited_dataset = OnlinePairDataset(csv_path, "phla", limit=2)

            self.assertEqual(len(full_dataset), 3)
            self.assertEqual(len(limited_dataset), 2)

    def _config(self, root: Path):
        from source.plm_unified.config import (
            ExperimentConfig,
            ModelConfig,
            PathsConfig,
            Stage2Config,
            SwanLabConfig,
            TrainingConfig,
        )

        return ExperimentConfig(
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
                mixed_precision="none",
                pin_memory=False,
                persistent_workers=False,
                device="cpu",
            ),
            stage2=Stage2Config(
                micro_batch_size=2,
                eval_batch_size=4,
                gradient_accumulation_steps=2,
                stage2a_rounds=2,
                stage2b_min_rounds=5,
                stage2b_max_rounds=20,
                early_stopping_patience=5,
                mixed_precision="none",
                gradient_checkpointing=False,
                log_every_optimizer_steps=1,
            ),
            swanlab=SwanLabConfig(mode="disabled"),
        )

    def _model(self, config):
        from source.plm_unified.encoders import ResidueBatch
        from source.plm_unified.finetuning import ProgressiveFineTuningModel
        from source.plm_unified.model import UnifiedBindingModel

        class TinyEncoder(nn.Module):
            def __init__(self, hidden_size: int, max_length: int, name: str):
                super().__init__()
                self.hidden_size = hidden_size
                self.max_length = max_length
                self.checkpoint_path = Path(f"/synthetic/{name}")
                self.checkpoint_fingerprint = f"fingerprint-{name}"
                self.gradient_checkpointing = False
                self.forward_calls = 0
                self.encoded_sequences = 0
                self.frozen_stem = nn.Linear(hidden_size, hidden_size)
                self.blocks = nn.ModuleList(
                    nn.Linear(hidden_size, hidden_size) for _ in range(4)
                )
                self.requires_grad_(False)

            def transformer_blocks(self):
                return self.blocks

            def set_active_training(self, active: bool):
                self.train(active)

            def metadata(self):
                return {
                    "checkpoint_path": str(self.checkpoint_path),
                    "checkpoint_fingerprint": self.checkpoint_fingerprint,
                    "hidden_size": self.hidden_size,
                    "max_length": self.max_length,
                    "encoder_class": type(self).__name__,
                    "transformer_layers": len(self.blocks),
                    "gradient_checkpointing": False,
                }

            def forward(self, sequences):
                self.forward_calls += 1
                self.encoded_sequences += len(sequences)
                batch_size = len(sequences)
                values = torch.zeros(
                    batch_size,
                    self.max_length,
                    self.hidden_size,
                )
                mask = torch.zeros(
                    batch_size,
                    self.max_length,
                    dtype=torch.bool,
                )
                lengths = torch.zeros(batch_size, dtype=torch.long)
                for row, sequence in enumerate(sequences):
                    length = min(len(sequence), self.max_length)
                    values[row, :length, 0] = 1.0
                    mask[row, :length] = True
                    lengths[row] = length
                hidden = torch.tanh(self.frozen_stem(values))
                for block in self.blocks:
                    hidden = torch.tanh(block(hidden))
                return ResidueBatch(hidden=hidden, mask=mask, lengths=lengths)

        return ProgressiveFineTuningModel(
            UnifiedBindingModel(config.model),
            TinyEncoder(12, 5, "peptide"),
            TinyEncoder(12, 6, "hla"),
            TinyEncoder(8, 6, "tcr"),
            peptide_unfrozen_layers=2,
            hla_unfrozen_layers=1,
            tcr_unfrozen_layers=2,
        )

    @staticmethod
    def _loaders():
        labels = torch.tensor([0, 1])
        phla_batch = {
            "peptide_sequences": ["AAAAAAAA", "BBBBBBBB"],
            "receptor_sequences": ["CCCCCC", "DDDDDD"],
            "labels": labels,
        }
        ptcr_batch = {
            "peptide_sequences": ["AAAAAAAA", "BBBBBBBB"],
            "receptor_sequences": ["EEEEEE", "FFFFFF"],
            "labels": labels,
        }
        return {
            "phla": [phla_batch, phla_batch, phla_batch],
            "ptcr": [ptcr_batch, ptcr_batch, ptcr_batch],
        }

    def test_progressive_unfreezing_and_discriminative_learning_rates(self):
        from source.plm_unified.stage2_trainer import build_stage2_optimizer

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            model = self._model(config)

            model.set_trainable_task("phla", "stage2a")
            peptide_blocks = model.peptide_encoder.transformer_blocks()
            self.assertFalse(
                any(parameter.requires_grad for parameter in peptide_blocks[-3].parameters())
            )
            self.assertTrue(
                all(parameter.requires_grad for parameter in peptide_blocks[-2].parameters())
            )
            self.assertFalse(
                any(parameter.requires_grad for parameter in model.hla_encoder.parameters())
            )
            self.assertFalse(
                any(parameter.requires_grad for parameter in model.tcr_encoder.parameters())
            )

            model.set_trainable_task("phla", "stage2b")
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for parameter in model.hla_encoder.transformer_blocks()[-1].parameters()
                )
            )
            self.assertFalse(
                any(parameter.requires_grad for parameter in model.tcr_encoder.parameters())
            )

            model.set_trainable_task("ptcr", "stage2b")
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for block in model.tcr_encoder.transformer_blocks()[-2:]
                    for parameter in block.parameters()
                )
            )
            self.assertFalse(
                any(parameter.requires_grad for parameter in model.hla_encoder.parameters())
            )

            stage2a = build_stage2_optimizer(model, config, "stage2a")
            rates2a = {
                group["name"]: group["lr"] for group in stage2a.param_groups
            }
            self.assertEqual(rates2a["peptide_plm_last"], 1e-5)
            self.assertEqual(rates2a["peptide_plm_lower"], 5e-6)
            self.assertNotIn("hla_plm_top", rates2a)
            self.assertNotIn("tcr_plm_last", rates2a)

            stage2b = build_stage2_optimizer(model, config, "stage2b")
            rates2b = {
                group["name"]: group["lr"] for group in stage2b.param_groups
            }
            self.assertEqual(rates2b["hla_plm_top"], 1e-5)
            self.assertEqual(rates2b["tcr_plm_last"], 1e-5)
            self.assertEqual(rates2b["tcr_plm_lower"], 5e-6)
            self.assertEqual(rates2b["adapters"], 5e-5)
            self.assertEqual(rates2b["cross_attention"], 1e-4)

    def test_stage2a_caches_frozen_hla_and_stage2b_clears_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._model(self._config(Path(temporary)))
            peptides = ["AAAAAAAA", "BBBBBBBB", "CCCCCCCC"]
            hlas = ["AAAAAA", "BBBBBB", "AAAAAA"]

            model.set_trainable_task("phla", "stage2a")
            model("phla", peptides, hlas)
            model("phla", peptides, hlas)

            self.assertEqual(model.hla_encoder.forward_calls, 1)
            self.assertEqual(model.hla_encoder.encoded_sequences, 2)
            self.assertEqual(
                model.stage2a_hla_cache_stats(),
                {"entries": 2, "requests": 6, "hits": 4, "misses": 2},
            )

            model.set_trainable_task("phla", "stage2b")
            self.assertEqual(model.stage2a_hla_cache_stats()["entries"], 0)
            model("phla", peptides, hlas)
            self.assertEqual(model.hla_encoder.forward_calls, 2)

    def test_gradient_accumulation_steps_optimizer_twice_for_three_batches(self):
        from source.plm_unified.stage2_trainer import Stage2Trainer
        from source.plm_unified.tracking import SwanLabTracker

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            model = self._model(config)
            loaders = self._loaders()
            trainer = Stage2Trainer(
                model,
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                run_timestamp="20260726_120000_000001",
            )
            console = StringIO()
            with redirect_stdout(console):
                result = trainer.train_epoch(
                    "phla",
                    loaders["phla"],
                    stage="stage2a",
                    round_id=1,
                )
            self.assertEqual(result.optimizer_steps, 2)
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(result.samples, 6)
            self.assertEqual(result.adversarial_loss, 0.0)
            output = console.getvalue()
            self.assertIn("[TRAIN START][STAGE2A][PHLA]", output)
            self.assertIn("[TRAIN PROGRESS][STAGE2A][PHLA]", output)
            self.assertIn("avg_loss=", output)
            self.assertIn("grad_norm=", output)
            self.assertIn("eta=", output)
            self.assertIn("lr=(", output)
            self.assertIn("[TRAIN END][STAGE2A][PHLA]", output)

    def test_best_stage2a_is_reloaded_and_stage2b_stops_after_five_misses(self):
        from source.plm_unified.metrics import BinaryMetrics
        from source.plm_unified.stage2_trainer import (
            Stage2Trainer,
            load_stage2_checkpoint,
        )
        from source.plm_unified.tracking import SwanLabTracker
        from source.plm_unified.trainer import EpochResult

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

        def epoch_result(score):
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
            config = self._config(Path(temporary))
            model = self._model(config)
            loaders = self._loaders()
            trainer = Stage2Trainer(
                model,
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                source_stage1_checkpoint=Path(temporary) / "stage1_last.pt",
                run_timestamp="20260726_120000_000001",
            )
            stage2b_start_values = []

            def fake_run_round(self, stage, round_id):
                tracked_parameter = (
                    self.model.binding_model.phla_classifier.network[-1].weight
                )
                if stage == "stage2a":
                    value = float(round_id)
                    score = 0.8 if round_id == 1 else 0.7
                else:
                    if round_id == 1:
                        stage2b_start_values.append(
                            float(tracked_parameter.flatten()[0])
                        )
                    value = float(100 + round_id)
                    score = 0.9 if round_id == 1 else 0.8
                with torch.no_grad():
                    tracked_parameter.fill_(value)
                result = {
                    "phla": epoch_result(score),
                    "ptcr": epoch_result(score),
                }
                validation = {
                    "phla": metrics(score),
                    "ptcr": metrics(score),
                }
                return result, validation, score

            trainer._run_round = MethodType(fake_run_round, trainer)
            summary = trainer.fit()
            self.assertEqual(summary["stage"], "complete")
            self.assertEqual(summary["stage2a_round"], 2)
            self.assertEqual(summary["stage2b_round"], 6)
            self.assertEqual(stage2b_start_values, [1.0])
            self.assertFalse(trainer.checkpoint_state()["fgm_enabled"])

            phla_task = torch.load(
                trainer.task_checkpoint_paths["phla"],
                map_location="cpu",
                weights_only=False,
            )
            ptcr_task = torch.load(
                trainer.task_checkpoint_paths["ptcr"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(phla_task["checkpoint_type"], "stage2_task_model")
            self.assertEqual(phla_task["task"], "phla")
            self.assertEqual(ptcr_task["task"], "ptcr")
            self.assertEqual(phla_task["receptor_entity"], "hla")
            self.assertEqual(ptcr_task["receptor_entity"], "tcr")
            self.assertIn("frozen_stem.weight", phla_task["model"]["peptide_encoder"])
            self.assertIn("frozen_stem.weight", phla_task["model"]["receptor_encoder"])
            self.assertTrue(
                all(
                    not name.startswith("ptcr_")
                    for name in phla_task["model"]["binding_model"]
                )
            )
            self.assertTrue(
                all(
                    not name.startswith("phla_")
                    for name in ptcr_task["model"]["binding_model"]
                )
            )

            best2a = load_stage2_checkpoint(
                trainer.stage2a_best_path,
                map_location="cpu",
            )
            self.assertIn(
                "peptide_encoder.frozen_stem.weight",
                best2a["model"],
            )
            self.assertIn(
                "hla_encoder.frozen_stem.weight",
                best2a["model"],
            )
            self.assertIn(
                "tcr_encoder.frozen_stem.weight",
                best2a["model"],
            )

            resumed = Stage2Trainer(
                self._model(config),
                config,
                loaders,
                loaders,
                SwanLabTracker(config),
                run_timestamp="20260726_120000_000001",
            )
            resumed.load_checkpoint(trainer.last_checkpoint_path)
            self.assertEqual(resumed.stage, "complete")
            self.assertEqual(resumed.stage2b_round, 6)
            self.assertEqual(
                resumed.source_stage1_checkpoint,
                str((Path(temporary) / "stage1_last.pt").resolve()),
            )


if __name__ == "__main__":
    unittest.main()
