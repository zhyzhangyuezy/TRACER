from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import _moving_average, build_encoder, summarize_prefix


class ParametricForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            encoder_type=encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
        self.main_head = nn.Linear(embedding_dim, 1)
        self.aux_head = nn.Linear(embedding_dim, 1)

    def encode(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.encoder(prefix)

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode(prefix)
        main_logit = self.main_head(embedding).squeeze(-1)
        aux_logit = self.aux_head(embedding).squeeze(-1)
        return {
            "embedding": embedding,
            "main_logit": main_logit,
            "aux_logit": aux_logit,
            "final_main_logit": main_logit,
            "final_aux_logit": aux_logit,
        }


class RetrievalForecaster(ParametricForecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_weighted: bool = False,
        retrieval_aware_gate: bool = False,
        use_residual_fusion: bool = False,
        similarity_temperature: float = 0.2,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.top_k = top_k
        self.similarity_weighted = similarity_weighted
        self.retrieval_aware_gate = retrieval_aware_gate
        self.use_residual_fusion = use_residual_fusion
        self.similarity_temperature = max(similarity_temperature, 1e-3)
        self.residual_scale = residual_scale

        gate_feature_dim = embedding_dim
        if retrieval_aware_gate:
            gate_feature_dim += embedding_dim + 6
            self.gate_head = nn.Sequential(
                nn.Linear(gate_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.gate_head = nn.Linear(gate_feature_dim, 1)

        if use_residual_fusion:
            fusion_feature_dim = embedding_dim * 2 + 7
            self.main_residual_head = nn.Sequential(
                nn.Linear(fusion_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_residual_head = nn.Sequential(
                nn.Linear(fusion_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def encode_memory(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.encode(prefix)

    def _neighbor_weights(self, retrieved_scores: torch.Tensor) -> torch.Tensor:
        if self.similarity_weighted:
            return F.softmax(retrieved_scores / self.similarity_temperature, dim=-1)
        return torch.full_like(retrieved_scores, 1.0 / max(retrieved_scores.shape[-1], 1))

    def _aggregate_neighbors(
        self,
        query_embedding: torch.Tensor,
        memory_embedding: torch.Tensor,
        memory_main_label: torch.Tensor,
        memory_aux_label: torch.Tensor,
        exclude_self: bool = False,
    ) -> dict[str, torch.Tensor]:
        query_norm = F.normalize(query_embedding, dim=-1)
        memory_norm = F.normalize(memory_embedding, dim=-1)
        similarity = query_norm @ memory_norm.T
        if exclude_self and similarity.shape[0] == similarity.shape[1]:
            diagonal = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
            similarity = similarity.masked_fill(diagonal, float("-inf"))
        k = min(self.top_k, memory_embedding.shape[0] - (1 if exclude_self else 0))
        if k <= 0:
            raise ValueError("Retrieval requires at least one memory example.")
        retrieved_scores, retrieved_indices = torch.topk(similarity, k=k, dim=-1)
        weights = self._neighbor_weights(retrieved_scores)
        neighbor_main_values = memory_main_label[retrieved_indices].float()
        neighbor_aux_values = memory_aux_label[retrieved_indices].float()
        neighbor_main_prob = (neighbor_main_values * weights).sum(dim=-1)
        neighbor_aux_prob = (neighbor_aux_values * weights).sum(dim=-1)
        retrieved_main_logit = torch.logit(neighbor_main_prob.clamp(1e-4, 1 - 1e-4))
        retrieved_aux_logit = torch.logit(neighbor_aux_prob.clamp(1e-4, 1 - 1e-4))
        neighbor_embeddings = memory_embedding[retrieved_indices]
        retrieved_context = torch.sum(neighbor_embeddings * weights.unsqueeze(-1), dim=1)

        score_mean = (retrieved_scores * weights).sum(dim=-1)
        score_delta = retrieved_scores - score_mean.unsqueeze(-1)
        score_std = torch.sqrt((weights * score_delta.square()).sum(dim=-1).clamp_min(1e-6))
        score_max = retrieved_scores.max(dim=-1).values
        main_dispersion = torch.sqrt(
            (weights * (neighbor_main_values - neighbor_main_prob.unsqueeze(-1)).square()).sum(dim=-1).clamp_min(1e-6)
        )
        aux_dispersion = torch.sqrt(
            (weights * (neighbor_aux_values - neighbor_aux_prob.unsqueeze(-1)).square()).sum(dim=-1).clamp_min(1e-6)
        )
        return {
            "retrieved_scores": retrieved_scores,
            "retrieved_indices": retrieved_indices,
            "retrieved_weights": weights,
            "retrieved_main_logit": retrieved_main_logit,
            "retrieved_aux_logit": retrieved_aux_logit,
            "retrieved_context": retrieved_context,
            "retrieved_main_prob": neighbor_main_prob,
            "retrieved_aux_prob": neighbor_aux_prob,
            "retrieval_score_mean": score_mean,
            "retrieval_score_std": score_std,
            "retrieval_score_max": score_max,
            "retrieval_main_dispersion": main_dispersion,
            "retrieval_aux_dispersion": aux_dispersion,
        }

    def _build_gate_features(
        self,
        embedding: torch.Tensor,
        main_logit: torch.Tensor,
        aux_logit: torch.Tensor,
        retrieved: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.retrieval_aware_gate:
            return embedding
        agreement_main = 1.0 - torch.abs(torch.sigmoid(main_logit) - retrieved["retrieved_main_prob"])
        agreement_aux = 1.0 - torch.abs(torch.sigmoid(aux_logit) - retrieved["retrieved_aux_prob"])
        stats = torch.stack(
            [
                retrieved["retrieval_score_mean"],
                retrieved["retrieval_score_std"],
                retrieved["retrieval_score_max"],
                agreement_main,
                agreement_aux,
                retrieved["retrieval_main_dispersion"],
            ],
            dim=-1,
        )
        return torch.cat([embedding, retrieved["retrieved_context"], stats], dim=-1)

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        gate_features = self._build_gate_features(
            embedding=outputs["embedding"],
            main_logit=outputs["main_logit"],
            aux_logit=outputs["aux_logit"],
            retrieved=retrieved,
        )
        gate = torch.sigmoid(self.gate_head(gate_features).squeeze(-1))
        outputs.update(retrieved)
        outputs["gate"] = gate
        outputs["final_main_logit"] = gate * outputs["main_logit"] + (1 - gate) * retrieved["retrieved_main_logit"]
        outputs["final_aux_logit"] = gate * outputs["aux_logit"] + (1 - gate) * retrieved["retrieved_aux_logit"]

        if self.use_residual_fusion:
            fusion_stats = torch.stack(
                [
                    outputs["main_logit"],
                    retrieved["retrieved_main_logit"],
                    outputs["main_logit"] - retrieved["retrieved_main_logit"],
                    retrieved["retrieval_score_mean"],
                    retrieved["retrieval_score_std"],
                    retrieved["retrieval_main_dispersion"],
                    gate,
                ],
                dim=-1,
            )
            main_fusion_input = torch.cat([outputs["embedding"], retrieved["retrieved_context"], fusion_stats], dim=-1)
            aux_fusion_stats = torch.stack(
                [
                    outputs["aux_logit"],
                    retrieved["retrieved_aux_logit"],
                    outputs["aux_logit"] - retrieved["retrieved_aux_logit"],
                    retrieved["retrieval_score_mean"],
                    retrieved["retrieval_score_std"],
                    retrieved["retrieval_aux_dispersion"],
                    gate,
                ],
                dim=-1,
            )
            aux_fusion_input = torch.cat([outputs["embedding"], retrieved["retrieved_context"], aux_fusion_stats], dim=-1)
            outputs["final_main_logit"] = outputs["final_main_logit"] + self.residual_scale * self.main_residual_head(main_fusion_input).squeeze(-1)
            outputs["final_aux_logit"] = outputs["final_aux_logit"] + self.residual_scale * self.aux_residual_head(aux_fusion_input).squeeze(-1)
        return outputs

    def forward_with_batch_memory(
        self,
        prefix: torch.Tensor,
        label_main: torch.Tensor,
        label_aux: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(prefix)
        retrieved = self._aggregate_neighbors(
            query_embedding=outputs["embedding"],
            memory_embedding=outputs["embedding"],
            memory_main_label=label_main,
            memory_aux_label=label_aux,
            exclude_self=True,
        )
        return self._apply_fusion(outputs, retrieved)

    def forward_with_external_memory(
        self,
        prefix: torch.Tensor,
        memory_embedding: torch.Tensor,
        memory_main_label: torch.Tensor,
        memory_aux_label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(prefix)
        retrieved = self._aggregate_neighbors(
            query_embedding=outputs["embedding"],
            memory_embedding=memory_embedding,
            memory_main_label=memory_main_label,
            memory_aux_label=memory_aux_label,
            exclude_self=False,
        )
        return self._apply_fusion(outputs, retrieved)


class RandomRetrievalForecaster(RetrievalForecaster):
    def _aggregate_neighbors(
        self,
        query_embedding: torch.Tensor,
        memory_embedding: torch.Tensor,
        memory_main_label: torch.Tensor,
        memory_aux_label: torch.Tensor,
        exclude_self: bool = False,
    ) -> dict[str, torch.Tensor]:
        query_norm = F.normalize(query_embedding, dim=-1)
        memory_norm = F.normalize(memory_embedding, dim=-1)
        similarity = query_norm @ memory_norm.T
        k = min(self.top_k, memory_embedding.shape[0] - (1 if exclude_self else 0))
        if k <= 0:
            raise ValueError("Retrieval requires at least one memory example.")

        noise = torch.rand_like(similarity)
        if exclude_self and similarity.shape[0] == similarity.shape[1]:
            diagonal = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
            noise = noise.masked_fill(diagonal, float("-inf"))
        _, retrieved_indices = torch.topk(noise, k=k, dim=-1)
        retrieved_scores = similarity.gather(1, retrieved_indices)
        neighbor_main_prob = memory_main_label[retrieved_indices].float().mean(dim=-1)
        neighbor_aux_prob = memory_aux_label[retrieved_indices].float().mean(dim=-1)
        retrieved_main_logit = torch.logit(neighbor_main_prob.clamp(1e-4, 1 - 1e-4))
        retrieved_aux_logit = torch.logit(neighbor_aux_prob.clamp(1e-4, 1 - 1e-4))
        return {
            "retrieved_scores": retrieved_scores,
            "retrieved_indices": retrieved_indices,
            "retrieved_main_logit": retrieved_main_logit,
            "retrieved_aux_logit": retrieved_aux_logit,
        }


class CampaignMemV2Forecaster(RetrievalForecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_encoder_type: str,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=forecast_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_weighted=True,
            retrieval_aware_gate=False,
            use_residual_fusion=False,
            similarity_temperature=similarity_temperature,
            residual_scale=residual_scale,
        )
        self.forecast_encoder = self.encoder
        self.retrieval_encoder = build_encoder(
            encoder_type=retrieval_encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.main_calibration_head = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_calibration_head = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.forecast_encoder(prefix)

    def encode_memory(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.retrieval_encoder(prefix)

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        forecast_embedding = self.forecast_encoder(prefix)
        retrieval_embedding = self.retrieval_encoder(prefix)
        main_logit = self.main_head(forecast_embedding).squeeze(-1)
        aux_logit = self.aux_head(forecast_embedding).squeeze(-1)
        return {
            "forecast_embedding": forecast_embedding,
            "retrieval_embedding": retrieval_embedding,
            "embedding": retrieval_embedding,
            "main_logit": main_logit,
            "aux_logit": aux_logit,
            "final_main_logit": main_logit,
            "final_aux_logit": aux_logit,
        }

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        forecast_embedding = outputs["forecast_embedding"]
        retrieval_embedding = outputs["retrieval_embedding"]
        main_prob = torch.sigmoid(outputs["main_logit"])
        aux_prob = torch.sigmoid(outputs["aux_logit"])
        retrieval_gap = torch.abs(main_prob - retrieved["retrieved_main_prob"])
        aux_gap = torch.abs(aux_prob - retrieved["retrieved_aux_prob"])
        reliability_features = torch.cat(
            [
                forecast_embedding,
                retrieval_embedding,
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieval_gap,
                        aux_gap,
                        retrieved["retrieval_main_dispersion"],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_features).squeeze(-1))

        main_delta = retrieved["retrieved_main_logit"] - outputs["main_logit"]
        aux_delta = retrieved["retrieved_aux_logit"] - outputs["aux_logit"]
        main_features = torch.cat(
            [
                forecast_embedding,
                retrieval_embedding,
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        outputs["main_logit"],
                        retrieved["retrieved_main_logit"],
                        main_delta,
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_main_dispersion"],
                        reliability,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                forecast_embedding,
                retrieval_embedding,
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        outputs["aux_logit"],
                        retrieved["retrieved_aux_logit"],
                        aux_delta,
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_aux_dispersion"],
                        reliability,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        outputs.update(retrieved)
        outputs["gate"] = reliability
        outputs["final_main_logit"] = (
            outputs["main_logit"]
            + reliability * main_delta
            + self.residual_scale * self.main_calibration_head(main_features).squeeze(-1)
        )
        outputs["final_aux_logit"] = (
            outputs["aux_logit"]
            + reliability * aux_delta
            + self.residual_scale * self.aux_calibration_head(aux_features).squeeze(-1)
        )
        return outputs


class CampaignMemV3Forecaster(RetrievalForecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_encoder_type: str,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.35,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_weighted=True,
            retrieval_aware_gate=False,
            use_residual_fusion=False,
            similarity_temperature=similarity_temperature,
            residual_scale=0.0,
        )
        self.forecast_encoder = build_encoder(
            encoder_type=forecast_encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
        self.forecast_main_head = nn.Linear(embedding_dim, 1)
        self.forecast_aux_head = nn.Linear(embedding_dim, 1)
        self.delta_scale = delta_scale
        correction_feature_dim = embedding_dim * 2 + 6
        self.main_correction_head = nn.Sequential(
            nn.Linear(correction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_correction_head = nn.Sequential(
            nn.Linear(correction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_memory(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.encode(prefix)

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        retrieval_embedding = self.encode(prefix)
        forecast_embedding = self.forecast_encoder(prefix)
        retrieval_main_logit = self.main_head(retrieval_embedding).squeeze(-1)
        retrieval_aux_logit = self.aux_head(retrieval_embedding).squeeze(-1)
        forecast_main_logit = self.forecast_main_head(forecast_embedding).squeeze(-1)
        forecast_aux_logit = self.forecast_aux_head(forecast_embedding).squeeze(-1)
        return {
            "embedding": retrieval_embedding,
            "retrieval_embedding": retrieval_embedding,
            "forecast_embedding": forecast_embedding,
            "main_logit": retrieval_main_logit,
            "aux_logit": retrieval_aux_logit,
            "forecast_main_logit": forecast_main_logit,
            "forecast_aux_logit": forecast_aux_logit,
            "final_main_logit": retrieval_main_logit,
            "final_aux_logit": retrieval_aux_logit,
        }

    def _bounded_delta(
        self,
        base_logit: torch.Tensor,
        forecast_logit: torch.Tensor,
        correction_gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.delta_scale * correction_gate * torch.tanh(forecast_logit - base_logit)

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        forecast_main_prob = torch.sigmoid(outputs["forecast_main_logit"])
        forecast_aux_prob = torch.sigmoid(outputs["forecast_aux_logit"])
        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        forecast_conf = torch.abs(forecast_main_prob - 0.5) * 2.0
        aux_conf = torch.abs(forecast_aux_prob - 0.5) * 2.0

        main_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_main_dispersion"],
                        torch.abs(forecast_main_prob - base_main_prob),
                        forecast_conf,
                        base_gate,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_aux_dispersion"],
                        torch.abs(forecast_aux_prob - base_aux_prob),
                        aux_conf,
                        base_gate,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        main_correction_gate = torch.sigmoid(self.main_correction_head(main_features).squeeze(-1))
        aux_correction_gate = torch.sigmoid(self.aux_correction_head(aux_features).squeeze(-1))
        main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit"],
            correction_gate=main_correction_gate,
        )
        aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit"],
            correction_gate=aux_correction_gate,
        )

        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (main_delta.abs().mean() + aux_delta.abs().mean())
        return outputs


class CampaignMemV4Forecaster(RetrievalForecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_encoder_type: str,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.25,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_weighted=True,
            retrieval_aware_gate=False,
            use_residual_fusion=False,
            similarity_temperature=similarity_temperature,
            residual_scale=0.0,
        )
        self.forecast_encoder = build_encoder(
            encoder_type=forecast_encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
        self.forecast_main_head = nn.Linear(embedding_dim, 1)
        self.forecast_aux_head = nn.Linear(embedding_dim, 1)
        self.delta_scale = delta_scale

        base_feature_dim = embedding_dim * 2 + 8
        correction_feature_dim = embedding_dim * 3 + 11
        self.base_gate_head = nn.Sequential(
            nn.Linear(base_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.main_correction_head = nn.Sequential(
            nn.Linear(correction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_correction_head = nn.Sequential(
            nn.Linear(correction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_memory(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.encode(prefix)

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        retrieval_embedding = self.encode(prefix)
        forecast_embedding = self.forecast_encoder(prefix)
        retrieval_main_logit = self.main_head(retrieval_embedding).squeeze(-1)
        retrieval_aux_logit = self.aux_head(retrieval_embedding).squeeze(-1)
        forecast_main_logit = self.forecast_main_head(forecast_embedding).squeeze(-1)
        forecast_aux_logit = self.forecast_aux_head(forecast_embedding).squeeze(-1)
        return {
            "embedding": retrieval_embedding,
            "retrieval_embedding": retrieval_embedding,
            "forecast_embedding": forecast_embedding,
            "main_logit": retrieval_main_logit,
            "aux_logit": retrieval_aux_logit,
            "forecast_main_logit": forecast_main_logit,
            "forecast_aux_logit": forecast_aux_logit,
            "final_main_logit": retrieval_main_logit,
            "final_aux_logit": retrieval_aux_logit,
        }

    def _bounded_delta(
        self,
        base_logit: torch.Tensor,
        forecast_logit: torch.Tensor,
        correction_gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.delta_scale * correction_gate * torch.tanh(forecast_logit - base_logit)

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        retrieval_main_prob = torch.sigmoid(outputs["main_logit"])
        retrieval_aux_prob = torch.sigmoid(outputs["aux_logit"])
        neighbor_main_prob = retrieved["retrieved_main_prob"]
        neighbor_aux_prob = retrieved["retrieved_aux_prob"]
        forecast_main_prob = torch.sigmoid(outputs["forecast_main_logit"])
        forecast_aux_prob = torch.sigmoid(outputs["forecast_aux_logit"])

        main_self_neighbor_gap = torch.abs(retrieval_main_prob - neighbor_main_prob)
        aux_self_neighbor_gap = torch.abs(retrieval_aux_prob - neighbor_aux_prob)
        agreement_score = 1.0 - 0.5 * (main_self_neighbor_gap + aux_self_neighbor_gap)
        base_features = torch.cat(
            [
                outputs["retrieval_embedding"],
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        main_self_neighbor_gap,
                        aux_self_neighbor_gap,
                        agreement_score,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        base_gate = torch.sigmoid(self.base_gate_head(base_features).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        forecast_conf = torch.abs(forecast_main_prob - 0.5) * 2.0
        base_conf = torch.abs(base_main_prob - 0.5) * 2.0
        aux_forecast_conf = torch.abs(forecast_aux_prob - 0.5) * 2.0
        aux_base_conf = torch.abs(base_aux_prob - 0.5) * 2.0

        main_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        torch.abs(forecast_main_prob - base_main_prob),
                        torch.abs(forecast_main_prob - neighbor_main_prob),
                        forecast_conf,
                        base_conf,
                        base_gate,
                        agreement_score,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        torch.abs(forecast_aux_prob - base_aux_prob),
                        torch.abs(forecast_aux_prob - neighbor_aux_prob),
                        aux_forecast_conf,
                        aux_base_conf,
                        base_gate,
                        agreement_score,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        main_correction_gate = torch.sigmoid(self.main_correction_head(main_features).squeeze(-1))
        aux_correction_gate = torch.sigmoid(self.aux_correction_head(aux_features).squeeze(-1))
        main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit"],
            correction_gate=main_correction_gate,
        )
        aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit"],
            correction_gate=aux_correction_gate,
        )

        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (main_delta.abs().mean() + aux_delta.abs().mean())
        return outputs


class CampaignMemV5Forecaster(CampaignMemV3Forecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_encoder_type: str,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.25,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=forecast_encoder_type,
            retrieval_encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_temperature=similarity_temperature,
            delta_scale=delta_scale,
        )
        correction_feature_dim = embedding_dim * 3 + 11
        self.main_correction_head = nn.Sequential(
            nn.Linear(correction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_correction_head = nn.Sequential(
            nn.Linear(correction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        forecast_main_prob = torch.sigmoid(outputs["forecast_main_logit"])
        forecast_aux_prob = torch.sigmoid(outputs["forecast_aux_logit"])
        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        neighbor_main_prob = retrieved["retrieved_main_prob"]
        neighbor_aux_prob = retrieved["retrieved_aux_prob"]
        forecast_conf = torch.abs(forecast_main_prob - 0.5) * 2.0
        base_conf = torch.abs(base_main_prob - 0.5) * 2.0
        aux_forecast_conf = torch.abs(forecast_aux_prob - 0.5) * 2.0
        aux_base_conf = torch.abs(base_aux_prob - 0.5) * 2.0
        agreement_score = 1.0 - 0.5 * (
            torch.abs(torch.sigmoid(outputs["main_logit"]) - neighbor_main_prob)
            + torch.abs(torch.sigmoid(outputs["aux_logit"]) - neighbor_aux_prob)
        )

        main_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        torch.abs(forecast_main_prob - base_main_prob),
                        torch.abs(forecast_main_prob - neighbor_main_prob),
                        forecast_conf,
                        base_conf,
                        base_gate,
                        agreement_score,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        torch.abs(forecast_aux_prob - base_aux_prob),
                        torch.abs(forecast_aux_prob - neighbor_aux_prob),
                        aux_forecast_conf,
                        aux_base_conf,
                        base_gate,
                        agreement_score,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        main_correction_gate = torch.sigmoid(self.main_correction_head(main_features).squeeze(-1))
        aux_correction_gate = torch.sigmoid(self.aux_correction_head(aux_features).squeeze(-1))
        main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit"],
            correction_gate=main_correction_gate,
        )
        aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit"],
            correction_gate=aux_correction_gate,
        )

        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (main_delta.abs().mean() + aux_delta.abs().mean())
        return outputs


class CampaignMemStructuredCalibrator(CampaignMemV3Forecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_encoder_type: str,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.25,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=forecast_encoder_type,
            retrieval_encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_temperature=similarity_temperature,
            delta_scale=delta_scale,
        )
        reliability_feature_dim = embedding_dim * 2 + 7
        correction_feature_dim = embedding_dim * 2 + 9
        self.retrieval_reliability_head = nn.Sequential(
            nn.Linear(reliability_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.main_correction_head = nn.Sequential(
            nn.Linear(correction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_correction_head = nn.Sequential(
            nn.Linear(correction_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        retrieval_main_prob = torch.sigmoid(outputs["main_logit"])
        retrieval_aux_prob = torch.sigmoid(outputs["aux_logit"])
        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        forecast_main_prob = torch.sigmoid(outputs["forecast_main_logit"])
        forecast_aux_prob = torch.sigmoid(outputs["forecast_aux_logit"])

        retrieval_self_neighbor_gap = torch.abs(retrieval_main_prob - retrieved["retrieved_main_prob"])
        retrieval_aux_neighbor_gap = torch.abs(retrieval_aux_prob - retrieved["retrieved_aux_prob"])
        reliability_features = torch.cat(
            [
                outputs["retrieval_embedding"],
                retrieved["retrieved_context"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        retrieval_self_neighbor_gap,
                        retrieval_aux_neighbor_gap,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        retrieval_reliability = torch.sigmoid(self.retrieval_reliability_head(reliability_features).squeeze(-1))

        forecast_main_conf = torch.abs(forecast_main_prob - 0.5) * 2.0
        forecast_aux_conf = torch.abs(forecast_aux_prob - 0.5) * 2.0
        main_disagreement = torch.abs(forecast_main_prob - base_main_prob)
        aux_disagreement = torch.abs(forecast_aux_prob - base_aux_prob)

        main_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_main_dispersion"],
                        main_disagreement,
                        forecast_main_conf,
                        retrieval_reliability,
                        base_gate,
                        retrieval_self_neighbor_gap,
                        torch.abs(base_main_prob - retrieved["retrieved_main_prob"]),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_aux_dispersion"],
                        aux_disagreement,
                        forecast_aux_conf,
                        retrieval_reliability,
                        base_gate,
                        retrieval_aux_neighbor_gap,
                        torch.abs(base_aux_prob - retrieved["retrieved_aux_prob"]),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        raw_main_gate = torch.sigmoid(self.main_correction_head(main_features).squeeze(-1))
        raw_aux_gate = torch.sigmoid(self.aux_correction_head(aux_features).squeeze(-1))
        main_budget = (1.0 - retrieval_reliability) * (0.5 * forecast_main_conf + 0.5 * main_disagreement)
        aux_budget = (1.0 - retrieval_reliability) * (0.5 * forecast_aux_conf + 0.5 * aux_disagreement)
        main_correction_gate = raw_main_gate * main_budget
        aux_correction_gate = raw_aux_gate * aux_budget

        main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit"],
            correction_gate=main_correction_gate,
        )
        aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit"],
            correction_gate=aux_correction_gate,
        )

        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["retrieval_reliability"] = retrieval_reliability
        outputs["calibration_gate"] = main_correction_gate
        outputs["raw_calibration_gate"] = raw_main_gate
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (
            (1.0 + retrieval_reliability) * (main_delta.abs() + aux_delta.abs())
        ).mean()
        return outputs


class CampaignMemSelectorCalibrator(CampaignMemV3Forecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_encoder_type: str,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.25,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=forecast_encoder_type,
            retrieval_encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_temperature=similarity_temperature,
            delta_scale=delta_scale,
        )
        selector_feature_dim = embedding_dim * 2 + 8
        self.main_selector_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_selector_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        forecast_main_prob = torch.sigmoid(outputs["forecast_main_logit"])
        forecast_aux_prob = torch.sigmoid(outputs["forecast_aux_logit"])
        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        forecast_main_conf = torch.abs(forecast_main_prob - 0.5) * 2.0
        forecast_aux_conf = torch.abs(forecast_aux_prob - 0.5) * 2.0
        main_disagreement = torch.abs(forecast_main_prob - base_main_prob)
        aux_disagreement = torch.abs(forecast_aux_prob - base_aux_prob)

        main_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        forecast_main_conf,
                        main_disagreement,
                        base_gate,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        forecast_aux_conf,
                        aux_disagreement,
                        base_gate,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        selector_main_logit = self.main_selector_head(main_features).squeeze(-1)
        selector_aux_logit = self.aux_selector_head(aux_features).squeeze(-1)
        main_correction_gate = torch.sigmoid(selector_main_logit)
        aux_correction_gate = torch.sigmoid(selector_aux_logit)

        main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit"],
            correction_gate=main_correction_gate,
        )
        aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit"],
            correction_gate=aux_correction_gate,
        )

        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["selector_main_logit"] = selector_main_logit
        outputs["selector_aux_logit"] = selector_aux_logit
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (main_delta.abs().mean() + aux_delta.abs().mean())
        return outputs


class CampaignMemDualSelectorCalibrator(RetrievalForecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_weighted=True,
            retrieval_aware_gate=False,
            use_residual_fusion=False,
            similarity_temperature=similarity_temperature,
            residual_scale=0.0,
        )
        self.forecast_encoder = nn.ModuleDict(
            {
                "dlinear": build_encoder(
                    encoder_type="dlinear",
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    seq_len=seq_len,
                ),
                "tcn": build_encoder(
                    encoder_type="tcn",
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    seq_len=seq_len,
                ),
            }
        )
        self.forecast_main_head = nn.ModuleDict(
            {
                "dlinear": nn.Linear(embedding_dim, 1),
                "tcn": nn.Linear(embedding_dim, 1),
            }
        )
        self.forecast_aux_head = nn.ModuleDict(
            {
                "dlinear": nn.Linear(embedding_dim, 1),
                "tcn": nn.Linear(embedding_dim, 1),
            }
        )
        self.delta_scale = delta_scale

        blend_feature_dim = embedding_dim * 3 + 8
        selector_feature_dim = embedding_dim * 3 + 10
        self.main_blend_head = nn.Sequential(
            nn.Linear(blend_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_blend_head = nn.Sequential(
            nn.Linear(blend_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.main_selector_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_selector_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        for module in (self.main_blend_head, self.aux_blend_head):
            final_linear = module[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.constant_(final_linear.bias, -1.0)
        for module in (self.main_selector_head, self.aux_selector_head):
            final_linear = module[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.constant_(final_linear.bias, -1.25)

    def encode_memory(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.encode(prefix)

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        retrieval_embedding = self.encode(prefix)
        dlinear_embedding = self.forecast_encoder["dlinear"](prefix)
        tcn_embedding = self.forecast_encoder["tcn"](prefix)
        outputs = {
            "embedding": retrieval_embedding,
            "retrieval_embedding": retrieval_embedding,
            "forecast_embedding_dlinear": dlinear_embedding,
            "forecast_embedding_tcn": tcn_embedding,
            "main_logit": self.main_head(retrieval_embedding).squeeze(-1),
            "aux_logit": self.aux_head(retrieval_embedding).squeeze(-1),
            "forecast_main_logit_dlinear": self.forecast_main_head["dlinear"](dlinear_embedding).squeeze(-1),
            "forecast_aux_logit_dlinear": self.forecast_aux_head["dlinear"](dlinear_embedding).squeeze(-1),
            "forecast_main_logit_tcn": self.forecast_main_head["tcn"](tcn_embedding).squeeze(-1),
            "forecast_aux_logit_tcn": self.forecast_aux_head["tcn"](tcn_embedding).squeeze(-1),
        }
        outputs["final_main_logit"] = outputs["main_logit"]
        outputs["final_aux_logit"] = outputs["aux_logit"]
        return outputs

    def _bounded_delta(
        self,
        base_logit: torch.Tensor,
        forecast_logit: torch.Tensor,
        correction_gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.delta_scale * correction_gate * torch.tanh(forecast_logit - base_logit)

    def _blend_forecasts(
        self,
        *,
        dlinear_embedding: torch.Tensor,
        tcn_embedding: torch.Tensor,
        retrieval_embedding: torch.Tensor,
        dlinear_logit: torch.Tensor,
        tcn_logit: torch.Tensor,
        base_prob: torch.Tensor,
        retrieved: dict[str, torch.Tensor],
        dispersion: torch.Tensor,
        blend_head: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dlinear_prob = torch.sigmoid(dlinear_logit)
        tcn_prob = torch.sigmoid(tcn_logit)
        dlinear_conf = torch.abs(dlinear_prob - 0.5) * 2.0
        tcn_conf = torch.abs(tcn_prob - 0.5) * 2.0
        disagreement = torch.abs(dlinear_prob - tcn_prob)
        blend_features = torch.cat(
            [
                dlinear_embedding,
                tcn_embedding,
                retrieval_embedding,
                torch.stack(
                    [
                        dlinear_conf,
                        tcn_conf,
                        disagreement,
                        torch.abs(dlinear_prob - base_prob),
                        torch.abs(tcn_prob - base_prob),
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        dispersion,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        tcn_weight = torch.sigmoid(blend_head(blend_features).squeeze(-1))
        mixed_logit = (1.0 - tcn_weight) * dlinear_logit + tcn_weight * tcn_logit
        return mixed_logit, tcn_weight

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        mixed_main_logit, tcn_main_weight = self._blend_forecasts(
            dlinear_embedding=outputs["forecast_embedding_dlinear"],
            tcn_embedding=outputs["forecast_embedding_tcn"],
            retrieval_embedding=outputs["retrieval_embedding"],
            dlinear_logit=outputs["forecast_main_logit_dlinear"],
            tcn_logit=outputs["forecast_main_logit_tcn"],
            base_prob=base_main_prob,
            retrieved=retrieved,
            dispersion=retrieved["retrieval_main_dispersion"],
            blend_head=self.main_blend_head,
        )
        mixed_aux_logit, tcn_aux_weight = self._blend_forecasts(
            dlinear_embedding=outputs["forecast_embedding_dlinear"],
            tcn_embedding=outputs["forecast_embedding_tcn"],
            retrieval_embedding=outputs["retrieval_embedding"],
            dlinear_logit=outputs["forecast_aux_logit_dlinear"],
            tcn_logit=outputs["forecast_aux_logit_tcn"],
            base_prob=base_aux_prob,
            retrieved=retrieved,
            dispersion=retrieved["retrieval_aux_dispersion"],
            blend_head=self.aux_blend_head,
        )

        mixed_main_prob = torch.sigmoid(mixed_main_logit)
        mixed_aux_prob = torch.sigmoid(mixed_aux_logit)
        mixed_main_conf = torch.abs(mixed_main_prob - 0.5) * 2.0
        mixed_aux_conf = torch.abs(mixed_aux_prob - 0.5) * 2.0
        main_expert_gap = torch.abs(
            torch.sigmoid(outputs["forecast_main_logit_dlinear"]) - torch.sigmoid(outputs["forecast_main_logit_tcn"])
        )
        aux_expert_gap = torch.abs(
            torch.sigmoid(outputs["forecast_aux_logit_dlinear"]) - torch.sigmoid(outputs["forecast_aux_logit_tcn"])
        )
        main_agreement = 1.0 - main_expert_gap
        aux_agreement = 1.0 - aux_expert_gap

        main_features = torch.cat(
            [
                outputs["forecast_embedding_dlinear"],
                outputs["forecast_embedding_tcn"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_main_dispersion"],
                        torch.abs(mixed_main_prob - base_main_prob),
                        mixed_main_conf,
                        base_gate,
                        tcn_main_weight,
                        main_expert_gap,
                        main_agreement,
                        torch.abs(mixed_main_prob - retrieved["retrieved_main_prob"]),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding_dlinear"],
                outputs["forecast_embedding_tcn"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_aux_dispersion"],
                        torch.abs(mixed_aux_prob - base_aux_prob),
                        mixed_aux_conf,
                        base_gate,
                        tcn_aux_weight,
                        aux_expert_gap,
                        aux_agreement,
                        torch.abs(mixed_aux_prob - retrieved["retrieved_aux_prob"]),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        selector_main_logit = self.main_selector_head(main_features).squeeze(-1)
        selector_aux_logit = self.aux_selector_head(aux_features).squeeze(-1)
        main_correction_gate = torch.sigmoid(selector_main_logit) * main_agreement
        aux_correction_gate = torch.sigmoid(selector_aux_logit) * aux_agreement

        main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=mixed_main_logit,
            correction_gate=main_correction_gate,
        )
        aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=mixed_aux_logit,
            correction_gate=aux_correction_gate,
        )

        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["selector_main_logit"] = selector_main_logit
        outputs["selector_aux_logit"] = selector_aux_logit
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["forecast_main_logit"] = mixed_main_logit
        outputs["forecast_aux_logit"] = mixed_aux_logit
        outputs["forecast_tcn_weight"] = tcn_main_weight
        outputs["forecast_expert_agreement"] = main_agreement
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = main_delta.abs().mean() + aux_delta.abs().mean()
        return outputs


class CampaignMemModularCalibrator(RetrievalForecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.2,
        use_tcn_expert: bool = True,
        use_lstm_expert: bool = False,
        use_softmax_expert_router: bool = False,
        use_abstention: bool = True,
        use_uncertainty_gate: bool = True,
        use_shift_gate: bool = False,
        use_aggressive_gate: bool = False,
        aggressive_route_on_delta: bool = False,
        uncertainty_gate_floor: float = 0.35,
        selector_agreement_floor: float = 0.15,
        shift_floor: float = 0.25,
        aggressive_gate_floor: float = 0.05,
        abstention_detach: bool = True,
        uncertainty_detach: bool = True,
        shift_detach: bool = True,
        aggressive_detach: bool = True,
        base_gate_override: float | None = None,
        calibration_transform: str = "bounded_tanh",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_weighted=True,
            retrieval_aware_gate=False,
            use_residual_fusion=False,
            similarity_temperature=similarity_temperature,
            residual_scale=0.0,
        )
        self.use_tcn_expert = use_tcn_expert
        self.use_lstm_expert = use_lstm_expert
        self.use_softmax_expert_router = use_softmax_expert_router
        self.use_abstention = use_abstention
        self.use_uncertainty_gate = use_uncertainty_gate
        self.use_shift_gate = use_shift_gate
        self.use_aggressive_gate = use_aggressive_gate
        self.aggressive_route_on_delta = aggressive_route_on_delta
        self.uncertainty_gate_floor = uncertainty_gate_floor
        self.selector_agreement_floor = selector_agreement_floor
        self.shift_floor = shift_floor
        self.aggressive_gate_floor = aggressive_gate_floor
        self.abstention_detach = abstention_detach
        self.uncertainty_detach = uncertainty_detach
        self.shift_detach = shift_detach
        self.aggressive_detach = aggressive_detach
        if base_gate_override is not None and not 0.0 <= base_gate_override <= 1.0:
            raise ValueError("base_gate_override must be in [0, 1] when provided")
        self.base_gate_override = base_gate_override
        self.calibration_transform = calibration_transform
        self.expert_names = ["dlinear"]

        self.forecast_encoder = nn.ModuleDict(
            {
                "dlinear": build_encoder(
                    encoder_type="dlinear",
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    seq_len=seq_len,
                ),
            }
        )
        self.forecast_main_head = nn.ModuleDict({"dlinear": nn.Linear(embedding_dim, 1)})
        self.forecast_aux_head = nn.ModuleDict({"dlinear": nn.Linear(embedding_dim, 1)})
        if use_tcn_expert:
            self.forecast_encoder["tcn"] = build_encoder(
                encoder_type="tcn",
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                seq_len=seq_len,
            )
            self.forecast_main_head["tcn"] = nn.Linear(embedding_dim, 1)
            self.forecast_aux_head["tcn"] = nn.Linear(embedding_dim, 1)
            self.expert_names.append("tcn")
        if use_lstm_expert:
            self.forecast_encoder["lstm"] = build_encoder(
                encoder_type="lstm",
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                seq_len=seq_len,
            )
            self.forecast_main_head["lstm"] = nn.Linear(embedding_dim, 1)
            self.forecast_aux_head["lstm"] = nn.Linear(embedding_dim, 1)
            self.expert_names.append("lstm")

        self.delta_scale = delta_scale

        blend_feature_dim = embedding_dim * 3 + 8
        routed_feature_dim = embedding_dim * 2 + 8
        aggressive_feature_dim = (routed_feature_dim if use_softmax_expert_router else blend_feature_dim) + 2
        selector_feature_dim = embedding_dim * 3 + 10
        if use_tcn_expert and not use_softmax_expert_router:
            self.main_blend_head = nn.Sequential(
                nn.Linear(blend_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_blend_head = nn.Sequential(
                nn.Linear(blend_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in (self.main_blend_head, self.aux_blend_head):
                final_linear = module[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.constant_(final_linear.bias, -1.0)
        if use_softmax_expert_router and len(self.expert_names) > 1:
            self.main_expert_router_head = nn.Sequential(
                nn.Linear(routed_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_expert_router_head = nn.Sequential(
                nn.Linear(routed_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in (self.main_expert_router_head, self.aux_expert_router_head):
                final_linear = module[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.constant_(final_linear.bias, 0.0)
        if len(self.expert_names) > 1 and use_aggressive_gate:
            self.main_aggressive_head = nn.Sequential(
                nn.Linear(aggressive_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_aggressive_head = nn.Sequential(
                nn.Linear(aggressive_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in (self.main_aggressive_head, self.aux_aggressive_head):
                final_linear = module[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.constant_(final_linear.bias, -1.75)
        self.main_selector_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_selector_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for module in (self.main_selector_head, self.aux_selector_head):
            final_linear = module[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.constant_(final_linear.bias, -1.25)

        if use_abstention:
            self.main_abstain_head = nn.Sequential(
                nn.Linear(selector_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_abstain_head = nn.Sequential(
                nn.Linear(selector_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in (self.main_abstain_head, self.aux_abstain_head):
                final_linear = module[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.constant_(final_linear.bias, -1.5)

        if use_uncertainty_gate:
            self.main_regime_head = nn.Sequential(
                nn.Linear(selector_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_regime_head = nn.Sequential(
                nn.Linear(selector_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in (self.main_regime_head, self.aux_regime_head):
                final_linear = module[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.constant_(final_linear.bias, 0.5)

        if use_shift_gate:
            shift_feature_dim = selector_feature_dim + 2
            self.main_shift_head = nn.Sequential(
                nn.Linear(shift_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_shift_head = nn.Sequential(
                nn.Linear(shift_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in (self.main_shift_head, self.aux_shift_head):
                final_linear = module[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.constant_(final_linear.bias, -0.25)

    def encode_memory(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.encode(prefix)

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        retrieval_embedding = self.encode(prefix)
        outputs = {
            "embedding": retrieval_embedding,
            "retrieval_embedding": retrieval_embedding,
            "main_logit": self.main_head(retrieval_embedding).squeeze(-1),
            "aux_logit": self.aux_head(retrieval_embedding).squeeze(-1),
        }
        for expert_name in self.expert_names:
            expert_embedding = self.forecast_encoder[expert_name](prefix)
            outputs[f"forecast_embedding_{expert_name}"] = expert_embedding
            outputs[f"forecast_main_logit_{expert_name}"] = self.forecast_main_head[expert_name](expert_embedding).squeeze(-1)
            outputs[f"forecast_aux_logit_{expert_name}"] = self.forecast_aux_head[expert_name](expert_embedding).squeeze(-1)
        for optional_name in ("tcn", "lstm"):
            if optional_name not in self.expert_names:
                outputs[f"forecast_embedding_{optional_name}"] = outputs["forecast_embedding_dlinear"]
                outputs[f"forecast_main_logit_{optional_name}"] = outputs["forecast_main_logit_dlinear"]
                outputs[f"forecast_aux_logit_{optional_name}"] = outputs["forecast_aux_logit_dlinear"]
        prefix_abs = prefix.abs()
        outputs["prefix_scale"] = prefix_abs.mean(dim=(1, 2))
        outputs["prefix_peak"] = prefix_abs.amax(dim=(1, 2))
        outputs["final_main_logit"] = outputs["main_logit"]
        outputs["final_aux_logit"] = outputs["aux_logit"]
        return outputs

    def _bounded_delta(
        self,
        base_logit: torch.Tensor,
        forecast_logit: torch.Tensor,
        correction_gate: torch.Tensor,
    ) -> torch.Tensor:
        logit_gap = forecast_logit - base_logit
        if self.calibration_transform == "bounded_tanh":
            return self.delta_scale * correction_gate * torch.tanh(logit_gap)
        if self.calibration_transform == "linear":
            return self.delta_scale * correction_gate * logit_gap
        if self.calibration_transform == "none":
            return torch.zeros_like(logit_gap)
        raise ValueError(f"Unsupported calibration_transform: {self.calibration_transform}")

    def _blend_forecasts(
        self,
        *,
        dlinear_embedding: torch.Tensor,
        tcn_embedding: torch.Tensor,
        retrieval_embedding: torch.Tensor,
        dlinear_logit: torch.Tensor,
        tcn_logit: torch.Tensor,
        base_prob: torch.Tensor,
        retrieved: dict[str, torch.Tensor],
        dispersion: torch.Tensor,
        blend_head: nn.Module | None,
        prefix_scale: torch.Tensor,
        prefix_peak: torch.Tensor,
        aggressive_head: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_tcn_expert or blend_head is None:
            zeros = torch.zeros_like(base_prob)
            ones = torch.ones_like(base_prob)
            return dlinear_logit, zeros, zeros, zeros, ones, ones

        dlinear_prob = torch.sigmoid(dlinear_logit)
        tcn_prob = torch.sigmoid(tcn_logit)
        dlinear_conf = torch.abs(dlinear_prob - 0.5) * 2.0
        tcn_conf = torch.abs(tcn_prob - 0.5) * 2.0
        disagreement = torch.abs(dlinear_prob - tcn_prob)
        blend_features = torch.cat(
            [
                dlinear_embedding,
                tcn_embedding,
                retrieval_embedding,
                torch.stack(
                    [
                        dlinear_conf,
                        tcn_conf,
                        disagreement,
                        torch.abs(dlinear_prob - base_prob),
                        torch.abs(tcn_prob - base_prob),
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        dispersion,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        raw_tcn_weight = torch.sigmoid(blend_head(blend_features).squeeze(-1))
        aggressive_logit = torch.zeros_like(raw_tcn_weight)
        aggressive_prob = torch.ones_like(raw_tcn_weight)
        aggressive_gate = torch.ones_like(raw_tcn_weight)
        if self.use_aggressive_gate and aggressive_head is not None:
            aggressive_features = torch.cat(
                [blend_features, torch.stack([prefix_scale, prefix_peak], dim=-1)],
                dim=-1,
            )
            if self.aggressive_detach:
                aggressive_features = aggressive_features.detach()
            aggressive_logit = aggressive_head(aggressive_features).squeeze(-1)
            aggressive_prob = torch.sigmoid(aggressive_logit)
            aggressive_gate = self.aggressive_gate_floor + (1.0 - self.aggressive_gate_floor) * aggressive_prob
        tcn_weight = raw_tcn_weight
        if not self.aggressive_route_on_delta:
            tcn_weight = tcn_weight * aggressive_gate
        mixed_logit = (1.0 - tcn_weight) * dlinear_logit + tcn_weight * tcn_logit
        return mixed_logit, tcn_weight, raw_tcn_weight, aggressive_logit, aggressive_prob, aggressive_gate

    def _route_experts(
        self,
        *,
        expert_embeddings: dict[str, torch.Tensor],
        expert_logits: dict[str, torch.Tensor],
        retrieval_embedding: torch.Tensor,
        base_prob: torch.Tensor,
        retrieved_prob: torch.Tensor,
        retrieved: dict[str, torch.Tensor],
        dispersion: torch.Tensor,
        router_head: nn.Module | None,
        prefix_scale: torch.Tensor,
        prefix_peak: torch.Tensor,
        aggressive_head: nn.Module | None,
    ) -> dict[str, torch.Tensor]:
        active_experts = [name for name in self.expert_names if name in expert_embeddings]
        if len(active_experts) <= 1 or router_head is None:
            only_name = active_experts[0]
            batch_size = base_prob.shape[0]
            weights = torch.ones((batch_size, 1), device=base_prob.device, dtype=base_prob.dtype)
            zeros = torch.zeros_like(base_prob)
            ones = torch.ones_like(base_prob)
            return {
                "mixed_logit": expert_logits[only_name],
                "mixed_embedding": expert_embeddings[only_name],
                "weights": weights,
                "raw_weights": weights,
                "non_dlinear_weight": zeros,
                "raw_non_dlinear_weight": zeros,
                "expert_gap": zeros,
                "agreement": ones,
                "aggressive_logit": zeros,
                "aggressive_prob": ones,
                "aggressive_gate": ones,
            }

        route_scores = []
        expert_prob_terms = []
        for name in active_experts:
            expert_logit = expert_logits[name]
            expert_prob = torch.sigmoid(expert_logit)
            expert_conf = torch.abs(expert_prob - 0.5) * 2.0
            route_features = torch.cat(
                [
                    expert_embeddings[name],
                    retrieval_embedding,
                    torch.stack(
                        [
                            expert_conf,
                            torch.abs(expert_prob - base_prob),
                            torch.abs(expert_prob - retrieved_prob),
                            retrieved["retrieval_score_mean"],
                            retrieved["retrieval_score_std"],
                            dispersion,
                            prefix_scale,
                            prefix_peak,
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )
            route_scores.append(router_head(route_features).squeeze(-1))
            expert_prob_terms.append(expert_prob)

        raw_score_tensor = torch.stack(route_scores, dim=-1)
        raw_weight_tensor = F.softmax(raw_score_tensor, dim=-1)
        dlinear_index = active_experts.index("dlinear")
        raw_non_dlinear_weight = 1.0 - raw_weight_tensor[:, dlinear_index]

        mixed_logit = torch.zeros_like(base_prob)
        mixed_embedding = torch.zeros_like(retrieval_embedding)
        for expert_index, name in enumerate(active_experts):
            current_weight = raw_weight_tensor[:, expert_index]
            mixed_logit = mixed_logit + current_weight * expert_logits[name]
            mixed_embedding = mixed_embedding + current_weight.unsqueeze(-1) * expert_embeddings[name]

        mixed_prob = torch.sigmoid(mixed_logit)
        expert_prob_tensor = torch.stack(expert_prob_terms, dim=-1)
        route_dispersion = (raw_weight_tensor * torch.abs(expert_prob_tensor - mixed_prob.unsqueeze(-1))).sum(dim=-1)
        entropy_normalizer = torch.log(
            torch.tensor(float(max(len(active_experts), 2)), device=base_prob.device, dtype=base_prob.dtype)
        )
        route_entropy = -(raw_weight_tensor * raw_weight_tensor.clamp_min(1e-6).log()).sum(dim=-1) / entropy_normalizer

        aggressive_logit = torch.zeros_like(base_prob)
        aggressive_prob = torch.ones_like(base_prob)
        aggressive_gate = torch.ones_like(base_prob)
        weight_tensor = raw_weight_tensor
        if self.use_aggressive_gate and aggressive_head is not None:
            aggressive_features = torch.cat(
                [
                    mixed_embedding,
                    retrieval_embedding,
                    torch.stack(
                        [
                            route_dispersion,
                            route_entropy,
                            torch.abs(mixed_prob - base_prob),
                            torch.abs(mixed_prob - retrieved_prob),
                            retrieved["retrieval_score_mean"],
                            retrieved["retrieval_score_std"],
                            dispersion,
                            raw_non_dlinear_weight,
                        ],
                        dim=-1,
                    ),
                    torch.stack([prefix_scale, prefix_peak], dim=-1),
                ],
                dim=-1,
            )
            if self.aggressive_detach:
                aggressive_features = aggressive_features.detach()
            aggressive_logit = aggressive_head(aggressive_features).squeeze(-1)
            aggressive_prob = torch.sigmoid(aggressive_logit)
            aggressive_gate = self.aggressive_gate_floor + (1.0 - self.aggressive_gate_floor) * aggressive_prob

            if not self.aggressive_route_on_delta:
                weight_tensor = raw_weight_tensor.clone()
                non_dlinear_mask = torch.ones_like(weight_tensor)
                non_dlinear_mask[:, dlinear_index] = 0.0
                weight_tensor = weight_tensor * (non_dlinear_mask * aggressive_gate.unsqueeze(-1) + (1.0 - non_dlinear_mask))
                weight_tensor[:, dlinear_index] = 1.0 - (weight_tensor * non_dlinear_mask).sum(dim=-1)
                mixed_logit = torch.zeros_like(base_prob)
                mixed_embedding = torch.zeros_like(retrieval_embedding)
                for expert_index, name in enumerate(active_experts):
                    current_weight = weight_tensor[:, expert_index]
                    mixed_logit = mixed_logit + current_weight * expert_logits[name]
                    mixed_embedding = mixed_embedding + current_weight.unsqueeze(-1) * expert_embeddings[name]
                mixed_prob = torch.sigmoid(mixed_logit)

        non_dlinear_weight = 1.0 - weight_tensor[:, dlinear_index]
        expert_gap = (weight_tensor * torch.abs(expert_prob_tensor - mixed_prob.unsqueeze(-1))).sum(dim=-1)
        agreement = self.selector_agreement_floor + (1.0 - self.selector_agreement_floor) * (1.0 - expert_gap.clamp(0.0, 1.0))
        return {
            "mixed_logit": mixed_logit,
            "mixed_embedding": mixed_embedding,
            "weights": weight_tensor,
            "raw_weights": raw_weight_tensor,
            "non_dlinear_weight": non_dlinear_weight,
            "raw_non_dlinear_weight": raw_non_dlinear_weight,
            "expert_gap": expert_gap,
            "agreement": agreement,
            "aggressive_logit": aggressive_logit,
            "aggressive_prob": aggressive_prob,
            "aggressive_gate": aggressive_gate,
        }

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        if self.base_gate_override is not None:
            base_gate = torch.full_like(base_gate, float(self.base_gate_override))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        expert_embeddings = {name: outputs[f"forecast_embedding_{name}"] for name in self.expert_names}
        main_expert_logits = {name: outputs[f"forecast_main_logit_{name}"] for name in self.expert_names}
        aux_expert_logits = {name: outputs[f"forecast_aux_logit_{name}"] for name in self.expert_names}
        if self.use_softmax_expert_router and len(self.expert_names) > 1:
            main_route = self._route_experts(
                expert_embeddings=expert_embeddings,
                expert_logits=main_expert_logits,
                retrieval_embedding=outputs["retrieval_embedding"],
                base_prob=base_main_prob,
                retrieved_prob=retrieved["retrieved_main_prob"],
                retrieved=retrieved,
                dispersion=retrieved["retrieval_main_dispersion"],
                router_head=getattr(self, "main_expert_router_head", None),
                prefix_scale=outputs["prefix_scale"],
                prefix_peak=outputs["prefix_peak"],
                aggressive_head=getattr(self, "main_aggressive_head", None),
            )
            mixed_main_logit = main_route["mixed_logit"]
            mixed_main_embedding = main_route["mixed_embedding"]
            tcn_main_weight = main_route["non_dlinear_weight"]
            raw_tcn_main_weight = main_route["raw_non_dlinear_weight"]
            aggressive_main_logit = main_route["aggressive_logit"]
            aggressive_main_prob = main_route["aggressive_prob"]
            aggressive_main_gate = main_route["aggressive_gate"]
            main_expert_gap = main_route["expert_gap"]
            main_agreement = main_route["agreement"]

            aux_route = self._route_experts(
                expert_embeddings=expert_embeddings,
                expert_logits=aux_expert_logits,
                retrieval_embedding=outputs["retrieval_embedding"],
                base_prob=base_aux_prob,
                retrieved_prob=retrieved["retrieved_aux_prob"],
                retrieved=retrieved,
                dispersion=retrieved["retrieval_aux_dispersion"],
                router_head=getattr(self, "aux_expert_router_head", None),
                prefix_scale=outputs["prefix_scale"],
                prefix_peak=outputs["prefix_peak"],
                aggressive_head=getattr(self, "aux_aggressive_head", None),
            )
            mixed_aux_logit = aux_route["mixed_logit"]
            mixed_aux_embedding = aux_route["mixed_embedding"]
            tcn_aux_weight = aux_route["non_dlinear_weight"]
            raw_tcn_aux_weight = aux_route["raw_non_dlinear_weight"]
            aggressive_aux_logit = aux_route["aggressive_logit"]
            aggressive_aux_prob = aux_route["aggressive_prob"]
            aggressive_aux_gate = aux_route["aggressive_gate"]
            aux_expert_gap = aux_route["expert_gap"]
            aux_agreement = aux_route["agreement"]
        else:
            mixed_main_logit, tcn_main_weight, raw_tcn_main_weight, aggressive_main_logit, aggressive_main_prob, aggressive_main_gate = self._blend_forecasts(
                dlinear_embedding=outputs["forecast_embedding_dlinear"],
                tcn_embedding=outputs["forecast_embedding_tcn"],
                retrieval_embedding=outputs["retrieval_embedding"],
                dlinear_logit=outputs["forecast_main_logit_dlinear"],
                tcn_logit=outputs["forecast_main_logit_tcn"],
                base_prob=base_main_prob,
                retrieved=retrieved,
                dispersion=retrieved["retrieval_main_dispersion"],
                blend_head=getattr(self, "main_blend_head", None),
                prefix_scale=outputs["prefix_scale"],
                prefix_peak=outputs["prefix_peak"],
                aggressive_head=getattr(self, "main_aggressive_head", None),
            )
            mixed_main_embedding = (
                (1.0 - tcn_main_weight).unsqueeze(-1) * outputs["forecast_embedding_dlinear"]
                + tcn_main_weight.unsqueeze(-1) * outputs["forecast_embedding_tcn"]
            )
            mixed_aux_logit, tcn_aux_weight, raw_tcn_aux_weight, aggressive_aux_logit, aggressive_aux_prob, aggressive_aux_gate = self._blend_forecasts(
                dlinear_embedding=outputs["forecast_embedding_dlinear"],
                tcn_embedding=outputs["forecast_embedding_tcn"],
                retrieval_embedding=outputs["retrieval_embedding"],
                dlinear_logit=outputs["forecast_aux_logit_dlinear"],
                tcn_logit=outputs["forecast_aux_logit_tcn"],
                base_prob=base_aux_prob,
                retrieved=retrieved,
                dispersion=retrieved["retrieval_aux_dispersion"],
                blend_head=getattr(self, "aux_blend_head", None),
                prefix_scale=outputs["prefix_scale"],
                prefix_peak=outputs["prefix_peak"],
                aggressive_head=getattr(self, "aux_aggressive_head", None),
            )
            mixed_aux_embedding = (
                (1.0 - tcn_aux_weight).unsqueeze(-1) * outputs["forecast_embedding_dlinear"]
                + tcn_aux_weight.unsqueeze(-1) * outputs["forecast_embedding_tcn"]
            )

        mixed_main_prob = torch.sigmoid(mixed_main_logit)
        mixed_aux_prob = torch.sigmoid(mixed_aux_logit)
        mixed_main_conf = torch.abs(mixed_main_prob - 0.5) * 2.0
        mixed_aux_conf = torch.abs(mixed_aux_prob - 0.5) * 2.0
        if self.use_softmax_expert_router and len(self.expert_names) > 1:
            outputs["forecast_route_weight_dlinear"] = main_route["weights"][:, self.expert_names.index("dlinear")]
            outputs["forecast_route_weight_dlinear_raw"] = main_route["raw_weights"][:, self.expert_names.index("dlinear")]
            outputs["forecast_route_aux_weight_dlinear_raw"] = aux_route["raw_weights"][:, self.expert_names.index("dlinear")]
            for expert_name in self.expert_names:
                expert_index = self.expert_names.index(expert_name)
                outputs[f"forecast_route_weight_{expert_name}"] = main_route["weights"][:, expert_index]
                outputs[f"forecast_route_weight_{expert_name}_raw"] = main_route["raw_weights"][:, expert_index]
                outputs[f"forecast_route_aux_weight_{expert_name}_raw"] = aux_route["raw_weights"][:, expert_index]
        elif self.use_tcn_expert:
            main_expert_gap = torch.abs(
                torch.sigmoid(outputs["forecast_main_logit_dlinear"]) - torch.sigmoid(outputs["forecast_main_logit_tcn"])
            )
            aux_expert_gap = torch.abs(
                torch.sigmoid(outputs["forecast_aux_logit_dlinear"]) - torch.sigmoid(outputs["forecast_aux_logit_tcn"])
            )
            main_agreement = self.selector_agreement_floor + (1.0 - self.selector_agreement_floor) * (1.0 - main_expert_gap)
            aux_agreement = self.selector_agreement_floor + (1.0 - self.selector_agreement_floor) * (1.0 - aux_expert_gap)
        else:
            main_expert_gap = torch.zeros_like(mixed_main_prob)
            aux_expert_gap = torch.zeros_like(mixed_aux_prob)
            main_agreement = torch.ones_like(mixed_main_prob)
            aux_agreement = torch.ones_like(mixed_aux_prob)

        main_features = torch.cat(
            [
                outputs["forecast_embedding_dlinear"],
                mixed_main_embedding,
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_main_dispersion"],
                        torch.abs(mixed_main_prob - base_main_prob),
                        mixed_main_conf,
                        base_gate,
                        tcn_main_weight,
                        main_expert_gap,
                        main_agreement,
                        torch.abs(mixed_main_prob - retrieved["retrieved_main_prob"]),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding_dlinear"],
                mixed_aux_embedding,
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_aux_dispersion"],
                        torch.abs(mixed_aux_prob - base_aux_prob),
                        mixed_aux_conf,
                        base_gate,
                        tcn_aux_weight,
                        aux_expert_gap,
                        aux_agreement,
                        torch.abs(mixed_aux_prob - retrieved["retrieved_aux_prob"]),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        selector_main_logit = self.main_selector_head(main_features).squeeze(-1)
        selector_aux_logit = self.aux_selector_head(aux_features).squeeze(-1)
        main_correction_gate = torch.sigmoid(selector_main_logit) * main_agreement
        aux_correction_gate = torch.sigmoid(selector_aux_logit) * aux_agreement

        abstain_main_prob = torch.zeros_like(main_correction_gate)
        abstain_aux_prob = torch.zeros_like(aux_correction_gate)
        if self.use_abstention:
            abstain_main_features = main_features.detach() if self.abstention_detach else main_features
            abstain_aux_features = aux_features.detach() if self.abstention_detach else aux_features
            abstain_main_logit = self.main_abstain_head(abstain_main_features).squeeze(-1)
            abstain_aux_logit = self.aux_abstain_head(abstain_aux_features).squeeze(-1)
            abstain_main_prob = torch.sigmoid(abstain_main_logit)
            abstain_aux_prob = torch.sigmoid(abstain_aux_logit)
            main_correction_gate = main_correction_gate * (1.0 - abstain_main_prob)
            aux_correction_gate = aux_correction_gate * (1.0 - abstain_aux_prob)
            outputs["abstain_main_logit"] = abstain_main_logit
            outputs["abstain_aux_logit"] = abstain_aux_logit
            outputs["abstain_main_prob"] = abstain_main_prob
            outputs["abstain_aux_prob"] = abstain_aux_prob

        regime_main_gate = torch.ones_like(main_correction_gate)
        regime_aux_gate = torch.ones_like(aux_correction_gate)
        if self.use_uncertainty_gate:
            regime_main_features = main_features.detach() if self.uncertainty_detach else main_features
            regime_aux_features = aux_features.detach() if self.uncertainty_detach else aux_features
            regime_main_logit = self.main_regime_head(regime_main_features).squeeze(-1)
            regime_aux_logit = self.aux_regime_head(regime_aux_features).squeeze(-1)
            regime_main_prob = torch.sigmoid(regime_main_logit)
            regime_aux_prob = torch.sigmoid(regime_aux_logit)
            regime_main_gate = self.uncertainty_gate_floor + (1.0 - self.uncertainty_gate_floor) * regime_main_prob
            regime_aux_gate = self.uncertainty_gate_floor + (1.0 - self.uncertainty_gate_floor) * regime_aux_prob
            main_correction_gate = main_correction_gate * regime_main_gate
            aux_correction_gate = aux_correction_gate * regime_aux_gate
            outputs["regime_main_logit"] = regime_main_logit
            outputs["regime_aux_logit"] = regime_aux_logit
            outputs["regime_main_prob"] = regime_main_prob
            outputs["regime_aux_prob"] = regime_aux_prob
            outputs["regime_main_gate"] = regime_main_gate
            outputs["regime_aux_gate"] = regime_aux_gate

        shift_main_gate = torch.ones_like(main_correction_gate)
        shift_aux_gate = torch.ones_like(aux_correction_gate)
        if self.use_shift_gate:
            shift_stats = torch.stack([outputs["prefix_scale"], outputs["prefix_peak"]], dim=-1)
            main_shift_features = torch.cat([main_features, shift_stats], dim=-1)
            aux_shift_features = torch.cat([aux_features, shift_stats], dim=-1)
            if self.shift_detach:
                main_shift_features = main_shift_features.detach()
                aux_shift_features = aux_shift_features.detach()
            shift_main_logit = self.main_shift_head(main_shift_features).squeeze(-1)
            shift_aux_logit = self.aux_shift_head(aux_shift_features).squeeze(-1)
            shift_main_prob = torch.sigmoid(shift_main_logit)
            shift_aux_prob = torch.sigmoid(shift_aux_logit)
            shift_main_gate = self.shift_floor + (1.0 - self.shift_floor) * shift_main_prob
            shift_aux_gate = self.shift_floor + (1.0 - self.shift_floor) * shift_aux_prob
            main_correction_gate = main_correction_gate * shift_main_gate
            aux_correction_gate = aux_correction_gate * shift_aux_gate
            outputs["shift_main_logit"] = shift_main_logit
            outputs["shift_aux_logit"] = shift_aux_logit
            outputs["shift_main_prob"] = shift_main_prob
            outputs["shift_aux_prob"] = shift_aux_prob

        if self.use_aggressive_gate and self.aggressive_route_on_delta and len(self.expert_names) > 1:
            stable_main_delta = self._bounded_delta(
                base_logit=base_main_logit,
                forecast_logit=outputs["forecast_main_logit_dlinear"],
                correction_gate=main_correction_gate,
            )
            stable_aux_delta = self._bounded_delta(
                base_logit=base_aux_logit,
                forecast_logit=outputs["forecast_aux_logit_dlinear"],
                correction_gate=aux_correction_gate,
            )
            aggressive_main_delta = self._bounded_delta(
                base_logit=base_main_logit,
                forecast_logit=mixed_main_logit,
                correction_gate=main_correction_gate,
            )
            aggressive_aux_delta = self._bounded_delta(
                base_logit=base_aux_logit,
                forecast_logit=mixed_aux_logit,
                correction_gate=aux_correction_gate,
            )
            main_delta = (1.0 - aggressive_main_gate) * stable_main_delta + aggressive_main_gate * aggressive_main_delta
            aux_delta = (1.0 - aggressive_aux_gate) * stable_aux_delta + aggressive_aux_gate * aggressive_aux_delta
            outputs["stable_main_delta"] = stable_main_delta
            outputs["stable_aux_delta"] = stable_aux_delta
            outputs["aggressive_main_delta"] = aggressive_main_delta
            outputs["aggressive_aux_delta"] = aggressive_aux_delta
        else:
            main_delta = self._bounded_delta(
                base_logit=base_main_logit,
                forecast_logit=mixed_main_logit,
                correction_gate=main_correction_gate,
            )
            aux_delta = self._bounded_delta(
                base_logit=base_aux_logit,
                forecast_logit=mixed_aux_logit,
                correction_gate=aux_correction_gate,
            )

        main_penalty_scale = 1.0 + abstain_main_prob + (1.0 - regime_main_gate) + (1.0 - shift_main_gate)
        aux_penalty_scale = 1.0 + abstain_aux_prob + (1.0 - regime_aux_gate) + (1.0 - shift_aux_gate)
        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["selector_main_logit"] = selector_main_logit
        outputs["selector_aux_logit"] = selector_aux_logit
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["forecast_main_logit"] = mixed_main_logit
        outputs["forecast_aux_logit"] = mixed_aux_logit
        outputs["forecast_tcn_weight"] = tcn_main_weight
        outputs["forecast_tcn_weight_raw"] = raw_tcn_main_weight
        outputs["forecast_expert_agreement"] = main_agreement
        if self.use_softmax_expert_router and len(self.expert_names) > 1:
            outputs["forecast_tcn_aux_weight_raw"] = raw_tcn_aux_weight
        if self.use_aggressive_gate and len(self.expert_names) > 1:
            outputs["aggressive_main_logit"] = aggressive_main_logit
            outputs["aggressive_aux_logit"] = aggressive_aux_logit
            outputs["aggressive_main_prob"] = aggressive_main_prob
            outputs["aggressive_aux_prob"] = aggressive_aux_prob
            outputs["aggressive_main_gate"] = aggressive_main_gate
            outputs["aggressive_aux_gate"] = aggressive_aux_gate
            outputs["forecast_tcn_aux_weight_raw"] = raw_tcn_aux_weight
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (
            (main_penalty_scale * main_delta.abs()).mean() + (aux_penalty_scale * aux_delta.abs()).mean()
        )
        return outputs


class CampaignMemRegimeRouterCalibrator(RetrievalForecaster):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        retrieval_encoder_type: str,
        shock_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.12,
        trend_kernel: int = 5,
        use_abstention: bool = True,
        use_shift_gate: bool = True,
        use_aggressive_gate: bool = True,
        selector_agreement_floor: float = 0.2,
        shift_floor: float = 0.35,
        shock_gate_floor: float = 0.1,
        shock_prior_scale: float = 0.6,
        abstention_detach: bool = True,
        shift_detach: bool = True,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_weighted=True,
            retrieval_aware_gate=False,
            use_residual_fusion=False,
            similarity_temperature=similarity_temperature,
            residual_scale=0.0,
        )
        self.seq_len = seq_len
        self.delta_scale = delta_scale
        self.trend_kernel = trend_kernel
        self.use_abstention = use_abstention
        self.use_shift_gate = use_shift_gate
        self.use_aggressive_gate = use_aggressive_gate
        self.selector_agreement_floor = selector_agreement_floor
        self.shift_floor = shift_floor
        self.shock_gate_floor = shock_gate_floor
        self.shock_prior_scale = shock_prior_scale
        self.abstention_detach = abstention_detach
        self.shift_detach = shift_detach

        self.trend_encoder = build_encoder(
            encoder_type="dlinear",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
        self.shock_encoder = build_encoder(
            encoder_type=shock_encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
        self.trend_main_head = nn.Linear(embedding_dim, 1)
        self.trend_aux_head = nn.Linear(embedding_dim, 1)
        self.shock_main_head = nn.Linear(embedding_dim, 1)
        self.shock_aux_head = nn.Linear(embedding_dim, 1)

        route_feature_dim = embedding_dim * 3 + 14
        selector_feature_dim = embedding_dim * 3 + 14
        shift_feature_dim = selector_feature_dim + 2
        self.main_route_head = nn.Sequential(
            nn.Linear(route_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_route_head = nn.Sequential(
            nn.Linear(route_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for module in (self.main_route_head, self.aux_route_head):
            final_linear = module[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.constant_(final_linear.bias, -0.75)

        self.main_selector_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_selector_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for module in (self.main_selector_head, self.aux_selector_head):
            final_linear = module[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.constant_(final_linear.bias, -1.0)

        if use_abstention:
            self.main_abstain_head = nn.Sequential(
                nn.Linear(selector_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_abstain_head = nn.Sequential(
                nn.Linear(selector_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in (self.main_abstain_head, self.aux_abstain_head):
                final_linear = module[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.constant_(final_linear.bias, -1.5)

        if use_shift_gate:
            self.main_shift_head = nn.Sequential(
                nn.Linear(shift_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.aux_shift_head = nn.Sequential(
                nn.Linear(shift_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in (self.main_shift_head, self.aux_shift_head):
                final_linear = module[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.constant_(final_linear.bias, -0.25)

    def encode_memory(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.encode(prefix)

    def _bounded_delta(
        self,
        base_logit: torch.Tensor,
        forecast_logit: torch.Tensor,
        correction_gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.delta_scale * correction_gate * torch.tanh(forecast_logit - base_logit)

    def _decompose_prefix(
        self,
        prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        kernel_size = min(max(self.trend_kernel, 1), max(prefix.shape[1], 1))
        trend = _moving_average(prefix, kernel_size=kernel_size)
        residual = prefix - trend
        prefix_abs = prefix.abs()
        trend_abs = trend.abs()
        residual_abs = residual.abs()
        prefix_scale = prefix_abs.mean(dim=(1, 2))
        prefix_peak = prefix_abs.amax(dim=(1, 2))
        trend_scale = trend_abs.mean(dim=(1, 2))
        residual_scale = residual_abs.mean(dim=(1, 2))
        residual_ratio = residual_scale / prefix_scale.clamp_min(1e-6)
        residual_peak_ratio = residual_abs.amax(dim=(1, 2)) / prefix_peak.clamp_min(1e-6)
        tail_steps = min(4, prefix.shape[1])
        shock_burst = residual_abs[:, -tail_steps:, :].mean(dim=(1, 2))
        if prefix.shape[1] > 1:
            shock_change = (residual[:, 1:, :] - residual[:, :-1, :]).abs().mean(dim=(1, 2))
        else:
            shock_change = torch.zeros_like(prefix_scale)
        trend_drift = (trend[:, -1, :] - trend[:, 0, :]).abs().mean(dim=-1)
        burst_ratio = shock_burst / prefix_scale.clamp_min(1e-6)
        change_ratio = shock_change / prefix_scale.clamp_min(1e-6)
        stats = {
            "prefix_scale": prefix_scale,
            "prefix_peak": prefix_peak,
            "trend_scale": trend_scale,
            "residual_scale": residual_scale,
            "residual_ratio": residual_ratio.clamp(0.0, 1.5),
            "residual_peak_ratio": residual_peak_ratio.clamp(0.0, 1.5),
            "shock_burst": shock_burst,
            "shock_change": shock_change,
            "burst_ratio": burst_ratio.clamp(0.0, 1.5),
            "change_ratio": change_ratio.clamp(0.0, 1.5),
            "trend_drift": trend_drift,
        }
        return trend, residual, stats

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        retrieval_embedding = self.encode(prefix)
        trend_prefix, residual_prefix, decomp_stats = self._decompose_prefix(prefix)
        trend_embedding = self.trend_encoder(trend_prefix)
        shock_embedding = self.shock_encoder(residual_prefix)
        outputs = {
            "embedding": retrieval_embedding,
            "retrieval_embedding": retrieval_embedding,
            "forecast_embedding_trend": trend_embedding,
            "forecast_embedding_shock": shock_embedding,
            "main_logit": self.main_head(retrieval_embedding).squeeze(-1),
            "aux_logit": self.aux_head(retrieval_embedding).squeeze(-1),
            "forecast_main_logit_trend": self.trend_main_head(trend_embedding).squeeze(-1),
            "forecast_aux_logit_trend": self.trend_aux_head(trend_embedding).squeeze(-1),
            "forecast_main_logit_shock": self.shock_main_head(shock_embedding).squeeze(-1),
            "forecast_aux_logit_shock": self.shock_aux_head(shock_embedding).squeeze(-1),
            "final_main_logit": self.main_head(retrieval_embedding).squeeze(-1),
            "final_aux_logit": self.aux_head(retrieval_embedding).squeeze(-1),
        }
        outputs.update(decomp_stats)
        return outputs

    def _route_branch(
        self,
        *,
        trend_embedding: torch.Tensor,
        shock_embedding: torch.Tensor,
        retrieval_embedding: torch.Tensor,
        trend_logit: torch.Tensor,
        shock_logit: torch.Tensor,
        base_prob: torch.Tensor,
        retrieved_prob: torch.Tensor,
        retrieved: dict[str, torch.Tensor],
        dispersion: torch.Tensor,
        route_head: nn.Module,
        residual_ratio: torch.Tensor,
        residual_peak_ratio: torch.Tensor,
        burst_ratio: torch.Tensor,
        change_ratio: torch.Tensor,
        trend_drift: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        trend_prob = torch.sigmoid(trend_logit)
        shock_prob = torch.sigmoid(shock_logit)
        trend_conf = torch.abs(trend_prob - 0.5) * 2.0
        shock_conf = torch.abs(shock_prob - 0.5) * 2.0
        branch_gap = torch.abs(trend_prob - shock_prob)
        shock_prior = torch.stack(
            [
                residual_ratio.clamp(0.0, 1.0),
                residual_peak_ratio.clamp(0.0, 1.0),
                burst_ratio.clamp(0.0, 1.0),
                change_ratio.clamp(0.0, 1.0),
            ],
            dim=-1,
        ).mean(dim=-1)
        route_features = torch.cat(
            [
                trend_embedding,
                shock_embedding,
                retrieval_embedding,
                torch.stack(
                    [
                        trend_conf,
                        shock_conf,
                        branch_gap,
                        torch.abs(trend_prob - base_prob),
                        torch.abs(shock_prob - base_prob),
                        torch.abs(shock_prob - retrieved_prob),
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        dispersion,
                        residual_ratio,
                        residual_peak_ratio,
                        burst_ratio,
                        change_ratio,
                        trend_drift,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        route_logit = route_head(route_features).squeeze(-1) + self.shock_prior_scale * torch.logit(
            shock_prior.clamp(0.05, 0.95)
        )
        route_prob = torch.sigmoid(route_logit)
        shock_gate = self.shock_gate_floor + (1.0 - self.shock_gate_floor) * route_prob
        mixed_logit = (1.0 - shock_gate) * trend_logit + shock_gate * shock_logit
        mixed_embedding = (1.0 - shock_gate).unsqueeze(-1) * trend_embedding + shock_gate.unsqueeze(-1) * shock_embedding
        agreement = self.selector_agreement_floor + (1.0 - self.selector_agreement_floor) * (1.0 - branch_gap.clamp(0.0, 1.0))
        return {
            "mixed_logit": mixed_logit,
            "mixed_embedding": mixed_embedding,
            "route_logit": route_logit,
            "route_prob": route_prob,
            "shock_gate": shock_gate,
            "branch_gap": branch_gap,
            "agreement": agreement,
        }

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1.0 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1.0 - base_gate) * retrieved["retrieved_aux_logit"]
        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)

        main_route = self._route_branch(
            trend_embedding=outputs["forecast_embedding_trend"],
            shock_embedding=outputs["forecast_embedding_shock"],
            retrieval_embedding=outputs["retrieval_embedding"],
            trend_logit=outputs["forecast_main_logit_trend"],
            shock_logit=outputs["forecast_main_logit_shock"],
            base_prob=base_main_prob,
            retrieved_prob=retrieved["retrieved_main_prob"],
            retrieved=retrieved,
            dispersion=retrieved["retrieval_main_dispersion"],
            route_head=self.main_route_head,
            residual_ratio=outputs["residual_ratio"],
            residual_peak_ratio=outputs["residual_peak_ratio"],
            burst_ratio=outputs["burst_ratio"],
            change_ratio=outputs["change_ratio"],
            trend_drift=outputs["trend_drift"],
        )
        aux_route = self._route_branch(
            trend_embedding=outputs["forecast_embedding_trend"],
            shock_embedding=outputs["forecast_embedding_shock"],
            retrieval_embedding=outputs["retrieval_embedding"],
            trend_logit=outputs["forecast_aux_logit_trend"],
            shock_logit=outputs["forecast_aux_logit_shock"],
            base_prob=base_aux_prob,
            retrieved_prob=retrieved["retrieved_aux_prob"],
            retrieved=retrieved,
            dispersion=retrieved["retrieval_aux_dispersion"],
            route_head=self.aux_route_head,
            residual_ratio=outputs["residual_ratio"],
            residual_peak_ratio=outputs["residual_peak_ratio"],
            burst_ratio=outputs["burst_ratio"],
            change_ratio=outputs["change_ratio"],
            trend_drift=outputs["trend_drift"],
        )

        mixed_main_logit = main_route["mixed_logit"]
        mixed_aux_logit = aux_route["mixed_logit"]
        mixed_main_prob = torch.sigmoid(mixed_main_logit)
        mixed_aux_prob = torch.sigmoid(mixed_aux_logit)
        mixed_main_conf = torch.abs(mixed_main_prob - 0.5) * 2.0
        mixed_aux_conf = torch.abs(mixed_aux_prob - 0.5) * 2.0

        main_features = torch.cat(
            [
                outputs["forecast_embedding_trend"],
                main_route["mixed_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_main_dispersion"],
                        torch.abs(mixed_main_prob - base_main_prob),
                        mixed_main_conf,
                        base_gate,
                        main_route["shock_gate"],
                        main_route["branch_gap"],
                        main_route["agreement"],
                        torch.abs(mixed_main_prob - retrieved["retrieved_main_prob"]),
                        outputs["residual_ratio"],
                        outputs["residual_peak_ratio"],
                        outputs["burst_ratio"],
                        outputs["change_ratio"],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding_trend"],
                aux_route["mixed_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_aux_dispersion"],
                        torch.abs(mixed_aux_prob - base_aux_prob),
                        mixed_aux_conf,
                        base_gate,
                        aux_route["shock_gate"],
                        aux_route["branch_gap"],
                        aux_route["agreement"],
                        torch.abs(mixed_aux_prob - retrieved["retrieved_aux_prob"]),
                        outputs["residual_ratio"],
                        outputs["residual_peak_ratio"],
                        outputs["burst_ratio"],
                        outputs["change_ratio"],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        selector_main_logit = self.main_selector_head(main_features).squeeze(-1)
        selector_aux_logit = self.aux_selector_head(aux_features).squeeze(-1)
        main_correction_gate = torch.sigmoid(selector_main_logit) * main_route["agreement"]
        aux_correction_gate = torch.sigmoid(selector_aux_logit) * aux_route["agreement"]

        abstain_main_prob = torch.zeros_like(main_correction_gate)
        abstain_aux_prob = torch.zeros_like(aux_correction_gate)
        if self.use_abstention:
            abstain_main_features = main_features.detach() if self.abstention_detach else main_features
            abstain_aux_features = aux_features.detach() if self.abstention_detach else aux_features
            abstain_main_logit = self.main_abstain_head(abstain_main_features).squeeze(-1)
            abstain_aux_logit = self.aux_abstain_head(abstain_aux_features).squeeze(-1)
            abstain_main_prob = torch.sigmoid(abstain_main_logit)
            abstain_aux_prob = torch.sigmoid(abstain_aux_logit)
            main_correction_gate = main_correction_gate * (1.0 - abstain_main_prob)
            aux_correction_gate = aux_correction_gate * (1.0 - abstain_aux_prob)
            outputs["abstain_main_logit"] = abstain_main_logit
            outputs["abstain_aux_logit"] = abstain_aux_logit
            outputs["abstain_main_prob"] = abstain_main_prob
            outputs["abstain_aux_prob"] = abstain_aux_prob

        shift_main_gate = torch.ones_like(main_correction_gate)
        shift_aux_gate = torch.ones_like(aux_correction_gate)
        if self.use_shift_gate:
            shift_stats = torch.stack([outputs["prefix_scale"], outputs["prefix_peak"]], dim=-1)
            main_shift_features = torch.cat([main_features, shift_stats], dim=-1)
            aux_shift_features = torch.cat([aux_features, shift_stats], dim=-1)
            if self.shift_detach:
                main_shift_features = main_shift_features.detach()
                aux_shift_features = aux_shift_features.detach()
            shift_main_logit = self.main_shift_head(main_shift_features).squeeze(-1)
            shift_aux_logit = self.aux_shift_head(aux_shift_features).squeeze(-1)
            shift_main_prob = torch.sigmoid(shift_main_logit)
            shift_aux_prob = torch.sigmoid(shift_aux_logit)
            shift_main_gate = self.shift_floor + (1.0 - self.shift_floor) * shift_main_prob
            shift_aux_gate = self.shift_floor + (1.0 - self.shift_floor) * shift_aux_prob
            main_correction_gate = main_correction_gate * shift_main_gate
            aux_correction_gate = aux_correction_gate * shift_aux_gate
            outputs["shift_main_logit"] = shift_main_logit
            outputs["shift_aux_logit"] = shift_aux_logit
            outputs["shift_main_prob"] = shift_main_prob
            outputs["shift_aux_prob"] = shift_aux_prob

        stable_main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit_trend"],
            correction_gate=main_correction_gate,
        )
        stable_aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit_trend"],
            correction_gate=aux_correction_gate,
        )
        shock_main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit_shock"],
            correction_gate=main_correction_gate,
        )
        shock_aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit_shock"],
            correction_gate=aux_correction_gate,
        )
        main_delta = (1.0 - main_route["shock_gate"]) * stable_main_delta + main_route["shock_gate"] * shock_main_delta
        aux_delta = (1.0 - aux_route["shock_gate"]) * stable_aux_delta + aux_route["shock_gate"] * shock_aux_delta

        main_penalty_scale = 1.0 + abstain_main_prob + (1.0 - shift_main_gate)
        aux_penalty_scale = 1.0 + abstain_aux_prob + (1.0 - shift_aux_gate)
        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["selector_main_logit"] = selector_main_logit
        outputs["selector_aux_logit"] = selector_aux_logit
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["forecast_main_logit"] = mixed_main_logit
        outputs["forecast_aux_logit"] = mixed_aux_logit
        outputs["forecast_shock_weight"] = main_route["shock_gate"]
        outputs["forecast_shock_weight_raw"] = main_route["route_prob"]
        outputs["forecast_tcn_weight"] = main_route["shock_gate"]
        outputs["forecast_tcn_weight_raw"] = main_route["route_prob"]
        outputs["forecast_expert_agreement"] = main_route["agreement"]
        outputs["stable_main_delta"] = stable_main_delta
        outputs["stable_aux_delta"] = stable_aux_delta
        outputs["aggressive_main_delta"] = shock_main_delta
        outputs["aggressive_aux_delta"] = shock_aux_delta
        if self.use_aggressive_gate:
            outputs["aggressive_main_logit"] = main_route["route_logit"]
            outputs["aggressive_aux_logit"] = aux_route["route_logit"]
            outputs["aggressive_main_prob"] = main_route["route_prob"]
            outputs["aggressive_aux_prob"] = aux_route["route_prob"]
            outputs["aggressive_main_gate"] = main_route["shock_gate"]
            outputs["aggressive_aux_gate"] = aux_route["shock_gate"]
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (
            (main_penalty_scale * main_delta.abs()).mean() + (aux_penalty_scale * aux_delta.abs()).mean()
        )
        return outputs


class CampaignMemDecompModularCalibrator(CampaignMemModularCalibrator):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        retrieval_encoder_type: str,
        stable_encoder_type: str,
        shock_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.12,
        trend_kernel: int = 5,
        use_abstention: bool = True,
        use_uncertainty_gate: bool = False,
        use_shift_gate: bool = True,
        use_aggressive_gate: bool = True,
        aggressive_route_on_delta: bool = True,
        uncertainty_gate_floor: float = 0.35,
        selector_agreement_floor: float = 0.2,
        shift_floor: float = 0.35,
        aggressive_gate_floor: float = 0.1,
        decomp_prior_mix: float = 0.0,
        decomp_prior_floor: float = 0.1,
        abstention_detach: bool = True,
        uncertainty_detach: bool = True,
        shift_detach: bool = True,
        aggressive_detach: bool = True,
        base_gate_override: float | None = None,
        calibration_transform: str = "bounded_tanh",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            retrieval_encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_temperature=similarity_temperature,
            delta_scale=delta_scale,
            use_tcn_expert=True,
            use_lstm_expert=False,
            use_softmax_expert_router=False,
            use_abstention=use_abstention,
            use_uncertainty_gate=use_uncertainty_gate,
            use_shift_gate=use_shift_gate,
            use_aggressive_gate=use_aggressive_gate,
            aggressive_route_on_delta=aggressive_route_on_delta,
            uncertainty_gate_floor=uncertainty_gate_floor,
            selector_agreement_floor=selector_agreement_floor,
            shift_floor=shift_floor,
            aggressive_gate_floor=aggressive_gate_floor,
            abstention_detach=abstention_detach,
            uncertainty_detach=uncertainty_detach,
            shift_detach=shift_detach,
            aggressive_detach=aggressive_detach,
            base_gate_override=base_gate_override,
            calibration_transform=calibration_transform,
        )
        self.trend_kernel = trend_kernel
        self.decomp_prior_mix = decomp_prior_mix
        self.decomp_prior_floor = decomp_prior_floor
        self.forecast_encoder["dlinear"] = build_encoder(
            encoder_type=stable_encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
        self.forecast_encoder["tcn"] = build_encoder(
            encoder_type=shock_encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )

    def _blend_forecasts(
        self,
        *,
        dlinear_embedding: torch.Tensor,
        tcn_embedding: torch.Tensor,
        retrieval_embedding: torch.Tensor,
        dlinear_logit: torch.Tensor,
        tcn_logit: torch.Tensor,
        base_prob: torch.Tensor,
        retrieved: dict[str, torch.Tensor],
        dispersion: torch.Tensor,
        blend_head: nn.Module | None,
        prefix_scale: torch.Tensor,
        prefix_peak: torch.Tensor,
        aggressive_head: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_logit, tcn_weight, raw_tcn_weight, aggressive_logit, aggressive_prob, aggressive_gate = super()._blend_forecasts(
            dlinear_embedding=dlinear_embedding,
            tcn_embedding=tcn_embedding,
            retrieval_embedding=retrieval_embedding,
            dlinear_logit=dlinear_logit,
            tcn_logit=tcn_logit,
            base_prob=base_prob,
            retrieved=retrieved,
            dispersion=dispersion,
            blend_head=blend_head,
            prefix_scale=prefix_scale,
            prefix_peak=prefix_peak,
            aggressive_head=aggressive_head,
        )
        shock_prior = getattr(self, "_cached_shock_prior", None)
        if shock_prior is not None and self.decomp_prior_mix > 0:
            prior_gate = self.decomp_prior_floor + (1.0 - self.decomp_prior_floor) * shock_prior
            tcn_weight = (1.0 - self.decomp_prior_mix) * tcn_weight + self.decomp_prior_mix * prior_gate
            raw_tcn_weight = (1.0 - self.decomp_prior_mix) * raw_tcn_weight + self.decomp_prior_mix * prior_gate
            mixed_logit = (1.0 - tcn_weight) * dlinear_logit + tcn_weight * tcn_logit
        return mixed_logit, tcn_weight, raw_tcn_weight, aggressive_logit, aggressive_prob, aggressive_gate

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        retrieval_embedding = self.encode(prefix)
        trend_prefix = _moving_average(prefix, kernel_size=min(max(self.trend_kernel, 1), max(prefix.shape[1], 1)))
        residual_prefix = prefix - trend_prefix
        dlinear_embedding = self.forecast_encoder["dlinear"](trend_prefix)
        tcn_embedding = self.forecast_encoder["tcn"](residual_prefix)
        prefix_abs = prefix.abs()
        residual_abs = residual_prefix.abs()
        outputs = {
            "embedding": retrieval_embedding,
            "retrieval_embedding": retrieval_embedding,
            "forecast_embedding_dlinear": dlinear_embedding,
            "forecast_embedding_tcn": tcn_embedding,
            "main_logit": self.main_head(retrieval_embedding).squeeze(-1),
            "aux_logit": self.aux_head(retrieval_embedding).squeeze(-1),
            "forecast_main_logit_dlinear": self.forecast_main_head["dlinear"](dlinear_embedding).squeeze(-1),
            "forecast_aux_logit_dlinear": self.forecast_aux_head["dlinear"](dlinear_embedding).squeeze(-1),
            "forecast_main_logit_tcn": self.forecast_main_head["tcn"](tcn_embedding).squeeze(-1),
            "forecast_aux_logit_tcn": self.forecast_aux_head["tcn"](tcn_embedding).squeeze(-1),
            "prefix_scale": prefix_abs.mean(dim=(1, 2)),
            "prefix_peak": prefix_abs.amax(dim=(1, 2)),
            "residual_ratio": residual_abs.mean(dim=(1, 2)) / prefix_abs.mean(dim=(1, 2)).clamp_min(1e-6),
            "residual_peak_ratio": residual_abs.amax(dim=(1, 2)) / prefix_abs.amax(dim=(1, 2)).clamp_min(1e-6),
        }
        outputs["shock_prior"] = 0.5 * (
            outputs["residual_ratio"].clamp(0.0, 1.0) + outputs["residual_peak_ratio"].clamp(0.0, 1.0)
        )
        self._cached_shock_prior = outputs["shock_prior"]
        outputs["final_main_logit"] = outputs["main_logit"]
        outputs["final_aux_logit"] = outputs["aux_logit"]
        return outputs


class CampaignMemAbstentionCalibrator(CampaignMemSelectorCalibrator):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_encoder_type: str,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.25,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=forecast_encoder_type,
            retrieval_encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_temperature=similarity_temperature,
            delta_scale=delta_scale,
        )
        selector_feature_dim = embedding_dim * 2 + 8
        self.main_abstain_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_abstain_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        forecast_main_prob = torch.sigmoid(outputs["forecast_main_logit"])
        forecast_aux_prob = torch.sigmoid(outputs["forecast_aux_logit"])
        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        forecast_main_conf = torch.abs(forecast_main_prob - 0.5) * 2.0
        forecast_aux_conf = torch.abs(forecast_aux_prob - 0.5) * 2.0
        main_disagreement = torch.abs(forecast_main_prob - base_main_prob)
        aux_disagreement = torch.abs(forecast_aux_prob - base_aux_prob)

        main_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        forecast_main_conf,
                        main_disagreement,
                        base_gate,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        forecast_aux_conf,
                        aux_disagreement,
                        base_gate,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        selector_main_logit = self.main_selector_head(main_features).squeeze(-1)
        selector_aux_logit = self.aux_selector_head(aux_features).squeeze(-1)
        # Abstention should learn when not to intervene, not reshape retrieval geometry.
        abstain_main_logit = self.main_abstain_head(main_features.detach()).squeeze(-1)
        abstain_aux_logit = self.aux_abstain_head(aux_features.detach()).squeeze(-1)

        selector_main_prob = torch.sigmoid(selector_main_logit)
        selector_aux_prob = torch.sigmoid(selector_aux_logit)
        abstain_main_prob = torch.sigmoid(abstain_main_logit)
        abstain_aux_prob = torch.sigmoid(abstain_aux_logit)

        main_correction_gate = selector_main_prob * (1.0 - abstain_main_prob)
        aux_correction_gate = selector_aux_prob * (1.0 - abstain_aux_prob)

        main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit"],
            correction_gate=main_correction_gate,
        )
        aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit"],
            correction_gate=aux_correction_gate,
        )

        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["selector_main_logit"] = selector_main_logit
        outputs["selector_aux_logit"] = selector_aux_logit
        outputs["abstain_main_logit"] = abstain_main_logit
        outputs["abstain_aux_logit"] = abstain_aux_logit
        outputs["abstain_main_prob"] = abstain_main_prob
        outputs["abstain_aux_prob"] = abstain_aux_prob
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (
            ((1.0 + abstain_main_prob) * main_delta.abs()).mean()
            + ((1.0 + abstain_aux_prob) * aux_delta.abs()).mean()
        )
        return outputs


class CampaignMemShiftAwareSelector(CampaignMemSelectorCalibrator):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_encoder_type: str,
        retrieval_encoder_type: str,
        hidden_dim: int,
        embedding_dim: int,
        top_k: int,
        similarity_temperature: float = 0.2,
        delta_scale: float = 0.25,
        shift_floor: float = 0.25,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=forecast_encoder_type,
            retrieval_encoder_type=retrieval_encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=top_k,
            similarity_temperature=similarity_temperature,
            delta_scale=delta_scale,
        )
        selector_feature_dim = embedding_dim * 2 + 10
        self.shift_floor = shift_floor
        self.main_shift_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_shift_head = nn.Sequential(
            nn.Linear(selector_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(prefix)
        prefix_abs = prefix.abs()
        outputs["prefix_scale"] = prefix_abs.mean(dim=(1, 2))
        outputs["prefix_peak"] = prefix_abs.amax(dim=(1, 2))
        return outputs

    def _apply_fusion(
        self,
        outputs: dict[str, torch.Tensor],
        retrieved: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base_gate = torch.sigmoid(self.gate_head(outputs["embedding"]).squeeze(-1))
        base_main_logit = base_gate * outputs["main_logit"] + (1 - base_gate) * retrieved["retrieved_main_logit"]
        base_aux_logit = base_gate * outputs["aux_logit"] + (1 - base_gate) * retrieved["retrieved_aux_logit"]

        forecast_main_prob = torch.sigmoid(outputs["forecast_main_logit"])
        forecast_aux_prob = torch.sigmoid(outputs["forecast_aux_logit"])
        base_main_prob = torch.sigmoid(base_main_logit)
        base_aux_prob = torch.sigmoid(base_aux_logit)
        forecast_main_conf = torch.abs(forecast_main_prob - 0.5) * 2.0
        forecast_aux_conf = torch.abs(forecast_aux_prob - 0.5) * 2.0
        main_disagreement = torch.abs(forecast_main_prob - base_main_prob)
        aux_disagreement = torch.abs(forecast_aux_prob - base_aux_prob)

        main_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        forecast_main_conf,
                        main_disagreement,
                        base_gate,
                        outputs["prefix_scale"],
                        outputs["prefix_peak"],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        aux_features = torch.cat(
            [
                outputs["forecast_embedding"],
                outputs["retrieval_embedding"],
                torch.stack(
                    [
                        retrieved["retrieval_score_mean"],
                        retrieved["retrieval_score_std"],
                        retrieved["retrieval_score_max"],
                        retrieved["retrieval_main_dispersion"],
                        retrieved["retrieval_aux_dispersion"],
                        forecast_aux_conf,
                        aux_disagreement,
                        base_gate,
                        outputs["prefix_scale"],
                        outputs["prefix_peak"],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        selector_main_logit = self.main_selector_head(main_features[..., :-2]).squeeze(-1)
        selector_aux_logit = self.aux_selector_head(aux_features[..., :-2]).squeeze(-1)
        shift_main_logit = self.main_shift_head(main_features.detach()).squeeze(-1)
        shift_aux_logit = self.aux_shift_head(aux_features.detach()).squeeze(-1)
        selector_main_prob = torch.sigmoid(selector_main_logit)
        selector_aux_prob = torch.sigmoid(selector_aux_logit)
        shift_main_prob = torch.sigmoid(shift_main_logit)
        shift_aux_prob = torch.sigmoid(shift_aux_logit)

        main_correction_gate = selector_main_prob * (self.shift_floor + (1.0 - self.shift_floor) * shift_main_prob)
        aux_correction_gate = selector_aux_prob * (self.shift_floor + (1.0 - self.shift_floor) * shift_aux_prob)

        main_delta = self._bounded_delta(
            base_logit=base_main_logit,
            forecast_logit=outputs["forecast_main_logit"],
            correction_gate=main_correction_gate,
        )
        aux_delta = self._bounded_delta(
            base_logit=base_aux_logit,
            forecast_logit=outputs["forecast_aux_logit"],
            correction_gate=aux_correction_gate,
        )

        outputs.update(retrieved)
        outputs["gate"] = base_gate
        outputs["calibration_gate"] = main_correction_gate
        outputs["selector_main_logit"] = selector_main_logit
        outputs["selector_aux_logit"] = selector_aux_logit
        outputs["shift_main_logit"] = shift_main_logit
        outputs["shift_aux_logit"] = shift_aux_logit
        outputs["shift_main_prob"] = shift_main_prob
        outputs["shift_aux_prob"] = shift_aux_prob
        outputs["base_main_logit"] = base_main_logit
        outputs["base_aux_logit"] = base_aux_logit
        outputs["final_main_logit"] = base_main_logit + main_delta
        outputs["final_aux_logit"] = base_aux_logit + aux_delta
        outputs["calibration_penalty"] = (main_delta.abs().mean() + aux_delta.abs().mean())
        return outputs


def pairwise_future_contrastive_loss(
    embedding: torch.Tensor,
    future_signature: torch.Tensor,
    prefix: torch.Tensor,
    use_hard_negatives: bool,
    margin: float = 0.25,
) -> torch.Tensor:
    batch_size = embedding.shape[0]
    if batch_size < 3:
        return embedding.new_tensor(0.0)
    similarity = F.normalize(embedding, dim=-1) @ F.normalize(embedding, dim=-1).T
    future_distance = torch.cdist(future_signature, future_signature, p=1)
    prefix_summary = summarize_prefix(prefix)
    prefix_similarity = F.normalize(prefix_summary, dim=-1) @ F.normalize(prefix_summary, dim=-1).T
    positive_mask = (future_distance <= future_distance.quantile(0.35)).float()
    negative_mask = (future_distance >= future_distance.quantile(0.7)).float()
    identity = torch.eye(batch_size, device=embedding.device)
    positive_mask = positive_mask * (1 - identity)
    negative_mask = negative_mask * (1 - identity)
    pos_loss = ((1.0 - similarity) * positive_mask).sum() / positive_mask.sum().clamp_min(1.0)
    neg_weight = 1.0 + prefix_similarity.clamp_min(0.0) if use_hard_negatives else 1.0
    neg_loss = (F.relu(similarity - margin) * negative_mask * neg_weight).sum() / negative_mask.sum().clamp_min(1.0)
    return pos_loss + neg_loss


def retrieval_utility_loss(
    retrieved_indices: torch.Tensor,
    future_signature: torch.Tensor,
) -> torch.Tensor:
    if retrieved_indices.numel() == 0:
        return future_signature.new_tensor(0.0)
    query = future_signature.unsqueeze(1)
    memory = future_signature[retrieved_indices]
    return torch.mean(torch.abs(query - memory))


def build_model(config: dict[str, Any], input_dim: int, seq_len: int) -> nn.Module:
    model_type = config["type"]
    encoder_type = config.get("encoder", "transformer")
    hidden_dim = int(config.get("hidden_dim", 128))
    embedding_dim = int(config.get("embedding_dim", hidden_dim))
    if model_type in {"tail_risk_linear"}:
        return ParametricForecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type="summary",
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
    if model_type in {
        "tcn",
        "transformer",
        "lstm",
        "dlinear",
        "patchtst",
        "itransformer",
        "timesnet",
        "tide",
        "tsmixer",
        "no_memory",
    }:
        return ParametricForecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
    if model_type in {"random_retrieval"}:
        return RandomRetrievalForecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_weighted=bool(config.get("similarity_weighted", False)),
            retrieval_aware_gate=bool(config.get("retrieval_aware_gate", False)),
            use_residual_fusion=bool(config.get("use_residual_fusion", False)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            residual_scale=float(config.get("residual_scale", 0.25)),
        )
    if model_type in {"prefix_retrieval", "campaign_mem"}:
        return RetrievalForecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_weighted=bool(config.get("similarity_weighted", False)),
            retrieval_aware_gate=bool(config.get("retrieval_aware_gate", False)),
            use_residual_fusion=bool(config.get("use_residual_fusion", False)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            residual_scale=float(config.get("residual_scale", 0.25)),
        )
    if model_type in {"campaign_mem_v2"}:
        return CampaignMemV2Forecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=str(config.get("forecast_encoder", encoder_type)),
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            residual_scale=float(config.get("residual_scale", 0.2)),
        )
    if model_type in {"campaign_mem_v3"}:
        return CampaignMemV3Forecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=str(config.get("forecast_encoder", encoder_type)),
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.35)),
        )
    if model_type in {"campaign_mem_v4"}:
        return CampaignMemV4Forecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=str(config.get("forecast_encoder", encoder_type)),
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.25)),
        )
    if model_type in {"campaign_mem_v5"}:
        return CampaignMemV5Forecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=str(config.get("forecast_encoder", encoder_type)),
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.25)),
        )
    if model_type in {"campaign_mem_structured"}:
        return CampaignMemStructuredCalibrator(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=str(config.get("forecast_encoder", encoder_type)),
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.25)),
        )
    if model_type in {"campaign_mem_selector"}:
        return CampaignMemSelectorCalibrator(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=str(config.get("forecast_encoder", encoder_type)),
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.25)),
        )
    if model_type in {"campaign_mem_dual_selector"}:
        return CampaignMemDualSelectorCalibrator(
            input_dim=input_dim,
            seq_len=seq_len,
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.2)),
        )
    if model_type in {"campaign_mem_modular"}:
        return CampaignMemModularCalibrator(
            input_dim=input_dim,
            seq_len=seq_len,
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.2)),
            use_tcn_expert=bool(config.get("use_tcn_expert", True)),
            use_lstm_expert=bool(config.get("use_lstm_expert", False)),
            use_softmax_expert_router=bool(config.get("use_softmax_expert_router", False)),
            use_abstention=bool(config.get("use_abstention", True)),
            use_uncertainty_gate=bool(config.get("use_uncertainty_gate", True)),
            use_shift_gate=bool(config.get("use_shift_gate", False)),
            use_aggressive_gate=bool(config.get("use_aggressive_gate", False)),
            aggressive_route_on_delta=bool(config.get("aggressive_route_on_delta", False)),
            uncertainty_gate_floor=float(config.get("uncertainty_gate_floor", 0.35)),
            selector_agreement_floor=float(config.get("selector_agreement_floor", 0.15)),
            shift_floor=float(config.get("shift_floor", 0.25)),
            aggressive_gate_floor=float(config.get("aggressive_gate_floor", 0.05)),
            abstention_detach=bool(config.get("abstention_detach", True)),
            uncertainty_detach=bool(config.get("uncertainty_detach", True)),
            shift_detach=bool(config.get("shift_detach", True)),
            aggressive_detach=bool(config.get("aggressive_detach", True)),
            base_gate_override=(
                None if config.get("base_gate_override") is None else float(config["base_gate_override"])
            ),
            calibration_transform=str(config.get("calibration_transform", "bounded_tanh")),
        )
    if model_type in {"campaign_mem_regime_router"}:
        return CampaignMemRegimeRouterCalibrator(
            input_dim=input_dim,
            seq_len=seq_len,
            retrieval_encoder_type=str(config.get("retrieval_encoder", "itransformer")),
            shock_encoder_type=str(config.get("shock_encoder", "patchtst")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.12)),
            trend_kernel=int(config.get("trend_kernel", 5)),
            use_abstention=bool(config.get("use_abstention", True)),
            use_shift_gate=bool(config.get("use_shift_gate", True)),
            use_aggressive_gate=bool(config.get("use_aggressive_gate", True)),
            selector_agreement_floor=float(config.get("selector_agreement_floor", 0.2)),
            shift_floor=float(config.get("shift_floor", 0.35)),
            shock_gate_floor=float(config.get("shock_gate_floor", 0.1)),
            shock_prior_scale=float(config.get("shock_prior_scale", 0.6)),
            abstention_detach=bool(config.get("abstention_detach", True)),
            shift_detach=bool(config.get("shift_detach", True)),
        )
    if model_type in {"campaign_mem_decomp_modular"}:
        return CampaignMemDecompModularCalibrator(
            input_dim=input_dim,
            seq_len=seq_len,
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            stable_encoder_type=str(config.get("stable_encoder", "dlinear")),
            shock_encoder_type=str(config.get("shock_encoder", "timesnet")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.12)),
            trend_kernel=int(config.get("trend_kernel", 5)),
            use_abstention=bool(config.get("use_abstention", True)),
            use_uncertainty_gate=bool(config.get("use_uncertainty_gate", False)),
            use_shift_gate=bool(config.get("use_shift_gate", True)),
            use_aggressive_gate=bool(config.get("use_aggressive_gate", True)),
            aggressive_route_on_delta=bool(config.get("aggressive_route_on_delta", True)),
            uncertainty_gate_floor=float(config.get("uncertainty_gate_floor", 0.35)),
            selector_agreement_floor=float(config.get("selector_agreement_floor", 0.2)),
            shift_floor=float(config.get("shift_floor", 0.35)),
            aggressive_gate_floor=float(config.get("aggressive_gate_floor", 0.1)),
            decomp_prior_mix=float(config.get("decomp_prior_mix", 0.0)),
            decomp_prior_floor=float(config.get("decomp_prior_floor", 0.1)),
            abstention_detach=bool(config.get("abstention_detach", True)),
            uncertainty_detach=bool(config.get("uncertainty_detach", True)),
            shift_detach=bool(config.get("shift_detach", True)),
            aggressive_detach=bool(config.get("aggressive_detach", True)),
            base_gate_override=(
                None if config.get("base_gate_override") is None else float(config["base_gate_override"])
            ),
            calibration_transform=str(config.get("calibration_transform", "bounded_tanh")),
        )
    if model_type in {"campaign_mem_final"}:
        return CampaignMemDecompModularCalibrator(
            input_dim=input_dim,
            seq_len=seq_len,
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            stable_encoder_type=str(config.get("stable_encoder", "dlinear")),
            shock_encoder_type=str(config.get("shock_encoder", "patchtst")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.15)),
            delta_scale=float(config.get("delta_scale", 0.12)),
            trend_kernel=int(config.get("trend_kernel", 5)),
            use_abstention=bool(config.get("use_abstention", True)),
            use_uncertainty_gate=bool(config.get("use_uncertainty_gate", False)),
            use_shift_gate=bool(config.get("use_shift_gate", True)),
            use_aggressive_gate=bool(config.get("use_aggressive_gate", True)),
            aggressive_route_on_delta=bool(config.get("aggressive_route_on_delta", True)),
            uncertainty_gate_floor=float(config.get("uncertainty_gate_floor", 0.35)),
            selector_agreement_floor=float(config.get("selector_agreement_floor", 0.2)),
            shift_floor=float(config.get("shift_floor", 0.35)),
            aggressive_gate_floor=float(config.get("aggressive_gate_floor", 0.15)),
            decomp_prior_mix=float(config.get("decomp_prior_mix", 0.0)),
            decomp_prior_floor=float(config.get("decomp_prior_floor", 0.1)),
            abstention_detach=bool(config.get("abstention_detach", True)),
            uncertainty_detach=bool(config.get("uncertainty_detach", True)),
            shift_detach=bool(config.get("shift_detach", True)),
            aggressive_detach=bool(config.get("aggressive_detach", True)),
            base_gate_override=(
                None if config.get("base_gate_override") is None else float(config["base_gate_override"])
            ),
            calibration_transform=str(config.get("calibration_transform", "bounded_tanh")),
        )
    if model_type in {"campaign_mem_abstain"}:
        return CampaignMemAbstentionCalibrator(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=str(config.get("forecast_encoder", encoder_type)),
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.25)),
        )
    if model_type in {"campaign_mem_shift_selector"}:
        return CampaignMemShiftAwareSelector(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_encoder_type=str(config.get("forecast_encoder", encoder_type)),
            retrieval_encoder_type=str(config.get("retrieval_encoder", "transformer")),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            top_k=int(config.get("top_k", 5)),
            similarity_temperature=float(config.get("similarity_temperature", 0.2)),
            delta_scale=float(config.get("delta_scale", 0.25)),
            shift_floor=float(config.get("shift_floor", 0.25)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")
