import torch
import numpy as np


class ConsistencyScorer:

    def __init__(self, model, tokenizer, steering_manager):
        self.model     = model
        self.tokenizer = tokenizer
        self.manager   = steering_manager

    def score(self, layer_range):
        print(f"\n[*] Phase 2: Computing Consistency Scores for {len(layer_range)} layers "
              f"(using cached raw_diffs, no extra forward pass)...")

        if not self.manager.raw_diffs:
            raise RuntimeError("raw_diffs is empty. Please run SteeringManager.compute_all_vectors() first.")

        EPS               = 1e-8
        snr_scores        = {}
        all_layer_indices = sorted(layer_range)
        candidate_layers  = set(all_layer_indices[0:32])

        for l in layer_range:
            if l not in self.manager.raw_diffs or l not in self.manager.steering_vectors:
                snr_scores[l] = 0.0
                continue

            all_diffs = self.manager.raw_diffs[l].float()
            u         = self.manager.steering_vectors[l].cpu().float()

            if l in self.manager.dev_weights:
                layer_w = self.manager.dev_weights[l].float()
                w_sum   = layer_w.sum().clamp(min=EPS)
                w_norm  = layer_w / w_sum
            else:
                N      = all_diffs.shape[0]
                w_norm = torch.ones(N, dtype=torch.float32) / N

            proj      = (all_diffs * u.unsqueeze(0)).sum(dim=-1)
            signal    = (proj * w_norm.view(-1, 1)).sum(dim=0)
            noise_var = (w_norm.view(-1, 1) *
                         (proj - signal.unsqueeze(0)) ** 2
                         ).sum(dim=0)
            noise         = noise_var.sqrt()
            per_token_snr = signal / (noise + EPS)
            snr_scores[l] = per_token_snr.mean().item()

        sorted_all = sorted(snr_scores.items(), key=lambda x: x[1], reverse=True)
        print("\n[*] Projection-SNR Scores (Top 15 layers, all):")
        for l, s in sorted_all[:15]:
            tag = "  [excluded: out of candidate range]" if l not in candidate_layers else ""
            print(f"  Layer {l:02d}: SNR = {s:.4f}{tag}")

        sorted_candidates = [(l, s) for l, s in sorted_all if l in candidate_layers]
        if not sorted_candidates:
            print("[Warning] No candidates in restricted range, falling back to all layers.")
            sorted_candidates = sorted_all

        K_MIN, K_MAX = 1, 8
        EPS_GAP      = 1e-8

        candidate_scores = [s for _, s in sorted_candidates]
        M = len(candidate_scores)

        if M <= 2:
            top_k    = max(K_MIN, min(M, K_MAX))
            gap_info = "N/A (too few candidates)"
        else:
            positive_end = M
            for idx, sc in enumerate(candidate_scores):
                if sc <= 0.0:
                    positive_end = idx
                    break
            search_len = max(2, min(positive_end, K_MAX + 1))

            rel_gaps = []
            for i in range(search_len - 1):
                s_cur  = candidate_scores[i]
                s_next = candidate_scores[i + 1]
                rel_gaps.append((s_cur - s_next) / (abs(s_cur) + EPS_GAP))

            best_gap_idx = int(np.argmax(rel_gaps))
            top_k        = best_gap_idx + 1
            top_k        = max(K_MIN, min(top_k, K_MAX))
            gap_info     = (f"max rel_gap={rel_gaps[best_gap_idx]:.4f} "
                            f"at position {best_gap_idx} "
                            f"(score {candidate_scores[best_gap_idx]:.4f}"
                            f" → {candidate_scores[best_gap_idx+1]:.4f})")

        best_layers = [l for l, _ in sorted_candidates[:top_k]]

        self.manager.best_layers = best_layers
        print(f"\n[*] Adaptive top_k = {top_k}  |  Gap info: {gap_info}")
        print(f"[*] Top-{top_k} Intervention Layers selected: {best_layers}")
        for l in best_layers:
            print(f"    Layer {l:02d}: SNR = {snr_scores[l]:.4f}")

        return snr_scores
