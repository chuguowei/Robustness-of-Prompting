import torch
import numpy as np
from tqdm import tqdm

from config import CHOICE_DATASETS, QUESTION_DATASETS, YESNO_DATASETS
from evaluation import (
    extract_answer_chose_ABCDE,
    extract_answer_QA,
    extract_answer_yesno,
    calculate_accuracy_num,
)


class SteeringManager:

    def __init__(self, model, tokenizer):
        self.model              = model
        self.tokenizer          = tokenizer
        self.steering_vectors   = {}
        self.layer_norms        = {}
        self.layer_norm_upper   = {}
        self.layer_mu_clean     = {}
        self.layer_mu_perturbed = {}
        self.raw_diffs          = {}
        self.dev_weights        = {}
        self.best_layers        = []
        self.hooks              = []
        self.trigger_phrase     = "\nTherefore, the answer is: "

    def _get_hidden_states(self, text, layers):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        states = {}
        for layer_idx in layers:
            idx = layer_idx + 1
            if idx < len(outputs.hidden_states):
                states[layer_idx] = outputs.hidden_states[idx][0]
        return states

    def _decode_and_check_correct(self, q_dirty, item, dataset):
        inputs = self.tokenizer(q_dirty, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        full_resp = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        trigger   = self.trigger_phrase.strip()
        if trigger in full_resp:
            gen_resp = full_resp.split(trigger)[-1].strip()
        else:
            prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            gen_resp    = full_resp[len(prompt_text):].strip()

        original_answer   = item.get('original_answer', '').replace(',', '')
        original_question = item.get('original_question', '')

        if dataset in CHOICE_DATASETS:
            pred = extract_answer_chose_ABCDE(gen_resp, original_question)
            return pred == original_answer
        elif dataset in QUESTION_DATASETS:
            pred = extract_answer_QA(gen_resp)
            return calculate_accuracy_num([pred], [original_answer]) > 0.99
        elif dataset in YESNO_DATASETS:
            pred = extract_answer_yesno(gen_resp)
            return pred == original_answer

    def compute_all_vectors(self, data_samples, layer_range, dataset=None):
        print(f"[*] Phase 1: Computing Steering Vectors for {len(layer_range)} layers "
              f"using {len(data_samples)} dev samples...")

        pairs_with_items = []
        for item in data_samples:
            if 'original_question' in item and 'rewritten_question' in item:
                q_clean = item['original_question']  + self.trigger_phrase
                q_dirty = item['rewritten_question'] + self.trigger_phrase
                pairs_with_items.append((q_clean, q_dirty, item))

        trigger_tokens = self.tokenizer(self.trigger_phrase, add_special_tokens=False)['input_ids']
        trigger_len    = len(trigger_tokens)

        W_WRONG = 1.0
        W_RIGHT = 0.1

        sample_weights = []
        if dataset is not None:
            for q_clean, q_dirty, item in tqdm(pairs_with_items,
                                               desc="Phase 1 - Pass 0: Weighting samples"):
                correct = self._decode_and_check_correct(q_dirty, item, dataset)
                sample_weights.append(W_RIGHT if correct else W_WRONG)
            n_wrong = sum(1 for w in sample_weights if w == W_WRONG)
            n_right = len(sample_weights) - n_wrong
            print(f"[*] Sample weights: {n_wrong} harmful (w={W_WRONG}), "
                  f"{n_right} harmless (w={W_RIGHT})")
        else:
            sample_weights = [1.0] * len(pairs_with_items)
            print("[*] No dataset specified; using uniform weights.")

        diff_collectors   = {l: [] for l in layer_range}
        clean_collectors  = {l: [] for l in layer_range}
        dirty_collectors  = {l: [] for l in layer_range}
        weight_collectors = {l: [] for l in layer_range}

        for idx, (q_clean, q_dirty, item) in enumerate(
                tqdm(pairs_with_items, desc="Phase 1 - Pass 1: Collecting hidden states")):
            states_clean = self._get_hidden_states(q_clean, layer_range)
            states_dirty = self._get_hidden_states(q_dirty, layer_range)

            for layer in layer_range:
                if layer not in states_clean or layer not in states_dirty:
                    continue
                vec_c = states_clean[layer][-trigger_len:, :]
                vec_d = states_dirty[layer][-trigger_len:, :]
                if vec_c.shape == vec_d.shape:
                    diff_collectors[layer].append((vec_c - vec_d).cpu())
                    clean_collectors[layer].append(vec_c.cpu())
                    dirty_collectors[layer].append(vec_d.cpu())
                    weight_collectors[layer].append(sample_weights[idx])

        EPS   = 1e-8
        count = 0
        for layer in layer_range:
            if not diff_collectors[layer]:
                continue

            all_diffs  = torch.stack(diff_collectors[layer])
            all_cleans = torch.stack(clean_collectors[layer])
            all_dirtys = torch.stack(dirty_collectors[layer])

            layer_w = torch.tensor(weight_collectors[layer], dtype=torch.float32)
            w_sum   = layer_w.sum().clamp(min=EPS)
            w_norm  = layer_w / w_sum

            v_raw   = (all_diffs.float() * w_norm.view(-1, 1, 1)).sum(dim=0)
            v_norms = torch.norm(v_raw, dim=-1)
            u       = v_raw / (v_norms.unsqueeze(-1) + EPS)

            clean_projs  = (all_cleans.float() * u.unsqueeze(0)).sum(dim=-1)
            dirty_projs  = (all_dirtys.float() * u.unsqueeze(0)).sum(dim=-1)
            mu_clean     = (clean_projs * w_norm.view(-1, 1)).sum(dim=0)
            mu_perturbed = (dirty_projs * w_norm.view(-1, 1)).sum(dim=0)

            proj_on_u   = (all_diffs.float() * u.unsqueeze(0)).sum(dim=-1)
            proj_mean_w = (proj_on_u * w_norm.view(-1, 1)).sum(dim=0)
            proj_var_w  = (w_norm.view(-1, 1) *
                           (proj_on_u - proj_mean_w.unsqueeze(0)) ** 2
                           ).sum(dim=0)
            proj_std_w  = proj_var_w.sqrt()
            norm_upper  = proj_mean_w + 2.0 * proj_std_w
            norm_upper  = torch.clamp(norm_upper, min=v_norms)

            self.steering_vectors[layer]   = u.to(self.model.device)
            self.layer_norms[layer]        = v_norms.to(self.model.device)
            self.layer_norm_upper[layer]   = norm_upper.to(self.model.device)
            self.layer_mu_clean[layer]     = mu_clean.to(self.model.device)
            self.layer_mu_perturbed[layer] = mu_perturbed.to(self.model.device)
            self.raw_diffs[layer]          = all_diffs
            self.dev_weights[layer]        = layer_w

            count += 1

        print(f"[*] Phase 1 Done. Computed vectors for {count} layers.")

    def register_inference_hooks(self, beta=1.0, beta_log=None):
        self.remove_hooks()

        if not self.best_layers:
            print("[Warning] best_layers not set. Please run ConsistencyScorer.score() first.")
            return

        trigger_tokens = self.tokenizer(self.trigger_phrase, add_special_tokens=False)['input_ids']
        trigger_len    = len(trigger_tokens)

        def make_hook(layer_idx, u, mu_clean, v_norms, norm_upper):
            def hook_fn(module, input, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                seq_len = hidden_states.shape[1]

                if seq_len >= trigger_len:
                    current_acts = hidden_states[:, -trigger_len:, :]

                    u_dev   = u.to(hidden_states.device).to(hidden_states.dtype)
                    mu_c    = mu_clean.to(hidden_states.device).to(hidden_states.dtype)
                    n_upper = norm_upper.to(hidden_states.device).to(hidden_states.dtype)

                    p_d          = (current_acts * u_dev.unsqueeze(0)).sum(dim=-1)
                    delta_p      = mu_c.unsqueeze(0) - p_d
                    zero_tensor  = torch.tensor(0.0, device=delta_p.device, dtype=delta_p.dtype)
                    beta_dynamic = torch.clamp(delta_p, min=zero_tensor, max=n_upper.unsqueeze(0))

                    perturbation = beta_dynamic.unsqueeze(-1) * u_dev.unsqueeze(0)
                    hidden_states[:, -trigger_len:, :] = current_acts + perturbation

                    if beta_log is not None and layer_idx not in beta_log:
                        beta_log[layer_idx] = beta_dynamic[0].mean().item()

                return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states
            return hook_fn

        registered = []
        for layer in self.best_layers:
            if layer not in self.steering_vectors:
                print(f"[Warning] No steering vector for layer={layer}, skipping.")
                continue

            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                layer_module = self.model.model.layers[layer]
            elif hasattr(self.model, "layers"):
                layer_module = self.model.layers[layer]
            else:
                print(f"[Warning] Cannot locate module for layer={layer}, skipping.")
                continue

            hook_fn = make_hook(
                layer,
                self.steering_vectors[layer],
                self.layer_mu_clean[layer],
                self.layer_norms[layer],
                self.layer_norm_upper[layer]
            )
            self.hooks.append(layer_module.register_forward_hook(hook_fn))
            registered.append(layer)

        print(f"[*] Dynamic Clamp hooks registered on Layers {registered}.")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
