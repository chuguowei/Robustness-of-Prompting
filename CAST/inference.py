import os
import json
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from config import CHOICE_DATASETS, QUESTION_DATASETS, YESNO_DATASETS
from evaluation import (
    extract_answer_chose_ABCDE,
    extract_answer_QA,
    extract_answer_yesno,
    calculate_accuracy_num,
    calculate_accuracy_choice,
)


def run_experiment(dataset, robust_types, model, tokenizer, steering_manager, eval_data):
    cudnn.deterministic = True
    print(f"[*] Formal Evaluation on {len(eval_data)} test samples "
          f"(best_layers={steering_manager.best_layers}).")

    beta_log = {}
    steering_manager.register_inference_hooks(beta_log=beta_log)

    predictions, answers, results = [], [], []
    trigger_phrase     = "\nTherefore, the answer is: "
    correct_count_rt   = 0
    processed_count_rt = 0
    acc_func           = None

    pbar = tqdm(enumerate(eval_data), total=len(eval_data),
                desc=f"Eval {dataset} (DynamicClamp)", unit="sample")

    for i, item in pbar:
        rewritten_question = item.get('rewritten_question', "")
        original_question  = item.get('original_question', "")
        if not rewritten_question:
            continue

        prompt    = rewritten_question + trigger_phrase
        inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        beta_log.clear()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                temperature=0.1,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        beta_dynamic_logged = {l: round(v, 6) for l, v in beta_log.items()} if beta_log else {}

        full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if trigger_phrase.strip() in full_response:
            gen_response = full_response.split(trigger_phrase.strip())[-1].strip()
        else:
            prompt_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            gen_response   = full_response[len(prompt_decoded):].strip()

        original_answer = item.get('original_answer', "").replace(",", "")
        is_correct      = False

        if dataset in CHOICE_DATASETS:
            pred     = extract_answer_chose_ABCDE(gen_response, original_question)
            acc_func = calculate_accuracy_choice
            if pred == original_answer:
                is_correct = True
        elif dataset in QUESTION_DATASETS:
            pred     = extract_answer_QA(gen_response)
            acc_func = calculate_accuracy_num
            if calculate_accuracy_num([pred], [original_answer]) > 0.99:
                is_correct = True
        elif dataset in YESNO_DATASETS:
            pred     = extract_answer_yesno(gen_response)
            acc_func = calculate_accuracy_choice
            if pred == original_answer:
                is_correct = True
        else:
            pred = gen_response[:16]

        if is_correct:
            correct_count_rt += 1
        processed_count_rt += 1
        current_acc = correct_count_rt / processed_count_rt if processed_count_rt > 0 else 0

        pbar.set_postfix({"Acc": f"{current_acc:.2%}",
                          "Pred": str(pred)[:8],
                          "GT": str(original_answer)[:8]})

        results.append({
            "id":           i,
            "prompt":       prompt,
            "response":     gen_response,
            "prediction":   pred,
            "ground_truth": original_answer,
            "beta_dynamic": beta_dynamic_logged
        })
        predictions.append(pred)
        answers.append(original_answer)

    steering_manager.remove_hooks()

    if acc_func is None:
        acc_func = calculate_accuracy_num
    accuracy = acc_func(predictions, answers)

    output_dir  = f'result/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    layers_str  = "-".join(str(l) for l in steering_manager.best_layers)
    output_path = (f'{output_dir}/CAS_Consistency_Clamp_{dataset}_{robust_types}'
                   f'_layers{layers_str}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[*] Results saved → {output_path}")
    return accuracy