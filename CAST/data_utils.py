import re
import json
from pathlib import Path
from config import CHOICE_DATASETS


def clean_rewritten_question(text):
    if not isinstance(text, str) or not text:
        return text
    answer_header_re = re.compile(r'\bA\w{0,20}\s*C\w{0,20}\s*[:：]', re.IGNORECASE)
    matches = list(answer_header_re.finditer(text))
    if not matches:
        return text
    if '?' in text:
        last_q_pos     = text.rfind('?')
        q_part         = text[:last_q_pos + 1]
        last_ans_start = matches[-1].start()
        answer_part    = text[last_ans_start:].strip()
        remainder_after_q = text[last_q_pos + 1:].strip()
        if remainder_after_q.startswith(answer_part) or remainder_after_q == answer_part:
            return text.strip()
        else:
            return (q_part + ' ' + answer_part).strip()
    else:
        if len(matches) == 1:
            return text
        else:
            first_ans_start = matches[0].start()
            prefix          = text[:first_ans_start].rstrip()
            last_ans_start  = matches[-1].start()
            answer_part     = text[last_ans_start:].strip()
            if prefix:
                return (prefix + ' ' + answer_part).strip()
            else:
                return answer_part


def preprocess_datasets_inplace(datasets, robust_types):
    print("=" * 40)
    print("[Pre-processing] Starting data preprocessing...")
    for dataset in datasets:
        file_path = f"data/data_{robust_types}/rewritten_{dataset}_1.json"
        p = Path(file_path)
        if not p.exists():
            continue
        if dataset not in CHOICE_DATASETS:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            modified_count = 0
            for item in data:
                if "rewritten_question" in item:
                    orig    = item["rewritten_question"]
                    cleaned = clean_rewritten_question(orig)
                    if cleaned != orig:
                        item["rewritten_question"] = cleaned
                        modified_count += 1
            if modified_count > 0:
                p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    print("[Pre-processing] Completed.")
    print("=" * 40)