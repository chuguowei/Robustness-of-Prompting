import re


def _parse_choices_from_question(original_question: str):
    choices = {}
    if not original_question:
        return choices
    pattern = re.compile(r'\(([A-E])\)\s*([^()]+?)(?=(\s*\([A-E]\)|$))', re.S)
    for m in pattern.finditer(original_question):
        label = m.group(1).strip()
        text  = m.group(2).strip()
        num_match = re.search(r'-?\d+(?:\.\d+)?', text)
        num   = num_match.group(0) if num_match else None
        choices[label] = {'text': text, 'num': num}
    return choices


def extract_answer_chose_ABCDE(response: str, original_question: str):
    if response is None:
        return ""
    lines   = [ln.strip() for ln in response.splitlines() if ln.strip()]
    lines   = lines[:5]
    choices = _parse_choices_from_question(original_question)
    letter_pattern = re.compile(
        r'(?<![A-Za-z0-9])(?:\(?\s*([A-F])\s*\)?|Answer\s*[:\-]?\s*([A-F])|\b[0-9]+\.\s*([A-F]))(?![A-Za-z0-9])')
    for line in lines:
        m = letter_pattern.search(line)
        if m:
            letter = next((g for g in m.groups() if g), None)
            if letter:
                return letter.upper()
    for label, info in choices.items():
        if info['text'] and info['text'] in response:
            return label.upper()
    return ""


def extract_answer_QA(response):
    if not response:
        return ""
    first_line = response.strip().split("\n")[0].replace(",", "")
    if "=" in first_line:
        rhs   = first_line.split('=')[-1].strip()
        match = re.search(r'-?\d+(?:/\d+)?(?:\.\d+)?', rhs)
        if match:
            return match.group(0)
    match = re.search(r'-?\d+(?:/\d+)?(?:\.\d+)?', first_line)
    if match:
        return match.group(0)
    match = re.search(r'-?\d+(?:/\d+)?(?:\.\d+)?', response.replace(",", ""))
    if match:
        return match.group(0)
    return ""


def extract_answer_yesno(response):
    pred = response.lower()
    pred = re.sub(r"\"|\'|\n|\.|\s|\:|\,", " ", pred)
    pred = pred.split(" ")
    pred = [i for i in pred if i in ("yes", "no")]
    if len(pred) == 0:
        return ""
    return pred[0]


def str_to_float(s: str):
    s = s.strip()
    if "/" in s:
        try:
            num, den = s.split("/")
            return float(num) / float(den)
        except Exception:
            return None
    else:
        try:
            return float(s)
        except Exception:
            return None


def calculate_accuracy_num(predictions, answers):
    correct, total = 0, 0
    for pred, ans in zip(predictions, answers):
        if not ans:
            continue
        total += 1
        if pred == ans:
            correct += 1
            continue
        pred_val = str_to_float(pred)
        ans_val  = str_to_float(ans)
        if pred_val is not None and ans_val is not None:
            if abs(pred_val - ans_val) < 0.001:
                correct += 1
    return correct / total if total > 0 else 0


def calculate_accuracy_choice(predictions, answers):
    correct, total = 0, 0
    for pred, ans in zip(predictions, answers):
        if not ans:
            continue
        total += 1
        if pred == ans:
            correct += 1
    return correct / total if total > 0 else 0
