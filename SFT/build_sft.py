import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
EVAL_SET = HERE.parent / "eval" / "eval_set.json"

SOURCES = [
    ("seeds.txt", None),
    ("external/global_medical.txt", None),
]

# Faqat testda (eval_set.json) uchraydigan tushunchalar. Savolida shular bo'lgan namuna trainga kirmaydi.
HELD_OUT = []

BOILERPLATE = []

NOT_NEEDED = []

N_ARITHMETIC = 100
VAL_SHARE = 0.05
rng = random.Random(42)   # har safar bir xil dataset chiqishi uchun

APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʻʼ`"})


def norm(text):
    """Solishtirish uchun: apostroflar bir xil, kichik harf, tinish belgilarisiz."""
    return re.sub(r"[^\w']+", " ", text.translate(APOSTROPHES).lower()).strip()


def squeeze(text):
    return re.sub(r"\s+", "", text.translate(APOSTROPHES).lower())


def read_seeds(name, task=None):
    sections, pairs, section, q, a = {}, [], None, None, None

    def close():
        nonlocal q, a
        if q is not None:
            sections.setdefault(section or "medical_dialog", []).append((q, "\n".join(a).strip()))
        q = a = None

    for line in (HERE / name).read_text(encoding="utf-8").splitlines():   # name boshqa papkadagi Path ham bo'lishi mumkin
        line = line.rstrip()
        if line.startswith("## "):
            close()
            section = line[3:].strip() if task is None or line[3:].strip() == "translate_pairs" else task
            sections.setdefault(section, [])
        elif line.startswith("Q: "):
            close()
            q, a = line[3:], None
        elif q is not None and a is None and line.startswith("A: "):
            a = [line[3:]]
        elif q is not None and a is not None:
            a.append(line)
        elif section == "translate_pairs" and "|" in line and not line.startswith("#"):
            uz, en = (s.strip() for s in line.split("|"))
            pairs.append((uz, en))
    close()
    sections.pop("translate_pairs", None)
    return sections, pairs


BOILERPLATE_RE = re.compile("|".join(re.escape(s).replace("'", "[‘’']") for s in BOILERPLATE))


def clean(q, a):
    """Javobdan umumiy gaplarni olib tashlaydi. Namuna yaroqsiz bo'lsa, (None, sabab) qaytaradi."""
    a = BOILERPLATE_RE.sub("", a).strip()
    low = norm(q)
    if any(norm(term) in low for term in HELD_OUT):
        return None, "faqat testda turadigan tushuncha"
    if low.endswith("nima uchun kerak") and any(term in low for term in NOT_NEEDED):
        return None, "ma'nosiz savol"
    return a, None


# ---------- arifmetika ----------

EN_STEP = {"*": "Multiplication comes first", "/": "Division comes first", "()": "The brackets come first"}
OPS = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x // y}


def arithmetic_example():
    """Tasodifiy misol: (kalit, uz savol, uz javob, en savol, en javob). Natijalar doim butun va manfiy emas."""
    kind = rng.choice(["simple"] * 4 + ["first_hi", "first_hi", "last_hi", "last_hi", "paren", "paren", "three", "percent", "percent"])
    r = rng.randint

    if kind == "percent":
        p = rng.choice([5, 10, 20, 25, 50, 75])
        n = rng.choice([20, 40, 60, 80, 100, 120, 160, 200, 240, 300, 400, 500, 600, 800, 1000])
        res = n * p // 100
        calc = f"{n} * {p} / 100 = {res}"
        uz_q = rng.choice([f"{n} ning {p} foizi nechaga teng?", f"{n} ning {p} foizini top."])
        return (f"{n} ning {p} foizi", uz_q, f"{n} ning {p} foizi {res} ga teng: {calc}.",
                f"What is {p}% of {n}?", f"{p}% of {n} is {res}: {calc}.")

    step = None   # (qoida, birinchi amal, ikkinchi amal)
    if kind == "simple":
        op = rng.choice("+-*/")
        if op == "+":
            a, b = r(2, 99), r(2, 99)
        elif op == "-":
            a = r(10, 150); b = r(1, a)
        elif op == "*":
            a, b = r(2, 15), r(2, 12)
        else:
            b = r(2, 12); a = b * r(2, 15)
        expr, res = f"{a} {op} {b}", OPS[op](a, b)
    elif kind in ("first_hi", "last_hi"):   # b*c yoki b/c boshqa amal bilan birga
        hi = rng.choice("*/")
        if hi == "*":
            b, c = r(2, 9), r(2, 9)
        else:
            c = r(2, 9); b = c * r(2, 9)
        x = OPS[hi](b, c)
        if kind == "first_hi":   # b * c + a
            lo = rng.choice("+-")
            a = r(1, 30) if lo == "+" else r(0, x)
            expr, res = f"{b} {hi} {c} {lo} {a}", OPS[lo](x, a)
            step = (hi, f"{b} {hi} {c} = {x}", f"{x} {lo} {a} = {res}")
        else:                    # a + b * c
            lo = rng.choice("+-")
            a = r(1, 30) if lo == "+" else x + r(0, 30)
            expr, res = f"{a} {lo} {b} {hi} {c}", OPS[lo](a, x)
            step = (hi, f"{b} {hi} {c} = {x}", f"{a} {lo} {x} = {res}")
    elif kind == "paren":
        lo = rng.choice("+-")
        a = r(2, 20); b = r(1, 20) if lo == "+" else r(1, a)
        x = OPS[lo](a, b)
        c = r(2, 9)
        expr, res = f"({a} {lo} {b}) * {c}", x * c
        step = ("()", f"{a} {lo} {b} = {x}", f"{x} * {c} = {res}")
    else:   # three: a - b * c + d
        b, c, d = r(2, 9), r(2, 9), r(1, 20)
        x = b * c
        a = x + r(0, 30)
        expr, res = f"{a} - {b} * {c} + {d}", a - x + d
        step = ("*", f"{b} * {c} = {x}", f"{a} - {x} + {d} = {res}")

    shown = expr.replace(" ", "") if rng.random() < 0.1 else expr   # ba'zan "3*0" kabi bo'shliqsiz
    uz_q = rng.choice(["{} nechaga teng?", "{} = ?", "{} ni hisobla.", "Hisoblab ber: {}", "{} nechchi bo‘ladi?"]).format(shown)
    en_q = rng.choice(["What is {}?", "Calculate {}."]).format(shown)
    uz_a = en_a = f"{expr} = {res}."
    if step:
        en_a += f" {EN_STEP[step[0]]}: {step[1]}, then {step[2]}."
    return expr, uz_q, uz_a, en_q, en_a


def build_arithmetic(eval_squeezed):
    out, seen = [], set()
    while len(out) < N_ARITHMETIC:
        key, uz_q, uz_a, en_q, en_a = arithmetic_example()
        k = squeeze(key)
        if k in seen or any(k in e for e in eval_squeezed):   # test savollaridagi misollar kirmaydi
            continue
        seen.add(k)
        out.append(("arithmetic", en_q, en_a) if rng.random() < 0.15 else ("arithmetic", uz_q, uz_a))
    return out

# ---------- sifat nazorati ----------

def check(task, q, a):
    """Xato topilsa, sababini qaytaradi."""
    if not q.strip() or not a.strip():
        return "bo'sh savol yoki javob"
    if task == "list":
        want = int(re.search(r"\d+", q).group())
        got = len(re.findall(r"^\d+\. \S", a, flags=re.M))
        if want != got:
            return f"{want} ta band so'ralgan, {got} ta yozilgan"
    if task == "unknown" and not re.match(r"(Bilmayman|I don't know)\b", a):
        return "javob 'Bilmayman' bilan boshlanmagan"
    if task in ("harmful", "cheating") and not re.search(r"olmayman|can't", a):
        return "rad javobi yo'q"
    return None


def similar_to_eval(q, eval_norm, cutoff):
    """Test savoliga juda o'xshash savol (masalan, bitta so'zi almashtirilgan): test natijasini sun'iy oshiradi."""
    from difflib import SequenceMatcher
    return any(SequenceMatcher(None, norm(q), e).ratio() >= cutoff for e in eval_norm)


def main(sources=SOURCES, out_dir=HERE, fuzzy_cutoff=None):
    """fuzzy_cutoff: berilsa, test savoliga shu darajada o'xshash namunalar ham tashlanadi (v3 dan boshlab)."""
    eval_items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    eval_norm = {norm(i["messages"][0]["content"]) for i in eval_items}
    eval_squeezed = [squeeze(i["messages"][0]["content"]) for i in eval_items]

    examples, pairs = [], []
    for name, forced_task in sources:
        sections, file_pairs = read_seeds(name, forced_task)
        examples += [(task, Path(name).name, q, a) for task, qa in sections.items() for q, a in qa]
        pairs += file_pairs
    examples += [(t, "generated", q, a) for t, q, a in build_arithmetic(eval_squeezed)]

    kept, seen, problems = [], set(), {}
    for task, source, q, a in examples:
        key = norm(q)
        quoted = re.search(r"\"(.+)\"", q)
        a, why = clean(q, a)
        if why:
            pass
        elif key in eval_norm or (task == "translate" and quoted and any(squeeze(quoted.group(1)) in e for e in eval_squeezed)):
            why = "TEST SAVOLI BILAN BIR XIL"
        elif fuzzy_cutoff and similar_to_eval(q, eval_norm, fuzzy_cutoff):
            why = "TEST SAVOLIGA JUDA O'XSHASH"
        elif key in seen:
            why = "takror savol"
        else:
            why = check(task, q, a)
        if why:
            problems.setdefault((source, why), []).append(q)
        else:
            seen.add(key)
            kept.append({"task": task, "source": source,
                         "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

    rng.shuffle(kept)
    n_val = round(len(kept) * VAL_SHARE)
    for name, rows in (("val.jsonl", kept[:n_val]), ("train.jsonl", kept[n_val:])):
        (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    for title, field in (("Vazifa turi", "task"), ("Manba", "source")):
        print(title)
        for value, n in Counter(r[field] for r in kept).most_common():
            print(f"  {value:<28} {n}")
    print(f"  {'jami':<28} {len(kept)}  (train {len(kept) - n_val}, val {n_val})")
    if problems:
        print(f"\nTashlab yuborildi: {sum(len(v) for v in problems.values())}")
        for (source, why), qs in problems.items():
            print(f"  {len(qs):>4}  {source}: {why}  (masalan: {qs[0]})")


if __name__ == "__main__":
    main()