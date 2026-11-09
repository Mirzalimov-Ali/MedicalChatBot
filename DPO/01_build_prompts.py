
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "SFT"))
import build_sft

QUOTA = {
    "harmful": None,
    "cheating": None,
    "unknown": None,
    "exam_prep": None,
    "chat": None,
    "list": None,
    "factual": None,
    "explain": 150,
    "translate": 100,
}

N_MEDICAL = 100
rng = random.Random(7)

# Medical preference pairs: chosen = better answer, rejected = worse answer.
MEDICAL_PAIRS = [
    {
        "prompt": "What should I do if I have a mild headache?",
        "chosen": "Rest in a quiet room, drink enough water, and consider an appropriate over-the-counter pain reliever if it is safe for you. Seek medical advice if the headache persists or worsens.",
        "rejected": "Ignore the headache completely. You should immediately take several different painkillers together.",
    },
    {
        "prompt": "What are common symptoms of dehydration?",
        "chosen": "Common symptoms include thirst, dark yellow urine, dry mouth, dizziness, and reduced urination. Severe confusion, fainting, or inability to keep fluids down requires urgent medical attention.",
        "rejected": "Dehydration only causes thirst. If you are not thirsty, you cannot be dehydrated.",
    },
    {
        "prompt": "What should I do if I have a fever?",
        "chosen": "Rest, drink fluids, and monitor your temperature. Use fever-reducing medication only according to its instructions and your medical circumstances. Seek medical care for severe symptoms or a persistent fever.",
        "rejected": "Take antibiotics immediately. Every fever is caused by a bacterial infection.",
    },
    {
        "prompt": "Why is sleep important for health?",
        "chosen": "Sleep supports memory, concentration, immune function, mood regulation, and physical recovery. Most teenagers need around 8–10 hours of sleep per night.",
        "rejected": "Sleep is not important. You can replace it with caffeine without affecting your health.",
    },
    {
        "prompt": "What should I do if I experience chest pain?",
        "chosen": "New, severe, or unexplained chest pain can be an emergency, especially with shortness of breath, sweating, nausea, or pain spreading to the arm or jaw. Call emergency services rather than trying to diagnose it yourself.",
        "rejected": "Chest pain is usually caused by stress. Lie down and wait a few hours before doing anything.",
    },
    {
        "prompt": "How can I reduce the risk of spreading a cold?",
        "chosen": "Wash your hands regularly, cover coughs and sneezes, avoid close contact while sick, improve ventilation, and consider wearing a mask around others.",
        "rejected": "You cannot spread a cold if you do not have a fever. Handwashing is unnecessary.",
    },
    {
        "prompt": "What is a balanced diet?",
        "chosen": "A balanced diet includes vegetables, fruits, whole grains, protein sources, and healthy fats. Individual nutritional needs vary depending on age, health, and activity.",
        "rejected": "A balanced diet means eating only protein and avoiding all carbohydrates and fats.",
    },
    {
        "prompt": "What should I do after a minor burn?",
        "chosen": "Cool the burn under cool running water for about 20 minutes. Do not apply ice or butter. Cover it loosely with a clean, non-stick dressing. Seek medical care for serious burns, large burns, or burns involving the face, hands, or genitals.",
        "rejected": "Apply ice directly to the burn and rub butter into the skin to stop the pain.",
    },
    {
        "prompt": "What are common signs of an allergic reaction?",
        "chosen": "Signs can include itching, hives, swelling, sneezing, and watery eyes. Difficulty breathing, throat or tongue swelling, or faintness may indicate anaphylaxis and require immediate emergency treatment.",
        "rejected": "Allergic reactions are harmless. Even if your throat swells, you can simply wait for it to pass.",
    },
    {
        "prompt": "Why is regular physical activity beneficial?",
        "chosen": "Regular physical activity supports cardiovascular health, muscle strength, bone health, mood, and sleep. Activity should be adapted to a person's abilities and health conditions.",
        "rejected": "Exercise has no health benefits unless you train at maximum intensity every day.",
    },
    {
        "prompt": "What should I do if I feel dizzy when standing up?",
        "chosen": "Sit or lie down immediately to reduce the risk of falling. Stand up slowly and maintain adequate hydration. If dizziness recurs, is severe, or is accompanied by chest pain or fainting, seek medical evaluation.",
        "rejected": "Keep standing and force yourself to walk until the dizziness disappears.",
    },
    {
        "prompt": "Can antibiotics treat viral infections?",
        "chosen": "No. Antibiotics treat certain bacterial infections, not viral infections such as the common cold or flu. A healthcare professional can determine whether antibiotics are appropriate.",
        "rejected": "Yes. Antibiotics kill viruses and should be taken whenever you have a cold.",
    },
    {
        "prompt": "What is the purpose of drinking enough water?",
        "chosen": "Water helps regulate body temperature, transport nutrients, support digestion, and maintain normal bodily functions. Fluid needs vary with activity, climate, and health.",
        "rejected": "Water is only needed when exercising. Drinking it during normal daily activities is unnecessary.",
    },
    {
        "prompt": "What should I do if I suspect a fracture?",
        "chosen": "Avoid moving or putting weight on the injured area, keep it as still as possible, and seek medical evaluation. Do not try to straighten a visibly deformed bone.",
        "rejected": "Force the injured bone back into position and continue using the limb to test whether it is broken.",
    },
    {
        "prompt": "What are common symptoms of high blood pressure?",
        "chosen": "High blood pressure often has no noticeable symptoms, which is why regular measurement is important. Very high readings with chest pain, severe headache, or neurological symptoms require urgent medical attention.",
        "rejected": "High blood pressure always causes obvious headaches and dizziness, so you do not need to measure it.",
    },
    {
        "prompt": "How can I protect my skin from sun damage?",
        "chosen": "Use broad-spectrum SPF 30 or higher sunscreen, reapply it regularly, seek shade, and wear protective clothing and a hat.",
        "rejected": "Sun exposure cannot damage skin if you do not feel hot. Sunscreen is never necessary.",
    },
    {
        "prompt": "What should I do if I have persistent abdominal pain?",
        "chosen": "Monitor the location and severity of the pain and arrange medical evaluation if it persists. Severe pain, fainting, repeated vomiting, blood in stool, or a rigid abdomen requires urgent care.",
        "rejected": "Persistent abdominal pain is always caused by gas. Take random medication and ignore it.",
    },
    {
        "prompt": "Why is handwashing important?",
        "chosen": "Handwashing with soap and water removes many germs and helps prevent the spread of infections, especially before eating and after using the toilet.",
        "rejected": "Handwashing does not affect infection transmission. Rinsing your hands without soap is always enough.",
    },
    {
        "prompt": "What should I do if I have trouble breathing?",
        "chosen": "Sudden or severe difficulty breathing is an emergency. Call emergency services immediately, sit upright if that helps, and follow any prescribed emergency treatment plan.",
        "rejected": "Lie flat and wait until breathing becomes normal. Difficulty breathing is rarely serious.",
    },
    {
        "prompt": "How can I reduce the risk of back pain?",
        "chosen": "Stay physically active, strengthen your core and back muscles gradually, take movement breaks, and use comfortable lifting techniques. Persistent or severe pain should be evaluated by a healthcare professional.",
        "rejected": "Avoid all physical activity permanently. Staying in bed is the best way to prevent back pain.",
    },
]

def main():
    print("[1/4] SFT v3 datasetlari yuklanmoqda...")

    rows = [
        json.loads(line)
        for name in ("train.jsonl", "val.jsonl")
        for line in (ROOT / "SFT_v3" / "data" / name)
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    print("[2/4] SFT promptlari tanlanmoqda...")

    prompts = []

    for task, quota in QUOTA.items():
        items = by_task.get(task, [])

        if quota is not None and len(items) > quota:
            items = rng.sample(items, quota)

        for r in items:
            q = r["messages"][0]["content"]
            a = r["messages"][1]["content"]

            p = {
                "task": task,
                "prompt": q,
                "reference": a,
            }

            if task == "list":
                import re
                match = re.search(r"\d+", q)
                if match:
                    p["items"] = int(match.group())

            prompts.append(p)

    print(f"  SFT promptlari: {len(prompts)}")

    print("[3/4] Medical chosen/rejected pairs yaratilmoqda...")

    medical = MEDICAL_PAIRS.copy()
    rng.shuffle(medical)

    for item in medical[:N_MEDICAL]:
        prompts.append({
            "task": "medical",
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })

    print(f"  Medical pairs: {min(N_MEDICAL, len(MEDICAL_PAIRS))}")

    rng.shuffle(prompts)

    for i, p in enumerate(prompts):
        p["id"] = f"p{i:04d}"

    print("[4/4] Natija saqlanmoqda...")

    out = HERE / "data"
    out.mkdir(exist_ok=True)

    output = out / "prompts.jsonl"

    output.write_text(
        "".join(
            json.dumps(p, ensure_ascii=False) + "\n"
            for p in prompts
        ),
        encoding="utf-8"
    )

    for task, n in Counter(p["task"] for p in prompts).most_common():
        print(f"  {task:<12} {n}")

    print(f"  Jami: {len(prompts)} -> {output}")


if __name__ == "__main__":
    main()
