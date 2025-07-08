import json
import re

with open("medquad_qa_semantic_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

keyword_map = {
    "symptoms": "Symptoms",
    "treatment": "Treatment",
    "treatments": "Treatment",
    "diagnose": "Diagnosis",
    "diagnosis": "Diagnosis",
    "diagnosed": "Diagnosis"
}

def find_semantic_subgroup(question):
    question = question.lower()
    for keyword, subgroup in keyword_map.items():
        if re.search(rf"\b{keyword}\b", question):
            return subgroup
    return "General"

for entry in data:
    question = entry.get("question", "")
    entry["semantic_subgroup"] = find_semantic_subgroup(question)

with open("medquad.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Updated JSON saved as 'medquad.json'")
