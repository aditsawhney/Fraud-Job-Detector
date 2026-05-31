"""
generate_dataset.py - Synthetic Indian job fraud dataset generator.

Uses Groq API to generate realistic fraudulent and legitimate job postings
grounded in the Indian fraud taxonomy. Supports checkpoint recovery.

Usage:
    python data_pipeline/generate_dataset.py [--n_fraud N] [--n_real N] [--output DIR]
"""

import json
import random
import time
import argparse
import csv
import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent))
try:
    from indian_fraud_taxonomy import SCAM_ARCHETYPES, CROSS_CUTTING_SIGNALS
except ImportError:
    print("Error: Could not import 'indian_fraud_taxonomy'. Ensure it's in the same directory.")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment or .env file.")
    client = Groq(api_key=api_key)
except ImportError:
    print("Install the Groq SDK: pip install groq")
    sys.exit(1)

MODEL = "llama-3.1-8b-instant"


class JobPosting(BaseModel):
    title:              str       = Field(description="Job title")
    company:            str       = Field(description="Company name")
    location:           str       = Field(description="City, state in India")
    salary:             str       = Field(description="Salary range or description")
    description:        str       = Field(description="Full job description (150-400 words)")
    requirements:       str       = Field(description="Requirements section")
    contact:            str       = Field(description="Contact details")
    archetype:          str       = Field(description="Archetype name you were given or 'legitimate'")
    fraud_signals_used: list[str] = Field(description="List of specific fraud signals included, empty if real")


FRAUD_SYSTEM_PROMPT = """You are a dataset generator for an ML research project on job fraud detection.
Your task is to generate realistic-looking FRAUDULENT Indian job postings.

These must look convincing enough to fool a real job seeker.
They should reflect how actual Indian job scams are written: sometimes with minor grammar
errors, mix of Hindi/English terms (Hinglish), real-sounding company names, etc.
Weave the provided signals organically into the text layout."""

REAL_SYSTEM_PROMPT = """You are a dataset generator for an ML research project on job fraud detection.
Your task is to generate realistic LEGITIMATE Indian job postings.

These should look like postings you'd see on Naukri.com or LinkedIn India from real companies.
They should be specific, professional, with realistic requirements and compensation.
Do NOT include any fee requests, suspicious WhatsApp contact, or artificial urgency language."""


def build_fraud_prompt(archetype_name: str, archetype: dict) -> str:
    signals = random.sample(
        archetype["linguistic_signals"],
        k=min(4, len(archetype["linguistic_signals"]))
    )
    cross_category = random.choice(list(CROSS_CUTTING_SIGNALS.keys()))
    cross_signals = random.sample(
        CROSS_CUTTING_SIGNALS[cross_category],
        k=min(2, len(CROSS_CUTTING_SIGNALS[cross_category]))
    )
    role = random.choice(archetype["roles"])

    return f"""Generate a fraudulent Indian job posting for the "{archetype_name}" scam type.

Scam description: {archetype['description']}
Target victim: {archetype['target']}
Role to advertise: {role}
Typical salary range used: {archetype['salary_range']}

You MUST naturally include these linguistic signals (weave them in dynamically):
Archetype signals: {', '.join(signals)}
Cross-cutting signals: {', '.join(cross_signals)}

Structural requirements:
- Contact should be: {random.choice(archetype['structural_signals'])}
- Location: pick a realistic Indian city
- Make it look convincing but include the fraud mechanics subtly

Populate 'archetype' with '{archetype_name}' and list the signals used in 'fraud_signals_used'."""


def build_real_prompt() -> str:
    sectors = [
        ("Software Engineer",       "Bengaluru",  "₹8-14 LPA",  "Infosys / Wipro / mid-size IT firm"),
        ("Marketing Executive",     "Mumbai",     "₹4-7 LPA",   "FMCG or D2C brand"),
        ("Data Analyst",            "Hyderabad",  "₹6-10 LPA",  "analytics or fintech firm"),
        ("Operations Manager",      "Pune",       "₹7-12 LPA",  "manufacturing or logistics company"),
        ("HR Business Partner",     "Gurugram",   "₹8-13 LPA",  "IT services company"),
        ("Content Writer",          "Remote",     "₹3-5 LPA",   "media or edtech startup"),
        ("Sales Executive",         "Chennai",    "₹3.5-6 LPA", "insurance or telecom company"),
        ("Financial Analyst",       "Mumbai",     "₹7-11 LPA",  "NBFC or bank"),
        ("UX Designer",             "Bengaluru",  "₹7-12 LPA",  "product startup"),
        ("Supply Chain Executive",  "Ahmedabad",  "₹4-7 LPA",   "manufacturing firm"),
    ]
    role, city, salary, company_type = random.choice(sectors)

    return f"""Generate a legitimate Indian job posting.

Role: {role}
City: {city}
Salary range: {salary}
Company type: {company_type}

Requirements:
- Professional tone, specific responsibilities (200-500 words)
- Realistic qualifications (years of experience, specific skills)
- Official-looking contact (company domain email, not Gmail)
- Include standard sections: About the role, Responsibilities, Requirements, What we offer

Populate 'archetype' with 'legitimate' and leave 'fraud_signals_used' as an empty array."""


def generate_one(system_prompt: str, user_prompt: str, retries: int = 5) -> dict | None:
    schema_fields = JobPosting.model_json_schema()["properties"]
    expected_keys = list(schema_fields.keys())

    # Enforce flat JSON output - the model sometimes wraps under a parent key
    groq_system_prompt = f"""{system_prompt}

CRITICAL: You must output a flat JSON object using EXACTLY these keys. Do not nest them under a parent key, and do not change their names:
{json.dumps(expected_keys)}

Example format:
{{
  "title": "...",
  "company": "...",
  "location": "...",
  "salary": "...",
  "description": "...",
  "requirements": "...",
  "contact": "...",
  "archetype": "...",
  "fraud_signals_used": []
}}"""

    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": groq_system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )

            raw_text  = chat_completion.choices[0].message.content
            data_dict = json.loads(raw_text)

            # Unbox nested objects (model sometimes wraps under "job_posting", "job", etc.)
            for wrapper in ["job_posting", "job", "posting"]:
                if wrapper in data_dict and isinstance(data_dict[wrapper], dict):
                    data_dict = data_dict[wrapper]
                    break
            if len(data_dict) == 1 and "title" not in data_dict and "job_title" not in data_dict:
                sole_key = list(data_dict.keys())[0]
                if isinstance(data_dict[sole_key], dict):
                    data_dict = data_dict[sole_key]

            # Heal key hallucinations
            if "job_title" in data_dict and "title" not in data_dict:
                data_dict["title"] = data_dict.pop("job_title")
            if "job_description" in data_dict and "description" not in data_dict:
                data_dict["description"] = data_dict.pop("job_description")
            if "contact_details" in data_dict and "contact" not in data_dict:
                data_dict["contact"] = data_dict.pop("contact_details")

            # Flatten any structured fields that should be strings
            for field in ["title", "company", "location", "salary", "description", "requirements", "contact", "archetype"]:
                if field in data_dict and not isinstance(data_dict[field], str):
                    if isinstance(data_dict[field], dict):
                        data_dict[field] = ", ".join(f"{k}: {v}" for k, v in data_dict[field].items())
                    elif isinstance(data_dict[field], list):
                        data_dict[field] = " ".join(str(item) for item in data_dict[field])
                    else:
                        data_dict[field] = str(data_dict[field])

            # Fill missing optional fields
            if not data_dict.get("contact"):
                data_dict["contact"] = "Contact via company career portal"
            if "fraud_signals_used" not in data_dict:
                data_dict["fraud_signals_used"] = []

            validated = JobPosting.model_validate(data_dict)
            return validated.model_dump()

        except Exception as e:
            if "429" in str(e):
                print(f"  [Rate limit] sleeping 3s...")
                time.sleep(3)
            else:
                print(f"  Error (attempt {attempt+1}): {e}")
                time.sleep(1.5)
    return None


def generate_dataset(n_fraud: int, n_real: int, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    json_path = os.path.join(output_dir, "synthetic_indian_jobs.json")
    csv_path  = os.path.join(output_dir, "synthetic_indian_jobs.csv")

    # Resume from checkpoint if available
    records = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            print(f"Resuming from checkpoint: {len(records)} records found.")
        except Exception as e:
            print(f"Could not read checkpoint, starting fresh. Error: {e}")

    existing_fraud_counts = {}
    existing_real_count   = 0
    for r in records:
        if r.get("label") == 1:
            arch = r.get("archetype", "unknown")
            existing_fraud_counts[arch] = existing_fraud_counts.get(arch, 0) + 1
        elif r.get("label") == 0:
            existing_real_count += 1

    archetype_names = list(SCAM_ARCHETYPES.keys())

    # Generate fraud posts
    print(f"\nTargeting {n_fraud} total fraudulent postings...")
    per_archetype = n_fraud // len(archetype_names)
    remainder     = n_fraud % len(archetype_names)

    for i, arch_name in enumerate(archetype_names):
        target_count    = per_archetype + (1 if i < remainder else 0)
        already_done    = existing_fraud_counts.get(arch_name, 0)
        remaining       = target_count - already_done

        if remaining <= 0:
            print(f"  {arch_name} -> already complete ({already_done}/{target_count})")
            continue

        print(f"  {arch_name} -> {already_done}/{target_count}, generating {remaining} more...")

        archetype = SCAM_ARCHETYPES[arch_name]
        for j in range(remaining):
            result = generate_one(FRAUD_SYSTEM_PROMPT, build_fraud_prompt(arch_name, archetype))
            if result:
                result["label"]  = 1
                result["source"] = "synthetic_indian"
                records.append(result)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
            if (j + 1) % 5 == 0 or (j + 1) == remaining:
                print(f"    {already_done + j + 1}/{target_count} done")
            time.sleep(0.2)

    # Generate real posts
    remaining_real = n_real - existing_real_count
    if remaining_real > 0:
        print(f"\nTargeting {n_real} legitimate postings. {existing_real_count}/{n_real} done, generating {remaining_real} more...")
        for j in range(remaining_real):
            result = generate_one(REAL_SYSTEM_PROMPT, build_real_prompt())
            if result:
                result["label"]  = 0
                result["source"] = "synthetic_legitimate"
                records.append(result)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
            if (existing_real_count + j + 1) % 20 == 0 or (j + 1) == remaining_real:
                print(f"  {existing_real_count + j + 1}/{n_real} done")
            time.sleep(0.2)
    else:
        print(f"\nLegitimate postings already complete ({existing_real_count}/{n_real}).")

    # Write final CSV
    print("\nFinalizing and writing CSV...")
    fieldnames = ["title", "company", "location", "salary", "description",
                  "requirements", "contact", "archetype", "label", "source"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    fraud_count = sum(1 for r in records if r["label"] == 1)
    real_count  = sum(1 for r in records if r["label"] == 0)
    print(f"\nDataset complete:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  Fraud: {fraud_count} | Real: {real_count} | Total: {len(records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Indian job fraud dataset using Groq")
    parser.add_argument("--n_fraud", type=int,  default=300,        help="Number of fraud posts to generate")
    parser.add_argument("--n_real",  type=int,  default=600,        help="Number of real posts to generate")
    parser.add_argument("--output",  type=str,  default="data/raw/", help="Output directory")
    args = parser.parse_args()

    generate_dataset(args.n_fraud, args.n_real, args.output)