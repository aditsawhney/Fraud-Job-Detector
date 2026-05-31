"""
indian_fraud_taxonomy.py

Six Indian job scam archetypes, grounded in:
- NCRB cybercrime reports (2021-2023)
- Cybercrime.gov.in complaint patterns
- Documented cases: CIEL HR, ScamDekho, Hirist, FactSuite
- Press reports (Navi Mumbai placement agency, Delhi cyber cell busts)
"""

SCAM_ARCHETYPES = {

    "placement_consultancy_fee": {
        "description": "Fake agency charges upfront registration/processing fee, promises placement, disappears",
        "real_cases": "Navi Mumbai agency cheated 6 people of ₹16.8 lakh (May 2025); Delhi NCR placement scams",
        "target": "Freshers, unemployed graduates, tier-2/3 city candidates",
        "mechanics": "Collects ₹2,000-25,000 as 'registration', 'processing', or 'training' fee; may conduct fake interview to build trust",
        "linguistic_signals": [
            "registration fee", "processing charges", "security deposit",
            "refundable fee", "training fee", "placement guarantee",
            "100% placement", "guaranteed job", "consultancy charges",
            "documentation charges", "joining fee", "enrollment fee",
        ],
        "structural_signals": [
            "gmail/yahoo contact email",
            "no company website or fake-looking website",
            "WhatsApp number as primary contact",
            "vague company address (often just city name)",
            "no LinkedIn company page",
        ],
        "salary_range": "₹15,000-45,000/month (believable but attractive)",
        "roles": ["HR Executive", "Data Entry Operator", "Back Office Executive",
                  "Customer Support", "Tele-caller", "Accounts Assistant"],
    },

    "fake_mnc_impersonation": {
        "description": "Scammer impersonates TCS, Infosys, Wipro, Amazon, Flipkart, etc.",
        "real_cases": "TCS impersonation scams documented by Hirist; Amazon/Flipkart fake recruiter profiles on LinkedIn",
        "target": "Freshers and 1-3 yr experience candidates desperate for MNC entry",
        "mechanics": "Uses near-identical company name, fake offer letters, asks for 'security deposit' or 'background check fee'",
        "linguistic_signals": [
            "immediate joining", "offer letter will be provided",
            "on-the-spot selection", "walk-in interview",
            "100 openings", "bulk hiring", "campus drive",
            "background verification fee", "medical test fee",
            "onboarding charges", "bond amount",
        ],
        "structural_signals": [
            "email domain slightly off (tcs-hr.com, infosys-careers.net)",
            "no official company email domain",
            "interview via WhatsApp call or Google Meet only",
            "salary much higher than market for experience level",
        ],
        "salary_range": "₹4-8 LPA for freshers (inflated)",
        "roles": ["Software Engineer", "Associate Consultant", "Business Analyst",
                  "Operations Executive", "Process Associate"],
    },

    "wfh_task_scam": {
        "description": "Work-from-home jobs requiring 'liking', 'reviewing', 'rating', or 'data entry' tasks",
        "real_cases": "Student promised ₹3,000/day for liking videos; WhatsApp ₹3,000-5,000/day messages (India Today verified hoax)",
        "target": "Students, homemakers, unemployed youth",
        "mechanics": "Small initial payout builds trust, then asks for 'investment' or 'task unlock fee' to access higher-paying tasks",
        "linguistic_signals": [
            "work from home", "part time", "2-3 hours daily",
            "earn from mobile", "no experience needed", "freshers welcome",
            "₹500-5000 per day", "daily payment", "instant payout",
            "simple tasks", "like and earn", "rate and earn",
            "product reviewer", "app tester", "survey jobs",
            "data typing", "copy paste work", "form filling",
        ],
        "structural_signals": [
            "Telegram group mentioned",
            "WhatsApp first contact",
            "no office address",
            "payment via UPI only",
            "screenshot of earnings shared as proof",
        ],
        "salary_range": "₹15,000-1,50,000/month (absurdly high for task described)",
        "roles": ["Product Reviewer", "Data Entry Executive", "Social Media Evaluator",
                  "App Tester", "Online Survey Specialist", "Content Rater"],
    },

    "overseas_job_scam": {
        "description": "Fake overseas jobs (Dubai, Singapore, Canada, UK) requiring visa/processing fees",
        "real_cases": "Delhi Police cyber cell busted Dubai visa racket; Punjab/Haryana agents booking under illegal migration",
        "target": "Blue-collar workers, aspirants from Punjab/Haryana/Kerala/UP",
        "mechanics": "Charges visa fee, processing fee, air ticket deposit; may provide fake offer letters",
        "linguistic_signals": [
            "Dubai", "Singapore", "Canada", "UK", "abroad",
            "visa sponsorship", "free accommodation", "food provided",
            "air ticket provided", "work permit", "PR opportunity",
            "Gulf job", "foreign placement", "overseas opportunity",
            "visa charges", "processing fee", "documentation fee",
        ],
        "structural_signals": [
            "no verifiable foreign company name",
            "contact via WhatsApp only",
            "fee payment via personal bank account or UPI",
        ],
        "salary_range": "AED 3,000-8,000 or SGD 2,000-5,000 (vague currency)",
        "roles": ["Driver", "Electrician", "Plumber", "Nurse", "Accountant",
                  "Warehouse Worker", "Security Guard", "Chef"],
    },

    "identity_harvest": {
        "description": "Job post designed to collect Aadhaar, PAN, bank details early in the process",
        "real_cases": "Quora Delhi NCR: recruiter asks for scanned Aadhaar+PAN+bank details; used for unauthorized KYC",
        "target": "Any job seeker, especially those unfamiliar with what's appropriate to share",
        "mechanics": "Asks for documents 'to check eligibility' or 'to process your application' before any interview",
        "linguistic_signals": [
            "send your Aadhaar", "PAN card required", "bank account details",
            "passport copy", "photograph required", "DOB required",
            "submit documents to proceed", "KYC verification",
            "background check fee", "CIBIL score check",
            "nominee details", "family details required",
        ],
        "structural_signals": [
            "asks for sensitive documents at application stage",
            "form collecting financial info",
            "no interview before document collection",
        ],
        "salary_range": "₹20,000-60,000 (varies, used as bait)",
        "roles": ["Any role - identity harvest is cross-cutting"],
    },

    "bpo_bulk_scam": {
        "description": "Fake BPO/call centre jobs targeting freshers with vague roles and immediate joining",
        "real_cases": "Bulk BPO scams documented across Bangalore, Hyderabad, NCR",
        "target": "12th pass, graduates, anyone needing immediate income",
        "mechanics": "Vague job description, 'immediate joining', walk-in address that doesn't exist or is a residential flat",
        "linguistic_signals": [
            "BPO", "call centre", "voice process", "non-voice process",
            "immediate joiners", "walk-in interview", "same day offer",
            "Sunday open", "no target pressure", "fixed shift",
            "5 days working", "both shifts available", "rotational shift",
            "salary + incentives", "upto ₹25,000", "CTC not mentioned",
            "apply now limited seats", "urgently required",
        ],
        "structural_signals": [
            "no specific company name, just 'leading BPO'",
            "address is a residential area or mall",
            "contact is personal mobile number",
            "no company email - only Gmail",
        ],
        "salary_range": "₹10,000-25,000 (realistic but vague, 'upto' language)",
        "roles": ["Customer Care Executive", "Tele-caller", "BPO Executive",
                  "Voice Process Associate", "Chat Support", "Data Entry"],
    },
}

# Signals that appear across multiple archetypes
CROSS_CUTTING_SIGNALS = {
    "fee_language": [
        "registration fee", "processing fee", "security deposit",
        "refundable deposit", "training charges", "documentation fee",
        "joining fee", "enrollment fee", "application fee",
        "bond amount", "caution money",
    ],
    "urgency_language": [
        "urgent requirement", "immediate joining", "apply now",
        "limited seats", "today only", "last date",
        "don't miss this opportunity", "hurry", "act fast",
        "walk-in tomorrow", "same day joining",
    ],
    "unrealistic_compensation": [
        "earn ₹500-5000 per day", "₹15,000-1,50,000 per month",
        "unlimited earning", "no income cap", "incentives unlimited",
        "salary negotiable", "best in industry", "highest package",
    ],
    "contact_red_flags": [
        "contact on WhatsApp", "WhatsApp only", "Telegram",
        "call/WhatsApp", "DM for details", "message for more info",
        "gmail.com", "yahoo.com", "hotmail.com", "rediffmail.com",
    ],
    "vague_company": [
        "leading company", "MNC", "reputed organization",
        "well-known firm", "our client", "confidential company",
        "top company", "Fortune 500 company",  # without naming it
    ],
    "document_red_flags": [
        "send Aadhaar", "PAN required", "bank details",
        "original documents", "passport size photo",
        "submit documents first", "verification charges",
    ],
    "credential_mismatch": [
        "no experience required", "freshers only", "12th pass can apply",
        "any graduate", "no skills required", "training will be provided",
        # combined with high salary = red flag
    ],
}

if __name__ == "__main__":
    print(f"Loaded {len(SCAM_ARCHETYPES)} scam archetypes")
    total_signals = sum(len(v["linguistic_signals"]) for v in SCAM_ARCHETYPES.values())
    print(f"Total archetype-level linguistic signals: {total_signals}")
    cross_signals = sum(len(v) for v in CROSS_CUTTING_SIGNALS.values())
    print(f"Cross-cutting signals: {cross_signals}")