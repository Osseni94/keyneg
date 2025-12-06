"""
KeyNeg Taxonomy Module
======================
Comprehensive taxonomy for extracting negative sentiment, complaints, and discontent
from text data for workforce intelligence and marketing analysis.

Author: Kaossara Osseni
Email: admin@grandnasser.com
Version: 2.1.0
"""

# =============================================================================
# SENTIMENT LABELS - Core labels for classification
# =============================================================================

SENTIMENT_LABELS = [
    # Work Environment & Culture
    "toxic culture",
    "hostile work environment",
    "bullying",
    "harassment",
    "discrimination",
    "retaliation",
    "favoritism",
    "nepotism",
    "exclusion",
    "office politics",

    # Management Issues
    "micromanagement",
    "poor leadership",
    "lack of direction",
    "incompetent management",
    "absent manager",
    "authoritarian management",
    "lack of accountability",
    "dismissive management",

    # Recognition & Value
    "lack of recognition",
    "undervalued",
    "unappreciated",
    "taken for granted",
    "credit stolen",
    "efforts ignored",

    # Trust & Transparency
    "lack of trust",
    "lack of transparency",
    "broken promises",
    "misleading communication",
    "secrecy",

    # Workload & Burnout
    "burnout",
    "exhaustion",
    "overwhelmed",
    "overworked",
    "unrealistic deadlines",
    "workload stress",
    "unsustainable pace",
    "crunch time",

    # Staffing
    "understaffed",
    "high turnover",
    "layoffs",
    "downsizing",
    "restructuring anxiety",

    # Compensation
    "underpaid",
    "pay disparity",
    "unfair compensation",
    "denied raise",
    "poor benefits",
    "unpaid overtime",

    # Career Development
    "career stagnation",
    "no growth opportunities",
    "dead end job",
    "promotion blocked",
    "glass ceiling",
    "lack of training",
    "skills obsolescence",

    # Work-Life Balance
    "work life imbalance",
    "excessive hours",
    "no flexibility",
    "always on call",
    "personal strain",

    # Communication
    "poor communication",
    "information silos",
    "miscommunication",
    "left in the dark",

    # Team Dynamics
    "team conflict",
    "poor teamwork",
    "isolation",
    "interpersonal tension",
    "lack of collaboration",

    # Morale & Engagement
    "low morale",
    "disengagement",
    "demotivation",
    "frustration",
    "dissatisfaction",
    "hopelessness",
    "resentment",

    # Job Security
    "job insecurity",
    "fear of layoff",
    "organizational instability",
    "uncertainty",

    # Process & Systems
    "bureaucracy",
    "inefficient processes",
    "outdated systems",
    "poor tools",
    "technical debt",

    # Safety & Ethics
    "safety concerns",
    "ethical violations",
    "compliance issues",
    "misconduct",

    # Customer/Product Issues (for marketing)
    "poor quality",
    "defective product",
    "poor customer service",
    "overpriced",
    "false advertising",
    "poor user experience",
]

# =============================================================================
# FULL TAXONOMY - Hierarchical keyword dictionary
# =============================================================================

NEGATIVE_TAXONOMY = {
    "work_environment_culture": {
        "toxic_culture": [
            "toxic", "hostile", "bullying", "harassment", "discrimination",
            "retaliation", "favoritism", "nepotism", "cliques", "exclusion",
            "bias", "gossip", "backstabbing", "unprofessional", "passive aggressive",
            "aggressive behavior", "unsafe environment", "unethical behavior",
            "gaslighting", "intimidation", "sabotage", "mobbing", "hazing",
            "scapegoating", "power trip", "abuse of power", "toxic positivity"
        ],
        "poor_management": [
            "micromanagement", "micromanager", "ineffective leadership", "lack of direction",
            "unclear expectations", "inconsistent", "unfair", "disorganized",
            "lack of accountability", "weak leadership", "unresponsive",
            "poor communication", "dismissive", "absent manager", "incompetent boss",
            "clueless manager", "hands-off management", "absentee leadership",
            "authoritarian", "dictatorial", "control freak", "power hungry",
            "out of touch", "disconnected", "ivory tower", "no backbone"
        ],
        "lack_of_recognition": [
            "undervalued", "unappreciated", "ignored", "overlooked",
            "no recognition", "inadequate feedback", "taken for granted",
            "invisible", "credit stolen", "thankless", "unrewarded",
            "accomplishments dismissed", "efforts unnoticed", "forgotten",
            "passed over", "sidelined", "marginalized"
        ],
        "low_trust": [
            "distrust", "secrecy", "lack of transparency", "broken promises",
            "misleading", "uncertainty", "hidden agenda", "deceptive",
            "dishonest", "lies", "cover up", "withholding information",
            "not trustworthy", "shady", "sketchy", "suspicious"
        ]
    },

    "workload_pressure": {
        "burnout": [
            "burnout", "burned out", "burnt out", "exhausted", "overwhelmed",
            "overworked", "long hours", "unreasonable workload", "unrealistic expectations",
            "stress", "stressed", "fatigue", "fatigued", "unsustainable pace",
            "running on empty", "depleted", "drained", "wiped out", "crushed",
            "breaking point", "at my limit", "can't take it anymore", "too much",
            "drowning", "swamped", "slammed", "buried", "crushed by work"
        ],
        "staffing_issues": [
            "understaffed", "short staffed", "turnover", "layoffs", "lack of support",
            "too much responsibility", "skeleton crew", "bare bones", "stretched thin",
            "no backup", "single point of failure", "one person team", "do it all",
            "wear many hats", "spread too thin", "headcount freeze", "hiring freeze"
        ],
        "deadline_pressure": [
            "unrealistic deadlines", "impossible timeline", "rushed", "crunch time",
            "death march", "perpetual crunch", "always urgent", "everything is priority",
            "moving goalposts", "scope creep", "last minute", "fire drill",
            "constant emergencies", "reactive", "never proactive"
        ],
        "role_misalignment": [
            "unclear role", "scope creep", "misaligned expectations",
            "job mismatch", "wrong responsibilities", "bait and switch",
            "not what I signed up for", "role creep", "mission creep",
            "unclear boundaries", "jack of all trades"
        ]
    },

    "compensation_benefits": {
        "pay_discontent": [
            "underpaid", "low pay", "unfair pay", "unequal pay",
            "pay gap", "not competitive", "denied raise", "poor salary structure",
            "below market", "poverty wages", "insulting offer", "lowballed",
            "pay freeze", "wage stagnation", "no cost of living adjustment",
            "compression", "salary compression", "new hires make more"
        ],
        "benefits_issues": [
            "poor benefits", "expensive benefits", "outdated benefits",
            "inadequate health insurance", "no flexibility", "no PTO",
            "use it or lose it", "no sick leave", "bad 401k", "no match",
            "high deductible", "poor coverage", "benefits cut",
            "no parental leave", "no remote work", "RTO", "return to office"
        ],
        "rewards_inequity": [
            "unfair bonus", "inconsistent rewards", "inequitable compensation",
            "promotion favoritism", "merit not recognized", "politics over performance",
            "boys club", "old boys network", "inner circle", "favorites get ahead"
        ]
    },

    "career_development": {
        "promotion_issues": [
            "no growth", "stagnant", "dead end", "dead end job", "limited advancement",
            "blocked promotion", "unfair promotions", "no opportunities",
            "glass ceiling", "stuck", "plateau", "plateaued", "going nowhere",
            "career suicide", "resume killer", "no upward mobility",
            "promoted over me", "passed over for promotion"
        ],
        "training_issues": [
            "lack of training", "no learning opportunities", "outdated skills",
            "no upskilling", "no mentorship", "sink or swim", "thrown in deep end",
            "no onboarding", "figure it out yourself", "no documentation",
            "tribal knowledge", "no knowledge transfer"
        ],
        "career_misalignment": [
            "unclear career path", "unstructured growth", "skills mismatch",
            "career uncertainty", "pigeonholed", "typecast", "no lateral moves",
            "stuck in role", "no rotation", "same thing every day"
        ]
    },

    "process_systems_operations": {
        "inefficiency": [
            "slow processes", "bureaucracy", "red tape", "outdated tools",
            "inefficient systems", "manual work", "poor automation", "bottlenecks",
            "legacy systems", "technical debt", "spaghetti code", "band-aid fixes",
            "duct tape solutions", "workarounds", "hacks everywhere",
            "approval hell", "sign-off nightmare", "too many approvals"
        ],
        "organizational_chaos": [
            "constant changes", "instability", "confusion", "lack of planning",
            "poor coordination", "chaos", "clusterfuck", "dumpster fire",
            "firefighting mode", "reactive not proactive", "no strategy",
            "flavor of the month", "pivot fatigue", "direction changes constantly",
            "whiplash", "reorganization", "reorg", "restructuring"
        ],
        "policy_friction": [
            "unclear policies", "inconsistent rules", "excessive procedures",
            "overly strict", "arbitrary rules", "one size fits all",
            "no exceptions", "zero tolerance", "policy for everything",
            "bureaucratic nightmare", "paperwork", "documentation overload"
        ]
    },

    "team_dynamics": {
        "collaboration_issues": [
            "poor teamwork", "isolation", "silos", "silo mentality",
            "lack of support", "conflict", "disagreement", "turf wars",
            "empire building", "hoarding information", "not my job",
            "finger pointing", "blame game", "throw under the bus",
            "thrown under bus", "scapegoat", "fall guy"
        ],
        "communication_problems": [
            "miscommunication", "unclear messages", "lack of feedback",
            "inconsistent communication", "radio silence", "ghosted",
            "left out of loop", "out of the loop", "OOTL", "not informed",
            "last to know", "surprises", "blindsided", "ambushed",
            "passive aggressive emails", "CYA emails", "cover your ass"
        ],
        "interpersonal_conflict": [
            "personality clashes", "tension", "disrespect", "lack of cooperation",
            "team distrust", "hostile", "antagonistic", "combative",
            "aggressive", "condescending", "patronizing", "rude",
            "snappy", "short tempered", "temper tantrums", "yelling",
            "screaming", "shouting", "belittling", "demeaning"
        ]
    },

    "job_satisfaction": {
        "low_morale": [
            "demotivated", "disengaged", "unhappy", "unfulfilling", "boring",
            "meaningless work", "pointless", "why bother", "checking out",
            "quiet quitting", "bare minimum", "just collecting paycheck",
            "going through motions", "clock watching", "counting down",
            "Sunday scaries", "Monday dread", "hate my job"
        ],
        "emotional_frustration": [
            "frustration", "frustrated", "anger", "angry", "disappointment",
            "disappointed", "annoyance", "annoyed", "irritation", "irritated",
            "resentment", "resentful", "bitterness", "bitter", "fed up",
            "had enough", "last straw", "breaking point", "snapping"
        ],
        "lack_of_autonomy": [
            "no control", "micromanaged", "restricted decision making",
            "powerless", "lack of ownership", "no say", "no voice",
            "not consulted", "decisions made for me", "no input",
            "rubber stamping", "just execute", "order taker"
        ]
    },

    "work_life_balance": {
        "personal_strain": [
            "no balance", "too many hours", "burnout", "family strain",
            "stress at home", "missing life", "no personal time",
            "always working", "can't disconnect", "work follows me home",
            "no boundaries", "boundary violation", "weekend work",
            "vacation shaming", "guilt for taking PTO"
        ],
        "scheduling_issues": [
            "inflexible schedule", "unpredictable hours", "bad shifts",
            "constant overtime", "on call always", "after hours calls",
            "early morning meetings", "late night meetings", "timezone issues",
            "no work from home", "mandatory office", "butts in seats"
        ]
    },

    "organizational_stability": {
        "strategic_concerns": [
            "unclear strategy", "poor planning", "instability", "volatility",
            "uncertainty", "no vision", "direction unclear", "lost",
            "rudderless", "adrift", "floundering", "sinking ship",
            "going under", "doomed", "circling drain"
        ],
        "restructuring_anxiety": [
            "layoffs", "downsizing", "cost cutting", "fear of job loss",
            "reorg fatigue", "restructuring", "RIF", "reduction in force",
            "workforce reduction", "headcount reduction", "pink slip",
            "chopping block", "on the block", "next to go", "survivor guilt",
            "layoff survivor", "job cuts", "staff cuts"
        ],
        "leadership_turmoil": [
            "leadership turnover", "ceo exit", "conflicting priorities",
            "lack of vision", "executive shuffle", "musical chairs",
            "leadership vacuum", "power struggle", "infighting",
            "board drama", "investor pressure", "activist investor"
        ]
    },

    "customer_market_discontent": {
        "product_issues": [
            "broken", "buggy", "defective", "slow", "glitchy", "unreliable",
            "hard to use", "poor quality", "cheap", "flimsy", "doesn't work",
            "stopped working", "failed", "malfunction", "error", "crash",
            "not as advertised", "misleading", "false advertising"
        ],
        "service_complaints": [
            "rude staff", "unhelpful support", "no response", "long wait times",
            "unresolved issue", "ignored", "dismissed", "runaround",
            "passed around", "transferred repeatedly", "hold forever",
            "no callback", "ticket closed", "issue not resolved"
        ],
        "pricing_complaints": [
            "expensive", "hidden fees", "overpriced", "poor value",
            "not worth it", "rip off", "ripoff", "price gouging",
            "bait and switch", "nickel and dime", "surprise charges",
            "billing issues", "overcharged", "double charged"
        ],
        "trust_issues": [
            "misleading", "unreliable brand", "security concerns", "data breach",
            "suspicious", "scam", "fraud", "fraudulent", "dishonest",
            "untrustworthy", "shady", "predatory", "exploitative"
        ]
    },

    "risk_safety_legal": {
        "safety_concerns": [
            "unsafe conditions", "hazards", "dangerous", "injury risk",
            "lack of safety", "safety violation", "OSHA violation",
            "no PPE", "unsafe equipment", "accident waiting to happen",
            "near miss", "injury", "hurt on job", "workers comp"
        ],
        "ethical_violations": [
            "misconduct", "fraud", "unethical", "non compliance", "violations",
            "illegal", "breaking law", "cutting corners", "cover up",
            "whistleblower retaliation", "looking the other way",
            "cooking the books", "falsifying", "lying to customers"
        ]
    },

    "emotional_states": {
        "core_negative_emotions": [
            "angry", "upset", "frustrated", "disappointed", "annoyed",
            "irritated", "offended", "resentful", "bitter", "anxious",
            "worried", "fearful", "stressed", "overwhelmed", "depressed",
            "hopeless", "distrustful", "skeptical", "cynical", "jaded",
            "demoralized", "defeated", "helpless", "trapped", "stuck"
        ],
        "intensity_expressions": {
            "mild": [
                "slightly disappointed", "not completely satisfied", "could be better",
                "room for improvement", "somewhat concerned", "a bit frustrated",
                "mildly annoying", "not ideal", "less than perfect", "suboptimal"
            ],
            "moderate": [
                "disappointed", "frustrated", "concerned", "unsatisfied",
                "unhappy", "annoyed", "bothered", "troubled", "dissatisfied",
                "let down", "upset", "bothered"
            ],
            "strong": [
                "angry", "furious", "horrible", "terrible", "awful",
                "dreadful", "atrocious", "appalling", "infuriating", "outrageous",
                "unacceptable", "ridiculous", "absurd", "insane"
            ],
            "extreme": [
                "outraged", "devastated", "betrayed", "disgusted", "unacceptable",
                "intolerable", "despicable", "contemptible", "abhorrent", "revolting",
                "livid", "seething", "fuming", "enraged", "incensed"
            ]
        }
    },

    "offensive_language": {
        "profanity": [
            "asshole", "bastard", "bitch", "bullshit", "crap", "damn", "dick",
            "fuck", "fucking", "fucked", "shit", "shitty", "piss", "pissed",
            "scumbag", "douche", "douchebag", "wanker", "jerk", "idiot",
            "moron", "imbecile", "stupid", "dumb", "incompetent"
        ],
        "censored_variations": [
            "f**k", "s**t", "a**hole", "b****", "d**k", "c**p",
            "f*ck", "sh*t", "a*s", "b*tch", "cr*p", "d*mn"
        ],
        "acronyms": [
            "WTF", "FFS", "BS", "GTFO", "STFU", "SMH", "FML",
            "POS", "SOB", "OMFG", "JFC"
        ],
        "euphemisms": [
            "fudge", "frick", "fricking", "freaking", "darn", "dang",
            "heck", "shoot", "crap", "jeez", "geez", "gosh"
        ],
        "internet_slang": [
            "karen", "boomer", "ok boomer", "cringe", "cringey", "yikes",
            "big yikes", "oof", "rekt", "fail", "epic fail", "trash",
            "garbage", "dumpster fire", "hot mess", "toxic"
        ]
    },

    "action_indicators": {
        "departure_intent": [
            "quitting", "leaving", "resigning", "job hunting", "looking for jobs",
            "updating resume", "interviewing", "two weeks notice", "giving notice",
            "walking out", "done here", "out of here", "moving on",
            "time to leave", "exit strategy", "one foot out door"
        ],
        "escalation_threats": [
            "going to sue", "legal action", "lawyer", "attorney", "complaint",
            "reporting to HR", "going to HR", "EEOC", "labor board",
            "whistleblower", "going public", "media", "social media blast",
            "glassdoor review", "BBB complaint", "consumer protection"
        ],
        "coping_mechanisms": [
            "quiet quitting", "bare minimum", "acting my wage", "coasting",
            "checked out", "tuned out", "disengaged", "autopilot",
            "phoning it in", "mailing it in", "going through motions"
        ]
    },

    "industry_specific": {
        "tech": [
            "buggy software", "technical debt", "legacy code", "spaghetti code",
            "constant fires", "on-call nightmare", "pager duty", "incidents",
            "outages", "downtime", "deployment failures", "rollback",
            "feature creep", "scope creep", "agile theater", "scrum but"
        ],
        "retail_hospitality": [
            "rude customers", "customer abuse", "no tips", "low tips",
            "standing all day", "no breaks", "holiday work", "weekend work",
            "understaffed", "skeleton crew", "impossible sales targets",
            "mystery shopper", "customer is always right"
        ],
        "healthcare": [
            "understaffed", "patient overload", "compassion fatigue",
            "mandatory overtime", "short staffed", "no beds", "waitlist",
            "insurance issues", "prior auth", "denied claims", "charting burden",
            "EHR nightmare", "documentation overload"
        ],
        "finance": [
            "audit nightmare", "compliance burden", "regulatory pressure",
            "quarter end crunch", "year end crunch", "billing targets",
            "billable hours", "utilization pressure", "client demands"
        ]
    },

    "multi_word_expressions": {
        "complaint_phrases": [
            "not happy with", "disappointed in", "frustrated by", "angry about",
            "waste of money", "waste of time", "poor experience", "bad experience",
            "would not recommend", "never again", "worst ever", "complete disaster",
            "stay away from", "avoid at all costs", "biggest mistake", "regret",
            "total nightmare", "complete joke", "absolute mess"
        ],
        "workplace_phrases": [
            "hostile work environment", "toxic workplace", "dead end job",
            "no work life balance", "always on call", "constant overtime",
            "no room for growth", "no advancement opportunities",
            "unfair treatment", "unequal pay", "pay discrimination",
            "glass ceiling", "old boys club", "boys club mentality"
        ],
        "warning_phrases": [
            "don't work here", "run away", "stay away", "avoid this company",
            "worst place to work", "terrible employer", "horrible company",
            "dysfunctional organization", "sinking ship", "rats leaving ship"
        ]
    },

    "intensity_modifiers": {
        "amplifiers": [
            "extremely", "incredibly", "absolutely", "completely", "totally",
            "utterly", "exceptionally", "particularly", "especially", "highly",
            "seriously", "literally", "actually", "genuinely", "truly"
        ],
        "frequency": [
            "always", "constantly", "continuously", "perpetually", "incessantly",
            "endlessly", "forever", "repeatedly", "frequently", "often",
            "regularly", "consistently", "habitually", "chronically", "persistently"
        ],
        "negation": [
            "not", "no", "never", "none", "nothing", "nowhere", "nobody",
            "no one", "hardly", "scarcely", "barely", "rarely", "seldom"
        ],
        "comparative": [
            "worse than", "not as good as", "inferior to", "poorer than",
            "less than", "worse off", "even worse", "getting worse"
        ],
        "superlative": [
            "worst", "least", "poorest", "most disappointing", "most frustrating",
            "absolute worst", "by far the worst", "hands down worst"
        ]
    },

    "contextual_signals": {
        "sarcasm_indicators": [
            "great just great", "just what I needed", "perfect timing",
            "fantastic more problems", "wonderful another issue", "oh joy",
            "how nice", "love that for me", "so professional", "very helpful"
        ],
        "resignation_indicators": [
            "whatever", "doesn't matter", "who cares", "why bother",
            "what's the point", "nothing will change", "waste of breath",
            "talking to a wall", "beating dead horse", "lost cause"
        ]
    }
}


def get_all_keywords() -> list:
    """Extract all keywords from the taxonomy as a flat list."""
    keywords = []

    def extract_from_dict(d):
        for key, value in d.items():
            if isinstance(value, list):
                keywords.extend(value)
            elif isinstance(value, dict):
                extract_from_dict(value)

    extract_from_dict(NEGATIVE_TAXONOMY)
    return list(set(keywords))


def get_keywords_by_category(category: str) -> list:
    """Get keywords for a specific category."""
    if category in NEGATIVE_TAXONOMY:
        keywords = []
        def extract(d):
            for v in d.values():
                if isinstance(v, list):
                    keywords.extend(v)
                elif isinstance(v, dict):
                    extract(v)
        extract(NEGATIVE_TAXONOMY[category])
        return keywords
    return []


def get_category_labels() -> list:
    """Get all top-level category names."""
    return list(NEGATIVE_TAXONOMY.keys())
