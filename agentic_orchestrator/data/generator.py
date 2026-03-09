"""
Synthetic transaction data generator for cross-border payments.
Generates realistic remittance transactions with fraud patterns.
"""

import random
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from agentic_orchestrator.config import CORRIDORS, RANDOM_SEED, SANCTIONS_COUNTRIES

random.seed(RANDOM_SEED)


@dataclass
class UserProfile:
    user_id: str
    name: str
    country: str
    typical_amount: float
    typical_recipients: List[str]
    typical_corridor: str
    registered_device_id: str
    registered_ip_prefix: str
    life_events: List[Dict[str, str]] = field(default_factory=list)
    transaction_history: List[Dict] = field(default_factory=list)


@dataclass
class Transaction:
    txn_id: str
    user_id: str
    amount_usd: float
    recipient_name: str
    recipient_country: str
    corridor: str
    device_id: str
    ip_address: str
    timestamp: str
    is_new_recipient: bool
    is_fraud: bool = False
    label: str = "legitimate"


# --- Name pools ---
SENDER_NAMES = [
    "Maria Santos", "Raj Patel", "Carlos Rivera", "Aisha Khan",
    "James Lee", "Priya Sharma", "Miguel Torres", "Fatima Ali",
    "David Chen", "Rosa Martinez",
]

RECIPIENT_NAMES = [
    "Elena Santos", "Vikram Patel", "Ana Rivera", "Omar Khan",
    "Wei Lee", "Sunita Sharma", "Diego Torres", "Amina Ali",
    "Lily Chen", "Isabel Martinez", "Pedro Garcia", "Noor Ahmed",
]


def _generate_user_profiles(n: int = 10) -> List[UserProfile]:
    """Generate synthetic user profiles with history."""
    profiles = []
    for i in range(n):
        corridor = random.choice(CORRIDORS)
        sender_country, recipient_country = corridor.split("-")
        typical_recipients = random.sample(RECIPIENT_NAMES, k=random.randint(1, 3))

        life_events = []
        if random.random() < 0.3:
            life_events.append({
                "event": random.choice([
                    "Sending money for sister's wedding",
                    "Monthly support for parents",
                    "Child's school tuition payment",
                    "Medical emergency for family member",
                ]),
                "date": (datetime.now() - timedelta(days=random.randint(30, 180))).isoformat(),
            })

        profiles.append(UserProfile(
            user_id=f"USR-{uuid.uuid4().hex[:8].upper()}",
            name=SENDER_NAMES[i % len(SENDER_NAMES)],
            country=sender_country,
            typical_amount=round(random.uniform(100, 800), 2),
            typical_recipients=typical_recipients,
            typical_corridor=corridor,
            registered_device_id=f"DEV-{uuid.uuid4().hex[:6].upper()}",
            registered_ip_prefix=f"{random.randint(10,200)}.{random.randint(0,255)}.{random.randint(0,255)}",
            life_events=life_events,
        ))
    return profiles


def generate_transactions(
    n_total: int = 100,
    fraud_ratio: float = 0.15,
    n_users: int = 10,
) -> tuple[List[Transaction], List[UserProfile]]:
    """Generate a mix of legitimate and fraudulent transactions."""
    users = _generate_user_profiles(n_users)
    transactions = []
    n_fraud = int(n_total * fraud_ratio)
    n_legit = n_total - n_fraud

    # --- Legitimate transactions ---
    for _ in range(n_legit):
        user = random.choice(users)
        amount = round(user.typical_amount * random.uniform(0.5, 1.5), 2)
        is_new = random.random() < 0.1
        recipient = (
            random.choice(RECIPIENT_NAMES) if is_new
            else random.choice(user.typical_recipients)
        )
        _, rcountry = user.typical_corridor.split("-")

        txn = Transaction(
            txn_id=f"TXN-{uuid.uuid4().hex[:10].upper()}",
            user_id=user.user_id,
            amount_usd=amount,
            recipient_name=recipient,
            recipient_country=rcountry,
            corridor=user.typical_corridor,
            device_id=user.registered_device_id,
            ip_address=f"{user.registered_ip_prefix}.{random.randint(1,254)}",
            timestamp=(datetime.now() - timedelta(
                hours=random.randint(1, 720)
            )).isoformat(),
            is_new_recipient=is_new,
            is_fraud=False,
            label="legitimate",
        )
        transactions.append(txn)

    # --- Fraudulent transactions ---
    for _ in range(n_fraud):
        user = random.choice(users)
        fraud_type = random.choice(["amount", "device", "country", "velocity"])
        amount = round(user.typical_amount * random.uniform(3.0, 10.0), 2)
        device = f"DEV-{uuid.uuid4().hex[:6].upper()}"
        ip = f"{random.randint(10,200)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        rcountry = random.choice(list(SANCTIONS_COUNTRIES)) if fraud_type == "country" else user.typical_corridor.split("-")[1]

        txn = Transaction(
            txn_id=f"TXN-{uuid.uuid4().hex[:10].upper()}",
            user_id=user.user_id,
            amount_usd=amount,
            recipient_name=random.choice(RECIPIENT_NAMES),
            recipient_country=rcountry,
            corridor=f"{user.country}-{rcountry}",
            device_id=device if fraud_type in ("device", "velocity") else user.registered_device_id,
            ip_address=ip if fraud_type in ("device", "velocity") else f"{user.registered_ip_prefix}.{random.randint(1,254)}",
            timestamp=(datetime.now() - timedelta(
                hours=random.randint(1, 48)
            )).isoformat(),
            is_new_recipient=True,
            is_fraud=True,
            label=f"fraud_{fraud_type}",
        )
        transactions.append(txn)

    random.shuffle(transactions)
    return transactions, users

