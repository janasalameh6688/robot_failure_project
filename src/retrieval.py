
"""Local maintenance retrieval and PostgreSQL-backed question answering."""

from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.database import get_connection
from src.maintenance_text import create_maintenance_description

DATA_PATH = Path("data/ai4i2020.csv")
INDEX_PATH = Path("models/maintenance_tfidf_index.joblib")

BASE_MODEL_NAME = "sshleifer/tiny-gpt2"
LORA_ADAPTER_PATH = Path("models/lora_maintenance_adapter")


def generate_grounded_answer(question: str, context: str) -> str:
    """Generate an answer using retrieved project evidence and the local LoRA model."""
    try:
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

        if LORA_ADAPTER_PATH.exists():
            model = PeftModel.from_pretrained(model, str(LORA_ADAPTER_PATH))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        prompt = (
            "Answer the question using only the evidence below. "
            "If the evidence is insufficient, say so. Do not invent facts.\n\n"
            f"Evidence:\n{context[:2500]}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return answer or "The local model did not generate a usable answer."

    except Exception as exc:
        return f"LLM answer generation unavailable: {exc}"


CAUSE_COLUMNS = {
    "TWF": "tool wear failure",
    "HDF": "heat dissipation failure",
    "PWF": "power failure",
    "OSF": "overstrain failure",
    "RNF": "random failure",
}

NUMERIC_COLUMNS = {
    "torque": ("Torque [Nm]", "torque_nm"),
    "tool wear": ("Tool wear [min]", "tool_wear_min"),
    "air temperature": ("Air temperature [K]", "air_temp_k"),
    "process temperature": ("Process temperature [K]", "process_temp_k"),
    "rotational speed": ("Rotational speed [rpm]", "rotational_speed_rpm"),
}


def create_documents(df: pd.DataFrame) -> list[dict]:
    """Create searchable documents without conflating failure labels and causes."""
    documents = []

    for _, row in df.iterrows():
        failure = int(row["Machine failure"])
        causes = [
            CAUSE_COLUMNS[column]
            for column in CAUSE_COLUMNS
            if column in df.columns and int(row[column]) == 1
        ]

        description = create_maintenance_description(row)
        cause_text = (
            "Cause flags: " + ", ".join(causes)
            if causes
            else "No cause flags recorded"
        )

        documents.append({
            "machine_id": int(row["UDI"]),
            "product_id": str(row["Product ID"]),
            "machine_type": str(row["Type"]),
            "failure": failure,
            "causes": causes,
            "sensors": {
                "torque": float(row["Torque [Nm]"]),
                "tool wear": float(row["Tool wear [min]"]),
                "air temperature": float(row["Air temperature [K]"]),
                "process temperature": float(row["Process temperature [K]"]),
                "rotational speed": float(row["Rotational speed [rpm]"]),
            },
            "text": (
                f"{description} "
                f"{'Machine failure label is 1.' if failure else 'Machine failure label is 0.'} "
                f"{cause_text}."
            ),
        })

    return documents


def build_index() -> None:
    """Build and persist the TF-IDF index and sensor metadata."""
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    documents = create_documents(df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([doc["text"] for doc in documents])

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": documents,
    }, INDEX_PATH)

    print(f"Indexed documents: {len(documents)}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Index saved to: {INDEX_PATH}")


def load_index() -> dict:
    """Load the persistent index."""
    if not INDEX_PATH.exists():
        raise RuntimeError("Index not found. Run 'python main.py index' first.")
    return joblib.load(INDEX_PATH)


def _requested_numeric_fields(question: str) -> list[str]:
    """Identify sensor measurements mentioned in a question."""
    q = question.lower()
    found = []

    aliases = {
        "torque": ("torque",),
        "tool wear": ("tool wear", "tool-wear", "wear"),
        "air temperature": ("air temperature", "air temp"),
        "process temperature": ("process temperature", "process temp"),
        "rotational speed": ("rotational speed", "rotation speed", "rpm", "speed"),
    }

    for field, terms in aliases.items():
        if any(term in q for term in terms):
            found.append(field)

    return found


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Retrieve by text similarity, with numeric-aware ranking when applicable."""
    index = load_index()
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    documents = index["documents"]

    query_vector = vectorizer.transform([question])
    text_scores = (matrix @ query_vector.T).toarray().ravel()

    fields = _requested_numeric_fields(question)
    q = question.lower()
    asks_high = any(word in q for word in ("high", "highest", "large", "elevated"))
    asks_low = any(word in q for word in ("low", "lowest", "small", "reduced"))

    # Use dataset percentiles to rank sensor values when the question
    # explicitly asks for high or low measurements.
    df = pd.read_csv(DATA_PATH)
    percentile_scores = []

    for doc in documents:
        scores = []
        for field in fields:
            column = NUMERIC_COLUMNS[field][0]
            value = doc["sensors"][field]

            if asks_high:
                score = float(value >= df[column].quantile(0.75))
            elif asks_low:
                score = float(value <= df[column].quantile(0.25))
            else:
                score = 0.0

            scores.append(score)

        percentile_scores.append(
            sum(scores) / len(scores) if scores else 0.0
        )

    percentile_scores = pd.Series(percentile_scores).to_numpy()

    if fields and (asks_high or asks_low):
        # Numeric criteria dominate, with text similarity as a tie-breaker.
        combined_scores = 0.85 * percentile_scores + 0.15 * text_scores
    else:
        combined_scores = text_scores

    ranked = combined_scores.argsort()[::-1]
    results = []

    for position in ranked:
        if combined_scores[position] <= 0:
            break

        item = documents[position].copy()
        item["score"] = float(combined_scores[position])
        item["text_score"] = float(text_scores[position])
        results.append(item)

        if len(results) >= top_k:
            break

    return results


def get_database_statistics() -> tuple[int, int, list[tuple]]:
    """Read machine and failure-cause event counts from PostgreSQL."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(DISTINCT m.machine_id),
                       COUNT(DISTINCT fe.failure_id)
                FROM machines AS m
                LEFT JOIN failure_events AS fe ON fe.machine_id = m.machine_id;
            """)
            total, failures = cur.fetchone()

            cur.execute("""
                SELECT fc.cause_code, fc.description,
                       COUNT(ec.failure_id) AS event_count
                FROM failure_causes AS fc
                LEFT JOIN event_causes AS ec ON ec.cause_code = fc.cause_code
                GROUP BY fc.cause_code, fc.description
                ORDER BY event_count DESC, fc.cause_code;
            """)
            causes = cur.fetchall()

        return int(total), int(failures), causes
    finally:
        conn.close()


def answer_question(question: str) -> None:
    """Answer with database statistics and relevant records."""
    if not question.strip():
        raise RuntimeError("Please enter a question after the ask command.")

    results = retrieve(question)
    context_parts = []

    try:
        total, failures, causes = get_database_statistics()
        context_parts.append(
            f"Database contains {total} machines and {failures} "
            f"recorded machine failure events."
        )
        context_parts.append(
            "Cause-flag counts (may overlap): "
            + "; ".join(
                f"{code} ({description}): {count}"
                for code, description, count in causes
            )
        )
    except RuntimeError:
        context_parts.append("Database statistics are unavailable.")

    for item in results:
        context_parts.append(
            f"Machine UDI {item['machine_id']}: "
            f"overall failure label={item['failure']}; "
            f"cause flags={item['causes']}; "
            f"sensor readings={item['sensors']}."
        )

    generated_answer = generate_grounded_answer(
        question, "\n".join(context_parts)
    )
    print(f"\nQuestion: {question}")
    print(f"Answer: {generated_answer}")

    try:
        total, failures, causes = get_database_statistics()
        rate = 100 * failures / total if total else 0
        print("\nDatabase statistics")
        print(f"Machines in database: {total}")
        print(f"Machine failure events: {failures}")
        print(f"Recorded machine-failure rate: {rate:.2f}%")
        print("Cause-flag event counts (may overlap):")
        for code, description, count in causes:
            print(f"  {code} — {description}: {count}")
    except RuntimeError as exc:
        print(f"\nDatabase statistics unavailable: {exc}")

    if not results:
        print("\nNo matching records found.")
        return

    print("\nMost relevant machine records")
    for item in results:
        print(
            f"\nMachine UDI {item['machine_id']} "
            f"(product {item['product_id']}, type {item['machine_type']})"
        )
        print(f"  Overall machine failure label: {item['failure']}")
        print(f"  Individual cause flags: {', '.join(item['causes']) or 'none'}")
        print(f"  Sensor values: {item['sensors']}")
        print(f"  Retrieval score: {item['score']:.4f}")

    print(
        "\nData-quality note: cause flags and the overall machine-failure label "
        "are separate source columns. Inconsistent combinations are reported "
        "as recorded, not treated as proof that the overall label is positive."
    )
    print(
        "Retrieval ranks records by text and, for explicit high/low sensor "
        "questions, dataset-relative numeric criteria. It does not establish causation."
    )
