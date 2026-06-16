"""
Simple standalone PWWS implementation.

No TextAttack dependency. The script reads SST-2 from a local zip file, loads a
local HuggingFace sentiment classifier, uses NLTK WordNet for synonym
candidates, and runs the Probability Weighted Word Saliency attack.
"""

import argparse
import csv
import io
import random
import re
import zipfile

import torch


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "d", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "ll", "m", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "us", "ve", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "yet", "you", "your",
    "yours", "yourself", "yourselves",
}

DEFAULT_SST2_ZIP = "SST-2.zip"
DEFAULT_MODEL_PATH = ()
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def load_sst2_examples(zip_path, split, limit, shuffle=False, seed=1234):
    """Read labeled SST-2 examples from SST-2.zip."""
    if split == "test":
        raise ValueError("SST-2/test.tsv has no labels. Use dev or train.")

    examples = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"SST-2/{split}.tsv") as binary_file:
            text_file = io.TextIOWrapper(binary_file, encoding="utf-8")
            reader = csv.DictReader(text_file, delimiter="\t")
            for row in reader:
                examples.append({"text": row["sentence"], "label": int(row["label"])})

    if shuffle:
        random.Random(seed).shuffle(examples)
    return examples[:limit]


def load_wordnet(download_if_missing=False, language="eng"):
    """Load NLTK WordNet and check that the requested language exists."""
    import nltk
    from nltk.corpus import wordnet

    if download_if_missing:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    try:
        wordnet.synsets("test")
    except LookupError as exc:
        raise LookupError(
            "WordNet data is missing. Run with --download-wordnet first."
        ) from exc

    if language not in wordnet.langs():
        raise ValueError(f"WordNet language {language!r} is unavailable.")
    return wordnet


def get_wordnet_synonyms(word, wordnet, language="eng"):
    """Return sorted one-word WordNet synonyms for a word."""
    synonyms = set()
    for synset in wordnet.synsets(word, lang=language):
        for lemma in synset.lemma_names(lang=language):
            if lemma.lower() == word.lower() or "_" in lemma:
                continue
            if re.fullmatch(r"[A-Za-z]+", lemma):
                synonyms.add(lemma)
    return sorted(synonyms)


def load_model(model_name_or_path, device=None):
    """Load tokenizer and classifier from a local path or HuggingFace name."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_probabilities(tokenizer, model, device, texts, batch_size=16, max_length=128):
    """Run batched model inference and return class probabilities."""
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            all_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    return all_probs


def predict_counted(state, texts):
    """Predict and add the number of model queries to state."""
    state["queries"] += len(texts)
    return predict_probabilities(
        state["tokenizer"],
        state["model"],
        state["device"],
        texts,
        state["batch_size"],
        state["max_length"],
    )


def tokenize(text):
    """Split text into word and punctuation tokens."""
    return TOKEN_RE.findall(text)


def detokenize(tokens):
    """Join simple tokens back into readable text."""
    if not tokens:
        return ""

    no_space_before = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "'"}
    no_space_after = {"(", "[", "{", "$", "#", "'"}
    text = tokens[0]
    for token in tokens[1:]:
        if token in no_space_before or text[-1] in no_space_after:
            text += token
        else:
            text += " " + token
    return text


def is_modifiable(token):
    """Only alphabetic non-stopwords can be replaced."""
    return token.isalpha() and token.lower() not in STOPWORDS


def match_case(candidate, source):
    """Preserve simple casing when replacing a word."""
    if source.isupper():
        return candidate.upper()
    if source[:1].isupper():
        return candidate.capitalize()
    return candidate


def valid_synonyms(word, wordnet, max_candidates):
    """Get filtered WordNet candidates for one word."""
    candidates = []
    seen = set()
    for synonym in get_wordnet_synonyms(word, wordnet):
        key = synonym.lower()
        if key == word.lower() or key in seen:
            continue
        seen.add(key)
        candidates.append(synonym)
        if len(candidates) >= max_candidates:
            break
    return candidates


def attack_score(probs, true_label):
    """Untargeted attack score: larger means lower confidence in true_label."""
    return 1.0 - float(probs[true_label])


def rank_words(tokens, true_label, state):
    """Compute PWWS word order.

    PWWS score = softmax(leave-one-out saliency) * best synonym substitution
    score. Words with larger scores are attacked first.
    """
    modifiable_indices = [
        index for index, token in enumerate(tokens) if is_modifiable(token)
    ]
    if not modifiable_indices:
        return [], []

    leave_one_texts = []
    for index in modifiable_indices:
        masked_tokens = list(tokens)
        masked_tokens[index] = state["unk_token"]
        leave_one_texts.append(detokenize(masked_tokens))

    leave_one_probs = predict_counted(state, leave_one_texts)
    saliency_scores = torch.tensor(
        [attack_score(probs, true_label) for probs in leave_one_probs],
        dtype=torch.float,
    )
    saliency_weights = torch.softmax(saliency_scores, dim=0)

    best_synonym_scores = []
    for index in modifiable_indices:
        word = tokens[index]
        candidates = valid_synonyms(word, state["wordnet"], state["max_candidates"])
        if not candidates:
            best_synonym_scores.append(0.0)
            continue

        synonym_texts = []
        for candidate in candidates:
            new_tokens = list(tokens)
            new_tokens[index] = match_case(candidate, word)
            synonym_texts.append(detokenize(new_tokens))

        synonym_probs = predict_counted(state, synonym_texts)
        scores = [attack_score(probs, true_label) for probs in synonym_probs]
        best_synonym_scores.append(max(scores))

    weighted_scores = saliency_weights * torch.tensor(
        best_synonym_scores, dtype=torch.float
    )
    sorted_positions = torch.argsort(weighted_scores, descending=True).tolist()
    word_order = [modifiable_indices[position] for position in sorted_positions]
    word_ranking = [
        (
            modifiable_indices[position],
            tokens[modifiable_indices[position]],
            float(weighted_scores[position]),
        )
        for position in sorted_positions
    ]
    return word_order, word_ranking


def attack_one(text, true_label, base_state):
    """Run PWWS on one text and return a plain result dictionary."""
    state = dict(base_state)
    state["queries"] = 0

    tokens = tokenize(text)
    original_probs = predict_counted(state, [text])[0]
    original_label = int(torch.tensor(original_probs).argmax().item())

    if original_label != true_label:
        return {
            "original_text": text,
            "adversarial_text": text,
            "original_label": original_label,
            "adversarial_label": original_label,
            "success": False,
            "skipped": True,
            "queries": state["queries"],
            "changed_words": [],
            "word_ranking": [],
        }

    current_tokens = list(tokens)
    current_score = attack_score(original_probs, true_label)
    word_order, word_ranking = rank_words(tokens, true_label, state)
    changed_words = []

    for index in word_order:
        word = current_tokens[index]
        candidates = valid_synonyms(word, state["wordnet"], state["max_candidates"])
        if not candidates:
            continue

        candidate_tokens = []
        candidate_texts = []
        for candidate in candidates:
            new_tokens = list(current_tokens)
            new_tokens[index] = match_case(candidate, word)
            candidate_tokens.append(new_tokens)
            candidate_texts.append(detokenize(new_tokens))

        candidate_probs = predict_counted(state, candidate_texts)
        scored_candidates = []
        for probs, new_tokens, new_text in zip(
            candidate_probs, candidate_tokens, candidate_texts
        ):
            scored_candidates.append(
                (attack_score(probs, true_label), probs, new_tokens, new_text)
            )

        best_score, best_probs, best_tokens, best_text = max(
            scored_candidates, key=lambda item: item[0]
        )
        if best_score <= current_score:
            continue

        changed_words.append((current_tokens[index], best_tokens[index], index))
        current_tokens = best_tokens
        current_score = best_score
        new_label = int(torch.tensor(best_probs).argmax().item())

        if new_label != true_label:
            return {
                "original_text": text,
                "adversarial_text": best_text,
                "original_label": original_label,
                "adversarial_label": new_label,
                "success": True,
                "skipped": False,
                "queries": state["queries"],
                "changed_words": changed_words,
                "word_ranking": word_ranking,
            }

    final_text = detokenize(current_tokens)
    final_probs = predict_counted(state, [final_text])[0]
    final_label = int(torch.tensor(final_probs).argmax().item())
    return {
        "original_text": text,
        "adversarial_text": final_text,
        "original_label": original_label,
        "adversarial_label": final_label,
        "success": False,
        "skipped": False,
        "queries": state["queries"],
        "changed_words": changed_words,
        "word_ranking": word_ranking,
    }


def evaluate(examples, state):
    """Attack each example and print per-example results plus summary."""
    clean_correct = 0
    successful = 0
    skipped = 0
    total_queries = 0
    total_changed_words = 0

    for index, example in enumerate(examples):
        result = attack_one(example["text"], example["label"], state)
        if result["skipped"]:
            skipped += 1
            status = "SKIP"
        elif result["success"]:
            clean_correct += 1
            successful += 1
            status = "SUCCESS"
        else:
            clean_correct += 1
            status = "FAILED"

        if not result["skipped"]:
            total_queries += result["queries"]
            total_changed_words += len(result["changed_words"])

        print(f"[{index:03d}] {status}")
        print("original:", result["original_text"])
        print("adversarial:", result["adversarial_text"])
        print("labels:", result["original_label"], "->", result["adversarial_label"])
        print("changed_words:", result["changed_words"])
        print("queries:", result["queries"])
        print("-" * 80)

    failed = clean_correct - successful
    summary = {
        "total": len(examples),
        "clean_correct": clean_correct,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "attack_success_rate": successful / clean_correct if clean_correct else 0.0,
        "avg_queries": total_queries / clean_correct if clean_correct else 0.0,
        "avg_changed_words": total_changed_words / clean_correct if clean_correct else 0.0,
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run simple standalone PWWS.")
    parser.add_argument("--sst2-zip", default=DEFAULT_SST2_ZIP)
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--download-wordnet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer, model, device = load_model(args.model_name_or_path)
    wordnet = load_wordnet(args.download_wordnet)
    examples = load_sst2_examples(
        args.sst2_zip,
        args.split,
        args.limit,
        args.shuffle,
        args.seed,
    )
    state = {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "wordnet": wordnet,
        "max_candidates": args.max_candidates,
        "unk_token": "[UNK]",
        "queries": 0,
    }
    summary = evaluate(examples, state)

    print("total:", summary["total"])
    print("clean_correct:", summary["clean_correct"])
    print("successful:", summary["successful"])
    print("failed:", summary["failed"])
    print("skipped:", summary["skipped"])
    print("attack_success_rate:", round(summary["attack_success_rate"], 4))
    print("avg_queries:", round(summary["avg_queries"], 2))
    print("avg_changed_words:", round(summary["avg_changed_words"], 2))


if __name__ == "__main__":
    main()
