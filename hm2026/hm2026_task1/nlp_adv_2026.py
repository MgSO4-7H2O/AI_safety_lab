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
DEFAULT_MODEL_PATH = ("")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def load_sst2_examples(zip_path, split, limit, shuffle=False, seed=1234):
    """Read SST-2 examples from the provided zip file."""
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
    """Load NLTK WordNet."""
    import nltk
    from nltk.corpus import wordnet

    if download_if_missing:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    wordnet.synsets("test")
    if language not in wordnet.langs():
        raise ValueError(f"WordNet language {language!r} is unavailable.")
    return wordnet


def get_wordnet_synonyms(word, wordnet, language="eng"):
    """TODO 1: return filtered one-word WordNet synonyms.

    Requirements:
    1. Use wordnet.synsets(word, lang=language).
    2. Iterate lemma_names(lang=language).
    3. Remove the original word, lemmas containing "_", and non-English words.
    4. Return sorted candidates without duplicates.
    """
    original = word.lower()
    synonyms = set()

    for synset in wordnet.synsets(word, lang=language):
        for lemma in synset.lemma_names(lang=language):
            candidate = lemma.lower()

            # WordNet uses "_" for multi-word expressions, e.g. "well_done".
            if "_" in candidate:
                continue

            # Keep only English alphabetic single words.
            if not candidate.isalpha():
                continue

            # Do not return the original word itself.
            if candidate == original:
                continue

            synonyms.add(candidate)

    return sorted(synonyms)


def load_model(model_name_or_path, device=None):
    """Load tokenizer and classification model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_probabilities(tokenizer, model, device, texts, batch_size=16, max_length=128):
    """Return model probabilities for a list of texts."""
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
    """Predict and count model queries."""
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
    """Split text into simple word/punctuation tokens."""
    return TOKEN_RE.findall(text)


def detokenize(tokens):
    """Join simple tokens back into text."""
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
    """Only alphabetic non-stopwords can be modified."""
    return token.isalpha() and token.lower() not in STOPWORDS


def match_case(candidate, source):
    """Preserve simple casing when replacing words."""
    if source.isupper():
        return candidate.upper()
    if source[:1].isupper():
        return candidate.capitalize()
    return candidate


def valid_synonyms(word, wordnet, max_candidates):
    """Filter synonym candidates and limit their number."""
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
    """TODO 2: implement the untargeted attack score.

    PWWS wants to reduce P(true_label), so the score should be larger when the
    model is less confident in the correct label.
    """
    return 1.0 - probs[true_label]


def rank_words(tokens, true_label, state):
    """TODO 3: compute PWWS ranking for all modifiable words.

    Required formula:
        PWWS score = softmax(leave-one-out saliency) * best synonym score

    Return:
        word_order: token indices sorted by descending PWWS score
        word_ranking: list of (index, word, score), for report analysis
    """
    modifiable_indices = [
        index for index, token in enumerate(tokens) if is_modifiable(token)
    ]
    if not modifiable_indices:
        return [], []

    # TODO 3.1: replace each candidate word with state["unk_token"], build
    # leave-one-out texts, and query them with predict_counted(...).
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

    # TODO 3.3: for each candidate word, try its valid synonyms and keep the
    # largest synonym attack score.
    ranking_items = []

    for position, index in enumerate(modifiable_indices):
        word = tokens[index]
        synonyms = valid_synonyms(
            word,
            state["wordnet"],
            state["max_candidates"],
        )

        if not synonyms:
            best_synonym_score = 0.0
        else:
            synonym_texts = []
            for synonym in synonyms:
                replaced_tokens = list(tokens)
                replaced_tokens[index] = match_case(synonym, word)
                synonym_texts.append(detokenize(replaced_tokens))

            synonym_probs = predict_counted(state, synonym_texts)
            best_synonym_score = max(
                attack_score(probs, true_label) for probs in synonym_probs
            )

        pwws_score = saliency_weights[position].item() * best_synonym_score

        ranking_items.append({
            "index": index,
            "word": word,
            "score": pwws_score,
        })
    # TODO 3.4: multiply saliency weights and best synonym scores, sort in
    # descending order, and return word_order plus word_ranking.
    ranking_items.sort(key=lambda item: item["score"], reverse=True)

    word_order = [item["index"] for item in ranking_items]
    word_ranking = [
        (item["index"], item["word"], item["score"])
        for item in ranking_items
    ]

    return word_order, word_ranking


def attack_one(text, true_label, base_state):
    """TODO 4: greedily attack one example using the PWWS word order.

    Steps:
    1. Predict clean label; if wrong, return a skipped result.
    2. Rank words with rank_words(...).
    3. For each word in order, test its synonym substitutions.
    4. Keep a substitution only if it improves attack_score.
    5. Stop once the predicted label changes.
    """
    state = dict(base_state)
    state["queries"] = 0

    # 1. Predict the clean example.
    clean_probs = predict_counted(state, [text])[0]
    clean_label = int(torch.tensor(clean_probs).argmax().item())

    result = {
        "original_text": text,
        "adversarial_text": text,
        "original_label": clean_label,
        "adversarial_label": clean_label,
        "success": False,
        "skipped": False,
        "queries": 0,
        "changed_words": [],
        "word_ranking": [],
    }

    # If the clean example is already misclassified, skip it.
    if clean_label != true_label:
        result["skipped"] = True
        result["queries"] = state["queries"]
        return result

    tokens = tokenize(text)
    current_tokens = list(tokens)
    current_score = attack_score(clean_probs, true_label)

    # 2. Rank words by PWWS score.
    word_order, word_ranking = rank_words(tokens, true_label, state)
    result["word_ranking"] = word_ranking

    # 3. Greedily replace words following the PWWS order.
    for index in word_order:
        original_word = current_tokens[index]
        synonyms = valid_synonyms(
            original_word,
            state["wordnet"],
            state["max_candidates"],
        )

        if not synonyms:
            continue

        candidate_texts = []
        candidate_words = []

        for synonym in synonyms:
            candidate_tokens = list(current_tokens)
            replacement = match_case(synonym, original_word)
            candidate_tokens[index] = replacement

            candidate_texts.append(detokenize(candidate_tokens))
            candidate_words.append(replacement)

        candidate_probs = predict_counted(state, candidate_texts)

        best_position = None
        best_score = current_score
        best_label = clean_label

        for position, probs in enumerate(candidate_probs):
            score = attack_score(probs, true_label)

            if score > best_score:
                best_position = position
                best_score = score
                best_label = int(torch.tensor(probs).argmax().item())

        # 4. Accept the substitution only if it improves the attack score.
        if best_position is None:
            continue

        new_word = candidate_words[best_position]
        current_tokens[index] = new_word
        current_score = best_score

        result["changed_words"].append(
            (original_word, new_word, index)
        )

        adversarial_text = detokenize(current_tokens)
        result["adversarial_text"] = adversarial_text
        result["adversarial_label"] = best_label

        # 5. Stop once the prediction changes.
        if best_label != true_label:
            result["success"] = True
            result["queries"] = state["queries"]
            return result

    result["queries"] = state["queries"]
    result["adversarial_text"] = detokenize(current_tokens)
    return result


def evaluate(examples, state):
    """Run attack and print results."""
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

    return {
        "total": len(examples),
        "clean_correct": clean_correct,
        "successful": successful,
        "failed": clean_correct - successful,
        "skipped": skipped,
        "attack_success_rate": successful / clean_correct if clean_correct else 0.0,
        "avg_queries": total_queries / clean_correct if clean_correct else 0.0,
        "avg_changed_words": total_changed_words / clean_correct if clean_correct else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="NLP adversarial attack task: PWWS.")
    parser.add_argument("--sst2-zip", default=DEFAULT_SST2_ZIP)
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=10)
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
