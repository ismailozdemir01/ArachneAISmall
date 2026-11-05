# Dataset Intake and Governance

ArachneAISmall does not bundle third-party corpora. It includes an autonomous Wikimedia Wikipedia acquisition adapter that runs during `arachne-small serve`; other sources still require explicit source and license records.

## Required intake record

For every source, retain:

- Source name, publisher/owner, URL or acquisition record, and version/date.
- Exact license or written permission, including whether machine-learning training is permitted.
- Collection date and responsible operator.
- Language and document counts.
- Filtering, normalization, deduplication, and exclusion rules.
- Personal-data review, consent or other lawful basis where required, and retention/deletion constraints.
- SHA-256 of the original source and the prepared train/validation files.

Do not treat a public URL, public visibility, or the ability to download a file as permission to train on it.

## Processing rules

1. Keep raw and processed corpora outside the Git repository.
2. Normalize Unicode and whitespace consistently.
3. Deduplicate before splitting.
4. Split at document level before tokenizer training and token encoding.
5. Train the tokenizer only on the training split.
6. Encode validation data with the same tokenizer but never use it for gradient updates.
7. Keep the generated dataset manifest with the local dataset artifacts.
8. Exclude credentials, private customer content, confidential material, and personal data without an approved lawful basis.

The built-in corpus manifest records source, declared license, source checksum, split seed, document counts, and duplicate/skipped counts. It is provenance metadata, not a legal determination or proof that the declared license is valid.

## User correction feedback

The inference UI exposes a separate correction form. Nothing is collected from a chat unless the user explicitly checks the consent box and submits a correction. Feedback is stored only in the configured local SQLite database. Intake rejects obvious credentials/contact information, exact duplicate submissions, and corrections that repeat the original answer. Eligible records are exported with per-record hashes and provenance; only the corrected answer is used as the target text.

A rating is a triage signal, not a factuality guarantee. Before a learning cycle, operators must review the correction corpus for quality, poisoning, rights, and privacy. Exported records are marked consumed only after a candidate passes evaluation and is promoted. Failed cycles leave the records available for retry. See [Controlled Continual Learning](CONTINUAL_LEARNING.md).


## Autonomous Wikimedia acquisition

The built-in self-taught worker retrieves Turkish and English Wikipedia random article summaries. For every page it separately requests the MediaWiki REST `/page/{title}/bare` metadata and accepts only an exact CC BY-SA 3.0 or CC BY-SA 4.0 license URL. Missing or unrecognized licenses are rejected. The record preserves the page URL, page history URL, page ID, revision ID, license title/URL, language, retrieval time and the fact that the extract was normalized and language-prefixed.

The worker also excludes pages categorized as living people and drops extracts matching obvious credential/contact patterns. These are limited automated filters; they do not establish that all personal data or third-party rights issues have been removed. Every acquired record remains traceable in the raw local JSONL file.

Wikimedia content reuse requires attribution and, for ShareAlike material, compliance with applicable ShareAlike conditions. The source crawler's technical license allowlist is not legal advice and does not resolve whether a trained model or its weights constitute adapted material. Before distributing weights, datasets, or model outputs, review the relevant page-level rights and obligations for the intended jurisdiction and release. See [Wikimedia content reuse](https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse) and [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Raw corpora and model artifacts remain local and must not be committed to Git. The self-taught worker never uses raw chat traffic or its own generated responses as training data.
