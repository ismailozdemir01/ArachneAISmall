# Dataset Intake and Governance

ArachneAISmall does not bundle third-party corpora. Dataset owners must provide the training material and document the right to use it for model training.

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
