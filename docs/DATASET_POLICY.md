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


## Autonomous global web research

The built-in self-taught worker uses a global web research engine. It is not tied to Wikimedia, a Wikimedia account, or a single encyclopedia. Search results can lead to public HTTPS pages across different publishers and domains. The crawler observes each origin's robots policy, applies per-host pacing and response-size limits, extracts HTML text without executing scripts, and records canonical URL, domain, title, retrieval time and content hash.

Research discovery and training eligibility are separate decisions. A publicly accessible page is not automatically eligible for model training. The training corpus accepts only pages that explicitly declare an exact CC0, CC BY or CC BY-SA license URL from the code's allowlist. Missing, ambiguous or incompatible reuse metadata causes exclusion from gradient training. The record preserves the declared license and source URL for attribution and audit.

The crawler excludes obvious credential/contact patterns and never logs in, bypasses access controls, runs downloaded code, or fetches arbitrary binary artifacts. Robots denial or inability to retrieve robots policy prevents crawling that origin. These automated checks are conservative technical controls, not a legal opinion or proof that every embedded element has the same rights status. Before distributing weights or datasets, review attribution, ShareAlike, privacy and model-output obligations for the actual source material and jurisdiction.

Wikimedia may be discovered as one source among many, but it is no longer the required source or the system's acquisition bottleneck. Raw corpora and model artifacts remain local and must not be committed to Git. The self-taught worker never uses raw chat traffic or its own generated responses as training data.
