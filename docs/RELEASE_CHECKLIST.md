# Release Checklist

This checklist separates publishing the source repository from releasing trained model weights.

## Source repository publication

- [x] Apache License 2.0 is selected for repository source code. Confirm separate licenses for datasets, third-party assets, and trained weights before distributing those materials.
- [ ] Review every committed file for secrets, private data, and accidentally included datasets or weights.
- [ ] Review dependency versions and security advisories.
- [ ] Run the complete test suite locally on the supported Python/PyTorch environment.
- [ ] Verify the development configuration and artifact verifier on CPU.
- [ ] Review documentation, data provenance requirements, and contribution/security reporting paths.
- [ ] Publish the repository only after the owner approves its visibility.

## Trained model release

- [ ] Train from scratch using documented, permitted datasets.
- [ ] Preserve dataset manifests and exact tokenizer identity.
- [ ] Complete the configured training schedule or document the exact stopping reason.
- [ ] Verify the exported package with `arachne-small verify-artifact`.
- [ ] Run held-out Turkish and English evaluations and publish the exact methodology/results.
- [ ] Run contamination, memorization, safety, and robustness evaluations.
- [ ] Record hardware, software, training duration, throughput, and resource usage.
- [ ] Review the model card and release limitations.
- [ ] Publish model weights only after the model owner approves the artifact and its license/terms.

A source-code repository can be published before training is complete, but it must not be described as a trained or benchmarked model release.
