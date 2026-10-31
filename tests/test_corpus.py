import json

from arachne_small.corpus import prepare_corpus


def test_prepare_corpus_deduplicates_and_writes_manifest(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [{"text": f"Belge numarası {index} için lisanslı örnek içerik."} for index in range(200)]
    rows.append({"text": "  " + rows[0]["text"] + "  "})
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    output = tmp_path / "prepared"
    manifest = prepare_corpus(
        str(source), str(output), validation_fraction=0.2, seed=42,
        source="User-provided licensed corpus", license_name="CC-BY-4.0",
    )

    assert manifest["train_documents"] + manifest["validation_documents"] == 200
    assert manifest["duplicates_removed"] == 1
    assert (output / "train.jsonl").is_file()
    assert (output / "validation.jsonl").is_file()
    saved_manifest = json.loads((output / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["input_sha256"] == manifest["input_sha256"]


def test_prepare_corpus_split_is_deterministic(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"text": f"unique document {index}"}) + "\n" for index in range(300)),
        encoding="utf-8",
    )
    first = prepare_corpus(str(source), str(tmp_path / "first"), validation_fraction=0.15,
                           seed=19, source="test source", license_name="test license")
    second = prepare_corpus(str(source), str(tmp_path / "second"), validation_fraction=0.15,
                            seed=19, source="test source", license_name="test license")
    assert first["train_documents"] == second["train_documents"]
    assert first["validation_documents"] == second["validation_documents"]
    assert (tmp_path / "first" / "validation.jsonl").read_bytes() == (tmp_path / "second" / "validation.jsonl").read_bytes()


def test_export_text_writes_one_normalized_document_per_line(tmp_path):
    from arachne_small.corpus import export_text

    source = tmp_path / "documents.jsonl"
    source.write_text(
        json.dumps({"text": "  Türkçe   metin\\nikinci satır  "}, ensure_ascii=False) + "\\n",
        encoding="utf-8",
    )
    output = tmp_path / "tokenizer" / "train.txt"
    count = export_text(str(source), str(output))
    assert count == 1
    assert output.read_text(encoding="utf-8") == "Türkçe metin ikinci satır\\n"
