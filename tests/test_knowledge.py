import json
import tempfile
import unittest
from pathlib import Path

from agent_forge_lab.knowledge import Document, KnowledgeBase, keywords

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "synthetic_kb.json"


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.from_json(SAMPLE)

    def test_fixture_loads(self):
        self.assertEqual(len(self.kb.ids()), 6)
        self.assertIn("INJECT-106", self.kb.ids())

    def test_search_relevant_documents(self):
        docs = self.kb.search("failed login attempts staging server")
        self.assertEqual(docs[0].doc_id, "INC-101")
        self.assertLessEqual(len(docs), 3)

    def test_search_casefold_and_stable(self):
        self.assertEqual(self.kb.search("LOGIN INVESTIGATION"),
                         self.kb.search("login investigation"))

    def test_search_no_match(self):
        self.assertEqual(self.kb.search("xylophonist xylophonic"), ())

    def test_search_stopwords(self):
        self.assertFalse(keywords("the and for in what"))

    def test_read_exact_document_id(self):
        doc = self.kb.read("INC-102")
        self.assertIn("Maintenance", doc.text)
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.kb.read("INC-999")

    def test_duplicate_ids_rejected(self):
        x = Document("DOC-1", "title", "text")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            KnowledgeBase([x, x])

    def test_bad_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            KnowledgeBase([Document("../secret", "title", "text")])

    def test_blank_or_oversized_document_rejected(self):
        for doc in [Document("DOC-1", "", "text"), Document("DOC-1", "title", "x" * 8001)]:
            with self.subTest(doc=doc):
                with self.assertRaises(ValueError):
                    KnowledgeBase([doc])

    def test_invalid_query_limits(self):
        for query in ["", "  ", "x" * 241]:
            with self.subTest(query=query[:10]):
                with self.assertRaises(ValueError):
                    self.kb.search(query)
        for limit in [0, 6, True]:
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    self.kb.search("login", limit=limit)

    def test_schema_rejects_unknown_keys(self):
        data = json.loads(SAMPLE.read_text())
        data["documents"][0]["secret_command"] = "delete"
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "data.json"
            file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "schema"):
                KnowledgeBase.from_json(file)

    def test_empty_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "data.json"
            file.write_text(json.dumps({"dataset": "x", "documents": []}))
            with self.assertRaisesRegex(ValueError, "bounded"):
                KnowledgeBase.from_json(file)


if __name__ == "__main__":
    unittest.main()
