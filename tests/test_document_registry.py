import tempfile
import unittest
from pathlib import Path

from grounded.document_registry import DocumentRegistry


class DocumentRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = (
            Path(self.temporary_directory.name) / "documents.sqlite3"
        )
        self.registry = DocumentRegistry(database_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_add_get_list_and_delete_document(self) -> None:
        added = self.registry.add_document(
            document_id="document-1",
            filename="example.pdf",
            stored_path="data/uploads/document-1.pdf",
            page_count=3,
            chunk_ids=["document-1:0", "document-1:1"],
        )

        self.assertEqual(added.chunk_count, 2)
        self.assertEqual(
            self.registry.get_document("document-1"),
            added,
        )
        self.assertEqual(
            self.registry.list_documents(),
            [added],
        )

        self.assertTrue(self.registry.delete_document("document-1"))
        self.assertIsNone(self.registry.get_document("document-1"))
        self.assertFalse(self.registry.delete_document("document-1"))


if __name__ == "__main__":
    unittest.main()
