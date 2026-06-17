import unittest

from reasoning import skills
from reasoning.skills import text as t

from reasoning.skills.tests._helpers import check_examples


class TestText(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("text."):
                check_examples(self, skill)

    def test_empty_collections(self):
        self.assertEqual(t.length([]), 0)
        self.assertEqual(t.unique([]), [])
        self.assertEqual(t.sort_items([]), [])

    def test_sort_descending(self):
        self.assertEqual(t.sort_items([1, 3, 2], reverse=True), [3, 2, 1])

    def test_unique_preserves_order(self):
        self.assertEqual(t.unique([3, 1, 3, 2, 1]), [3, 1, 2])


if __name__ == "__main__":
    unittest.main()
