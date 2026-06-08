import unittest

from scripts import generate_corpus


class GenerateCorpusTests(unittest.TestCase):
    def test_difficulty_pool_has_expected_size(self):
        pool = generate_corpus.build_difficulty_pool(seed=42)
        self.assertEqual(1000, len(pool))

    def test_single_language_generation_counts(self):
        rows = generate_corpus.build_samples_for_language(
            lang="ja",
            per_language=1000,
            seed=42,
            llm_url="",
            llm_api_key="",
            offline=True,
        )
        self.assertEqual(1000, len(rows))

        tier_counts = {"S": 0, "M": 0, "L": 0, "XL": 0}
        for row in rows:
            tier_counts[row["length_tier"]] += 1

        self.assertEqual({"S": 250, "M": 450, "L": 250, "XL": 50}, tier_counts)


if __name__ == "__main__":
    unittest.main()
