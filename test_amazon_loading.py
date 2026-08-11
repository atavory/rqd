import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_wsdm_web_recsys import _load_amazon_arrays


ROWS = [
    ("u1", "i1", 5.0, 1000),
    ("u1", "i2", 4.0, 2000),
    ("u2", "i1", 3.0, 3000),
]


class AmazonLoadingTest(unittest.TestCase):
    def check_arrays(self, arrays):
        users, items, ratings, timestamps, n_users, n_items = arrays
        self.assertEqual((n_users, n_items), (2, 2))
        np.testing.assert_array_equal(users, [0, 0, 1])
        np.testing.assert_array_equal(items, [0, 1, 0])
        np.testing.assert_allclose(ratings, [5.0, 4.0, 3.0])
        np.testing.assert_array_equal(timestamps, [1000, 2000, 3000])

    def test_amazon_2023_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Books.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["user_id", "parent_asin", "rating", "timestamp"],
                )
                writer.writeheader()
                for user, item, rating, timestamp in ROWS:
                    writer.writerow({
                        "user_id": user,
                        "parent_asin": item,
                        "rating": rating,
                        "timestamp": timestamp,
                    })
            self.check_arrays(_load_amazon_arrays(path, core_passes=0))

    def test_amazon_2018_jsonl_gz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Electronics.json.gz"
            with gzip.open(path, "wt") as output:
                for user, item, rating, timestamp in ROWS:
                    output.write(json.dumps({
                        "reviewerID": user,
                        "asin": item,
                        "overall": rating,
                        "unixReviewTime": timestamp,
                    }) + "\n")
            self.check_arrays(_load_amazon_arrays(path, core_passes=0))


if __name__ == "__main__":
    unittest.main()
