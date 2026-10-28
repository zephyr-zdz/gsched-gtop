import unittest

from support import gtop


class FormattingTests(unittest.TestCase):
    def test_expand_nodes(self):
        for text, expected in [
            (None, []), ("", []), ("node01", ["node01"]),
            ("gn[01-03,05],x1", ["gn01", "gn02", "gn03", "gn05", "x1"]),
            ("a[1-3],b[09-10]", ["a1", "a2", "a3", "b09", "b10"]),
        ]:
            with self.subTest(text=text):
                self.assertEqual(gtop.expand_nodes(text), expected)

    def test_bar_clamping_and_fractional_cell(self):
        for frac, width, expected in [
            (-1, 4, "····"), (0, 4, "····"), (0.5, 4, "██··"),
            (0.375, 4, "█▌··"), (1, 4, "████"), (2, 4, "████"), (0.5, 0, ""),
        ]:
            with self.subTest(frac=frac, width=width):
                self.assertEqual(gtop.bar(frac, width), expected)

    def test_spark_padding_recent_samples_and_saturation(self):
        self.assertEqual(gtop.spark([], 4), ("····", ""))
        self.assertEqual(gtop.spark([0, 50, 100], 5), ("··", " ▅█"))
        self.assertEqual(gtop.spark([1, 0, 25, 200], 3), ("", " ▃█"))
        self.assertEqual(gtop.spark(iter([12.5, 87.5]), 2), ("", "▂█"))

    def test_hms_boundaries(self):
        for seconds, expected in [(0, "0m00s"), (59.9, "0m59s"), (60, "1m00s"),
                                  (3599, "59m59s"), (3600, "1h00m"), (3660, "1h01m"),
                                  (86399, "23h59m"), (86400, "1d00h"), (176400, "2d01h")]:
            with self.subTest(seconds=seconds):
                self.assertEqual(gtop.hms(seconds), expected)

    def test_slurm_real_idx_numeric_order(self):
        self.assertEqual(gtop.slurm_real_idx({"step_gpus": "10,2", "gpuinfo": {"0": [], "1": []}}),
                         {"0": "2", "1": "10"})
        self.assertEqual(gtop.slurm_real_idx({"step_gpus": "3", "gpuinfo": {"0": []}}), {"0": "3"})

    def test_slurm_real_idx_missing_invalid_or_mismatched(self):
        for data in [{}, {"step_gpus": "GPU-uuid", "gpuinfo": {"0": []}},
                     {"step_gpus": "1,2", "gpuinfo": {"0": []}}]:
            with self.subTest(data=data):
                self.assertEqual(gtop.slurm_real_idx(data), {})
