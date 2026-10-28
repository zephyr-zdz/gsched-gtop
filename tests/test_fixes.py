import unittest

from support import core, gtop, host, card, node_view, planned


class ReviewFixTests(unittest.TestCase):
    def test_packing_onto_a_card_the_client_holds_stays_within_its_cap(self):
        d = {"clients": {"bot": {"max_gpus": {"node": 1}}}}
        v = node_view(client_cards={"bot": {0}}, client_gpus={"bot": 1})
        spec = {"client": "bot", "gpus": 1, "gpu_mem_gb": 4}
        self.assertTrue(core.within_client_cap(spec, "node", v, d, {}, [0]))
        self.assertFalse(core.within_client_cap(spec, "node", v, d, {}, [1]))
        self.assertFalse(core.within_client_cap({"client": "bot", "gpus": 1}, "node", v, d, {}))

    def test_gpu_mem_filters_whole_idle_cards(self):
        v = node_view(gpus=[card(0, total=16 * 1024), card(1, total=48 * 1024)])
        d = {"host_mem_reserve_gb": 0}
        r = core.place_node({"gpus": 1, "gpu_mem_gb": 30}, host(), v, [], d)
        self.assertEqual(r[1], [1])
        v = node_view(gpus=[card(0, total=16 * 1024)])
        self.assertIsNone(core.place_node({"gpus": 1, "gpu_mem_gb": 30}, host(), v, [], d))

    def test_failed_squeue_is_reported(self):
        v = core.slurm_view({"nodes": "", "squeue": "", "squeue_ok": False})
        self.assertFalse(v["squeue_ok"])
        self.assertTrue(core.slurm_view({"nodes": "", "squeue": ""})["squeue_ok"])

    def test_smi_rows_skip_error_text(self):
        good = "0, GPU-a, 1, 2, 3, X, 40, 30, 300, 30, 1000"
        rows = gtop.smi_rows("Unable to determine the device handle\n" + good + "\n1, short\n")
        self.assertEqual(list(rows), ["0"])


if __name__ == "__main__":
    unittest.main()
