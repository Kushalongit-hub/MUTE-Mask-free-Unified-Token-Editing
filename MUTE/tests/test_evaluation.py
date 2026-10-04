import json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from mute.evaluation import source_mask, score


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((Path(__file__).resolve().parents[1]/"evaluation/frozen_v1/protocol.json").read_text())
        self.source = np.full((64,64,3),128,dtype=np.uint8)
        self.source[16:48,16:48]=[255,0,0]
        self.image=Image.fromarray(self.source)
        self.mask=source_mask(self.image,self.protocol)

    def test_perfect_recolor_passes_joint_proxy(self):
        target=self.source.copy();target[16:48,16:48]=[0,0,255]
        result=score(self.image,Image.fromarray(target),self.mask,"blue",self.protocol)
        self.assertEqual(result["target_color_coverage"],1)
        self.assertEqual(result["colored_region_iou"],1)
        self.assertEqual(result["colored_region_centroid_shift_px"],0)
        self.assertEqual(result["independent_color_mask_background_rgb_mae"],0)
        self.assertTrue(result["joint_proxy_pass"])

    def test_failed_edit_shift_and_background_drift_fail(self):
        failed=score(self.image,self.image,self.mask,"blue",self.protocol)
        self.assertFalse(failed["joint_proxy_pass"])
        self.assertEqual(failed["colored_region_iou"],0)
        self.assertIsNone(failed["colored_region_centroid_shift_px"])
        shifted=np.full_like(self.source,128);shifted[16:48,32:64]=[0,0,255]
        result=score(self.image,Image.fromarray(shifted),self.mask,"blue",self.protocol)
        self.assertEqual(result["colored_region_centroid_shift_px"],16)
        self.assertLess(result["colored_region_iou"],.7)
        self.assertFalse(result["joint_proxy_pass"])
        drift=self.source.copy();drift[self.mask]=[0,0,255];drift[0:8]=255
        self.assertFalse(score(self.image,Image.fromarray(drift),self.mask,"blue",self.protocol)["joint_proxy_pass"])

    def test_largest_source_component_and_invalid_mask(self):
        image=self.source.copy();image[0:2,0:2]=[255,0,0]
        self.assertTrue(np.array_equal(source_mask(Image.fromarray(image),self.protocol),self.mask))
        result=score(self.image,self.image,np.zeros((64,64),bool),"blue",self.protocol)
        self.assertFalse(result["source_valid"])
        self.assertFalse(result["joint_proxy_pass"])


if __name__ == "__main__":
    unittest.main()
