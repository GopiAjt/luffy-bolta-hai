import unittest

import numpy as np

from app.utils.generate_slideshow import (
    cube_rotation_effect,
    iris_wipe_effect,
    ken_burns_headroom_for_motion,
    ken_burns_viewport,
    motion_blur_slide_effect,
    radial_wipe_effect,
    transition_progress,
    whip_pan_effect,
)
from app.utils.visual_effects import choose_motion_preset


class SlideshowEffectsTests(unittest.TestCase):
    def setUp(self):
        self.current = np.full((24, 32, 3), (20, 40, 80), dtype=np.uint8)
        self.next = np.full((24, 32, 3), (180, 160, 120), dtype=np.uint8)

    def assert_frame_equal(self, actual, expected):
        self.assertTrue(np.array_equal(actual, expected))

    def test_transition_progress_includes_clean_endpoints(self):
        self.assertEqual(transition_progress(0, 12), 0.0)
        self.assertEqual(transition_progress(11, 12), 1.0)

    def test_feathered_wipes_do_not_leak_next_frame_at_endpoints(self):
        for effect in (iris_wipe_effect, radial_wipe_effect):
            with self.subTest(effect=effect.__name__, progress=0):
                self.assert_frame_equal(effect(self.current, self.next, 0.0), self.current)
            with self.subTest(effect=effect.__name__, progress=1):
                self.assert_frame_equal(effect(self.current, self.next, 1.0), self.next)

    def test_directional_effects_preserve_start_and_end_frames(self):
        effects = (
            cube_rotation_effect,
            motion_blur_slide_effect,
            whip_pan_effect,
        )
        for effect in effects:
            for direction in ("left", "right", "up", "down"):
                with self.subTest(effect=effect.__name__, direction=direction, progress=0):
                    self.assert_frame_equal(
                        effect(self.current, self.next, 0.0, direction=direction),
                        self.current,
                    )
                with self.subTest(effect=effect.__name__, direction=direction, progress=1):
                    self.assert_frame_equal(
                        effect(self.current, self.next, 1.0, direction=direction),
                        self.next,
                    )

    def test_directional_effects_have_real_midpoint_blend(self):
        for effect in (cube_rotation_effect, motion_blur_slide_effect, whip_pan_effect):
            frame = effect(self.current, self.next, 0.5, direction="right")
            self.assertEqual(frame.shape, self.current.shape)
            self.assertFalse(np.array_equal(frame, self.current))
            self.assertFalse(np.array_equal(frame, self.next))

    def test_ken_burns_headroom_matches_motion_intensity(self):
        self.assertLess(
            ken_burns_headroom_for_motion("hold_still"),
            ken_burns_headroom_for_motion("slow_push"),
        )
        self.assertLess(
            ken_burns_headroom_for_motion("slow_push"),
            ken_burns_headroom_for_motion("impact_zoom"),
        )

    def test_pull_out_viewport_stays_centered(self):
        x, y, view_w, view_h = ken_burns_viewport(
            scaled_w=400,
            scaled_h=600,
            resolution=(100, 150),
            idx=0,
            t=0.25,
            motion="pull_out",
            pan_strength=1.0,
        )
        self.assertAlmostEqual(x + view_w / 2.0, 200.0)
        self.assertAlmostEqual(y + view_h / 2.0, 300.0)

    def test_zoom_only_ken_burns_motions_stay_centered(self):
        for motion in ("slow_push", "impact_zoom", "hold_still", "pull_out"):
            with self.subTest(motion=motion):
                x, y, view_w, view_h = ken_burns_viewport(
                    scaled_w=400,
                    scaled_h=600,
                    resolution=(100, 150),
                    idx=0,
                    t=0.5,
                    motion=motion,
                    pan_strength=1.0,
                )
                self.assertAlmostEqual(x + view_w / 2.0, 200.0)
                self.assertAlmostEqual(y + view_h / 2.0, 300.0)

    def test_stable_pan_alternates_horizontal_direction(self):
        even_start = ken_burns_viewport(400, 600, (100, 150), idx=0, t=0.0, motion="stable_pan")[0]
        even_end = ken_burns_viewport(400, 600, (100, 150), idx=0, t=1.0, motion="stable_pan")[0]
        odd_start = ken_burns_viewport(400, 600, (100, 150), idx=1, t=0.0, motion="stable_pan")[0]
        odd_end = ken_burns_viewport(400, 600, (100, 150), idx=1, t=1.0, motion="stable_pan")[0]

        self.assertLess(even_start, even_end)
        self.assertGreater(odd_start, odd_end)

    def test_clean_pro_motion_selection_avoids_side_pan(self):
        motions = {choose_motion_preset("clean_pro", "evidence", i) for i in range(12)}
        self.assertNotIn("stable_pan", motions)
        self.assertLessEqual(motions, {"slow_push", "hold_still"})


if __name__ == "__main__":
    unittest.main()
