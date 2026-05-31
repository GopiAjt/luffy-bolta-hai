import unittest

import numpy as np

from app.utils.slides.generate_slideshow import (
    choose_slide_motion,
    choose_transition,
    cube_rotation_effect,
    iris_wipe_effect,
    ken_burns_headroom_for_motion,
    ken_burns_viewport,
    motion_uses_ken_burns,
    motion_blur_slide_effect,
    radial_wipe_effect,
    safe_transition_name,
    should_use_subject_layout,
    transition_progress,
    whip_pan_effect,
)
from app.utils.video.visual_effects import choose_motion_preset


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

    def test_static_hold_disables_ken_burns_per_slide(self):
        self.assertFalse(motion_uses_ken_burns("static_hold"))
        self.assertTrue(motion_uses_ken_burns("slow_push"))

    def test_wide_images_use_safe_subject_layout_by_default(self):
        wide = np.zeros((300, 1200, 3), dtype=np.uint8)

        self.assertTrue(should_use_subject_layout(wide, (1080, 1920)))

    def test_harsh_transition_names_map_to_frame_safe_transitions(self):
        self.assertEqual(safe_transition_name("whip_pan_right"), "crossfade")
        self.assertEqual(safe_transition_name("motion_slide_left"), "crossfade")
        self.assertEqual(safe_transition_name("cube_rotation_right"), "iris_wipe")
        self.assertEqual(safe_transition_name("page_curl_tr"), "fade_eased")

    def test_auto_transition_pool_uses_frame_safe_choices(self):
        allowed = {"fade", "fade_eased", "crossfade", "zoom_dissolve", "iris_wipe", "radial_wipe"}
        for idx in range(24):
            transition = choose_transition(idx, 4.5, None)
            self.assertIn(transition, allowed)

    def test_slide_motion_can_choose_static_hold_for_cta(self):
        original_uniform = __import__("random").uniform
        try:
            __import__("random").uniform = lambda _a, _b: 0.0
            self.assertEqual(
                choose_slide_motion(
                    visual_style="clean_pro",
                    beat="cta",
                    index=4,
                    duration=2.5,
                    previous_motion="slow_push",
                ),
                "static_hold",
            )
        finally:
            __import__("random").uniform = original_uniform


if __name__ == "__main__":
    unittest.main()
