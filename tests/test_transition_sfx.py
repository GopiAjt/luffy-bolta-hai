import unittest

from app.config import TRANSITION_SFX_DIR
from app.utils.audio import transition_sfx


class TransitionSfxTests(unittest.TestCase):
    def test_default_sfx_dir_points_to_bundled_assets(self):
        self.assertEqual(transition_sfx.SFX_DIR, TRANSITION_SFX_DIR)
        self.assertTrue((transition_sfx.SFX_DIR / "soft_whoosh.mp3").exists())

    def test_default_transition_resolves_real_sfx_file(self):
        resolved = transition_sfx.resolve_transition_sfx("fade")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, "soft_whoosh.mp3")

    def test_default_sfx_gain_is_audible(self):
        resolved = transition_sfx.resolve_transition_sfx("fade")
        self.assertIsNotNone(resolved)
        self.assertGreaterEqual(transition_sfx.sfx_volume_for_path(resolved), 1.5)
        self.assertGreaterEqual(transition_sfx.TRANSITION_SFX_MIX_WEIGHT, 2.0)

    def test_repeated_gentle_transitions_are_spaced_and_varied(self):
        events = [
            {"time": 3.0, "transition": "crossfade"},
            {"time": 6.0, "transition": "fade_eased"},
            {"time": 10.0, "transition": "crossfade"},
            {"time": 14.0, "transition": "fade_eased"},
            {"time": 18.0, "transition": "crossfade"},
            {"time": 23.0, "transition": "fade_eased"},
        ]

        cues = transition_sfx.plan_transition_sfx_cues(events)
        stems = [cue["stem"] for cue in cues]

        self.assertLess(len(cues), len(events))
        self.assertGreater(len(set(stems)), 1)
        self.assertFalse(any(a == b for a, b in zip(stems, stems[1:])))

    def test_impact_hits_have_cooldown(self):
        events = [
            {"time": 10.0, "transition": "zoom_dissolve"},
            {"time": 18.0, "transition": "zoom_dissolve"},
            {"time": 30.0, "transition": "zoom_dissolve"},
        ]

        cues = transition_sfx.plan_transition_sfx_cues(events)
        impact_times = [cue["time"] for cue in cues if cue["stem"] == "impact_hit"]

        self.assertEqual(impact_times, [10.0, 30.0])


if __name__ == "__main__":
    unittest.main()
