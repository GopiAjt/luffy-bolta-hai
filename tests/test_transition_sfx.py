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


if __name__ == "__main__":
    unittest.main()
