"""Branding stays escaped; native widgets own application presentation."""
import unittest
from ui_styles import CSS, header


class UIStyleTests(unittest.TestCase):
    def test_branding_escapes_dynamic_text(self):
        for html in (header('<script>x</script>'), header('<img onerror="x">')):
            self.assertNotIn('<script>', html)
            self.assertNotIn('<img', html)
            self.assertIn('&lt;', html)

    def test_one_scoped_stylesheet_leaves_widgets_to_native_theme(self):
        self.assertEqual(CSS.count('<style>'), 1)
        self.assertEqual(CSS.count('</style>'), 1)
        self.assertNotIn('data-testid', CSS)
        self.assertNotIn('!important', CSS)
        self.assertNotIn('.stButton', CSS)
        self.assertIn('.st-key-welcome-profile', CSS)
