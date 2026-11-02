"""Regression checks for corrected timing, state selection and FDR handling."""
import importlib
import unittest
import numpy as np
from common_still import BIN, states
bh = importlib.import_module('22_robustness').bh
neurons = importlib.import_module('10_neurons')


class RobustnessTests(unittest.TestCase):
    def test_region_boot_ignores_neuron_duplication_within_mouse(self):
        import pandas as pd
        from robust_stats import mouse_region_boot
        D = pd.DataFrame(dict(subject=['a','a','b','b'], region=['core','other']*2,
                              value=[2., 0., 4., 2.]))
        duplicated = pd.concat([D, D.iloc[[0]*100]], ignore_index=True)
        self.assertEqual(mouse_region_boot(D, ['core','other'], 'value', ['core'], ['other']), [2., 2.])
        self.assertEqual(mouse_region_boot(duplicated, ['core','other'], 'value', ['core'], ['other']), [2., 2.])

    def test_bh_monotonic_and_missing(self):
        p = np.array([.04, .001, np.nan, .03, .2])
        np.testing.assert_allclose(bh(p), [.0533333333, .004, np.nan, .0533333333, .2], equal_nan=True)
        self.assertTrue(np.isnan(bh([np.nan])[0]))

    def test_gate_crossing_is_not_movement_offset(self):
        nb = 500
        grey = np.ones(nb, bool)
        # Actual movement ends at bin 49, but pupil gate starts a bout at 100.
        tsm = np.maximum(0, np.arange(nb)-49)
        S = np.zeros(nb, bool)
        S[100:200] = True
        st = dict(nb=nb, grey=grey, tsm=tsm, mov=np.arange(nb)<50)
        starts, off, on = neurons.movement_events(st, S)
        self.assertEqual(len(starts), 0)
        self.assertEqual(len(off), 0)
        self.assertEqual(len(on), 0)

    def test_real_offset_and_onset(self):
        nb = 500
        mov = (np.arange(nb)<99) | (np.arange(nb)>=220)
        from common_still import run_lengths
        S = np.zeros(nb, bool)
        S[100:219] = True
        st = dict(nb=nb, grey=np.ones(nb, bool), tsm=run_lengths(mov), mov=mov)
        starts, off, on = neurons.movement_events(st, S)
        np.testing.assert_array_equal(starts, [100])
        np.testing.assert_array_equal(off, [99])
        np.testing.assert_array_equal(on, [220])

    def test_pupil_and_movement_margin(self):
        n = 1000
        s = dict(wheel_speed=np.zeros(n), licks=np.zeros(n), motion_energy=np.zeros(n),
                 pupil=np.full(n, 10.), spont=np.array([20., 50.]),
                 intervals_0=np.array([0., 10.]), intervals_1=np.array([5., 15.]), mask=np.ones(2, bool))
        s['wheel_speed'][450] = .1
        s['pupil'][470:480] = 1
        st = states(s)
        self.assertFalse(st['awake'][449:452].any())
        self.assertTrue(st['awake'][452])
        self.assertFalse(st['awake'][470:480].any())
        self.assertTrue(st['pupil_ok'])


if __name__ == '__main__':
    unittest.main()
