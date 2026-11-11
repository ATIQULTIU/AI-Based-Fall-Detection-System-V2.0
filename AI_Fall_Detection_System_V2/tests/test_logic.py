import unittest
from fall_detection.logic import TemporalFallTracker, is_fall_candidate

class FallLogicTests(unittest.TestCase):
    def test_angle_candidate(self): self.assertTrue(is_fall_candidate(70,.6))
    def test_horizontal_candidate(self): self.assertTrue(is_fall_candidate(10,1.4))
    def test_normal_pose(self): self.assertFalse(is_fall_candidate(15,.45))
    def test_requires_confirmation_duration(self):
        t=TemporalFallTracker(1.0,smoothing=1.0)
        self.assertFalse(t.update(7,75,.8,now=10.0)[0])
        fallen,new,*_=t.update(7,75,.8,now=11.1)
        self.assertTrue(fallen); self.assertTrue(new)
    def test_event_only_once_until_recovery(self):
        t=TemporalFallTracker(.5,smoothing=1.0)
        t.update(2,80,1.4,now=1.0)
        self.assertTrue(t.update(2,80,1.4,now=1.6)[1])
        self.assertFalse(t.update(2,80,1.4,now=2.0)[1])
        t.update(2,10,.5,now=2.1); t.update(2,80,1.4,now=3.0)
        self.assertTrue(t.update(2,80,1.4,now=3.6)[1])

if __name__=='__main__': unittest.main()
