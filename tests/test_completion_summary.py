from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, main

from whistlemamba_studio.web import completion_summary


class CompletionSummaryTest(TestCase):
    def test_modes(self):
        cases = [
            ("learned",3,7,"3 learned candidate tracks"),
            ("heuristic",3,7,"7 heuristic candidate tracks"),
            ("both",3,7,"3 learned + 7 heuristic = 10 candidate tracks"),
            ("learned",0,0,"0 learned candidate tracks"),
            ("heuristic",0,0,"0 heuristic candidate tracks"),
            ("both",0,0,"0 learned + 0 heuristic = 0 candidate tracks"),
            ("learned",1,0,"1 learned candidate track"),
            ("heuristic",0,1,"1 heuristic candidate track"),
            ("both",0,1,"0 learned + 1 heuristic = 1 candidate track"),
            ("both",4,0,"4 learned + 0 heuristic = 4 candidate tracks"),
        ]
        for mode,learned,heuristic,want in cases:
            with self.subTest(mode=mode, learned=learned, heuristic=heuristic):
                result=SimpleNamespace(learned_contours=learned,heuristic_contours=heuristic,project={"tracks":[{"id":"fixed"}]})
                before=deepcopy(result)
                self.assertEqual(completion_summary(mode,result),want)
                self.assertEqual(result,before)

    def test_unknown_not_reported_as_combined(self):
        with self.assertRaises(ValueError):
            completion_summary("unknown",None)

if __name__=="__main__": main()
