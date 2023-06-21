import unittest

import pagerank as pr

class TestPageRank(unittest.TestCase):

    def test_transition_mode(self):
        corpus = {
            "1.html": {"2.html", "3.html"},
            "2.html": {"3.html"},
            "3.html": {"2.html"}
        }
        
        damping_factor = 0.85
        expected_1 = {
            "1.html": 0.05,
            "2.html": 0.475,
            "3.html": 0.475
        }
        self.assertEqual(pr.transition_model(corpus, "1.html", damping_factor), expected_1)

        expected_2 = {
            "1.html": 0.05,
            "2.html": 0.05,
            "3.html": 0.9
        }
        self.assertEqual(pr.transition_model(corpus, "2.html", damping_factor), expected_2)

    def test_sample_pagerank(self):
        corpus = {
            "1.html": {"2.html", "3.html"},
            "2.html": {"3.html"},
            "3.html": {"2.html"}
        }
        damping_factor = 0.85
        n = 10000
        pageranks = pr.sample_pagerank(corpus, damping_factor, n)
        self.assertEqual(len(pageranks), 3)
        self.assertAlmostEqual(pageranks["1.html"], 0.05, delta=0.01)
        self.assertAlmostEqual(pageranks["2.html"], 0.475, delta=0.01)
        self.assertAlmostEqual(pageranks["3.html"], 0.475, delta=0.01)

    def test_iterate_pagerank(self):
        corpus = {
            "1.html": {"2.html", "3.html"},
            "2.html": {"3.html"},
            "3.html": {"2.html"}
        }
        damping_factor = 0.85
        pageranks = pr.iterate_pagerank(corpus, damping_factor)
        self.assertEqual(len(pageranks), 3)
        self.assertAlmostEqual(pageranks["1.html"], 0.05, delta=0.01)
        self.assertAlmostEqual(pageranks["2.html"], 0.475, delta=0.01)
        self.assertAlmostEqual(pageranks["3.html"], 0.475, delta=0.01)

if __name__ == '__main__':
    unittest.main()