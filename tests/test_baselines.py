import itertools
import random
import unittest
from examples.summary_baselines import decode, knapsack, temporal_f1, uniform_keyframes


class BaselineTests(unittest.TestCase):
    def test_knapsack_matches_exhaustive_subsets(self):
        rng = random.Random(8)
        for _ in range(50):
            lengths = [rng.randint(1,8) for _ in range(7)]
            values = [rng.randint(-2,15) for _ in range(7)]
            capacity = rng.randint(0,20)
            chosen = knapsack(lengths,values,capacity)
            feasible = [sum(v*b for v,b in zip(values,bits)) for bits in itertools.product((0,1),repeat=7) if sum(n*b for n,b in zip(lengths,bits)) <= capacity]
            self.assertEqual(sum(values[i] for i in chosen),max(feasible))
            self.assertLessEqual(sum(lengths[i] for i in chosen),capacity)
            self.assertEqual(len(chosen),len(set(chosen)))

    def test_budget_and_tie_break(self):
        self.assertEqual(knapsack([6,5,5],[9,8,8],10),[1,2])
        self.assertEqual(knapsack([2,2],[1,1],2),[0])
        mask, chosen = decode([1]*20,[(0,2),(2,5),(5,20)])
        self.assertEqual(sum(mask),2)
        self.assertEqual(chosen,[0])

    def test_pooling_changes_objective(self):
        scores = [1]*10
        segments = [(0,2),(2,5),(5,10)]
        self.assertEqual(decode(scores,segments,.5,'mean')[1],[0,1])
        # Equal integrated utility keeps the earlier set; both use five frames.
        self.assertEqual(sum(decode(scores,segments,.5,'sum')[0]),5)
        scores = [1,1]+[.9]*4+[0]*6
        segments = [(0,2),(2,6),(6,12)]
        self.assertEqual(decode(scores,segments,1/3,'mean')[1],[0])
        self.assertEqual(decode(scores,segments,1/3,'sum')[1],[1])

    def test_reference_aggregation_and_empty(self):
        refs = [[1,0],[0,1]]
        self.assertEqual(temporal_f1([1,0],refs),.5)
        self.assertEqual(temporal_f1([1,0],refs,'max'),1)
        self.assertEqual(temporal_f1([0,0],[[0,0]]),0)

    def test_invalid_timelines_and_values(self):
        for segments in [[(1,3)],[(0,2),(1,3)],[(0,2)]]:
            with self.assertRaises(ValueError): decode([1,2,3],segments)
        with self.assertRaises(ValueError): decode([float('nan')],[(0,1)])
        with self.assertRaises(ValueError): temporal_f1([1],[[1,0]])
        with self.assertRaises(ValueError): knapsack([0],[1],3)

    def test_uniform_storyboard(self):
        self.assertEqual(uniform_keyframes(10,3),[1,5,8])
        self.assertEqual(uniform_keyframes(0,0),[])
        self.assertEqual(uniform_keyframes(3,3),[0,1,2])


if __name__ == '__main__': unittest.main()
