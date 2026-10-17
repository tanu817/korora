"""Run with:  python -m unittest discover tests   (no extra installs needed)"""
import random
import unittest

from shell_game import config
from shell_game.board import BoardState, Move, all_instructions, trained_moves
from shell_game.runner import play_game
from shell_game.shuffle import choose_next_move, shortest_path_to_end
from shell_game.sim import SimEyes, SimHands, SimWorld


class TestMoves(unittest.TestCase):
    def test_slide(self):
        s = BoardState()
        s.apply(Move("slide", ("A1", "B1")))
        self.assertEqual(s.occupied, frozenset({"B1", "A2", "A3"}))
        self.assertEqual(s.ball_slot, "B1")  # ball was under the A1 cup

    def test_push_moves_both_cups(self):
        s = BoardState(cups=["B1", "A2", "A3"], ball_cup=1)   # ball under the A2 cup
        s.apply(Move("push", ("A3", "A2", "A1")))
        self.assertEqual(s.cups, ["B1", "A1", "A2"])
        self.assertEqual(s.ball_slot, "A1")                     # pushed cup carried the ball

    def test_illegal_move_raises(self):
        with self.assertRaises(ValueError):
            BoardState().apply(Move("slide", ("A1", "A2")))     # A2 is occupied

    def test_result_matches_apply(self):
        for m in trained_moves():
            s = BoardState(cups=["B1", "A2", "A3"])
            if m.is_legal(set(s.cups)):
                expected = m.result(s.occupied)
                s.apply(m)
                self.assertEqual(s.occupied, expected)


class TestShuffle(unittest.TestCase):
    def test_never_stuck_and_always_ends_home(self):
        rng = random.Random(0)
        for _ in range(2000):
            s = BoardState()
            n, target = 0, rng.randint(*config.MIN_MOVES_RANGE)
            while (m := choose_next_move(s.occupied, n, target, rng)) is not None:
                s.apply(m)
                n += 1
                self.assertEqual(len(set(s.cups)), 3)            # never two cups in one slot
                self.assertLessEqual(n, config.MAX_MOVES + 10)
            self.assertGreaterEqual(n, target)
            self.assertEqual(s.occupied, config.END_LAYOUT)

    def test_every_reachable_layout_can_get_home(self):
        seen, frontier = set(), [frozenset(config.START_LAYOUT)]
        while frontier:
            occ = frontier.pop()
            if occ in seen:
                continue
            seen.add(occ)
            self.assertIsNotNone(shortest_path_to_end(occ), sorted(occ))
            frontier += [m.result(occ) for m in trained_moves() if m.is_legal(set(occ))]

    def test_ball_really_gets_mixed(self):
        rng = random.Random(1)
        ends = {s: 0 for s in config.END_LAYOUT}
        for _ in range(3000):
            s = BoardState(ball_cup=1)
            n, target = 0, rng.randint(*config.MIN_MOVES_RANGE)
            while (m := choose_next_move(s.occupied, n, target, rng)) is not None:
                s.apply(m)
                n += 1
            ends[s.ball_slot] += 1
        for slot, count in ends.items():
            self.assertGreater(count / 3000, 0.15, f"ball almost never ends at {slot}: {ends}")


class TestFullGameInSim(unittest.TestCase):
    def _play(self, seed, fail=0.0, wrong=0.0):
        rng = random.Random(seed)
        ball = rng.choice(config.START_LAYOUT)
        world = SimWorld(ball)
        return play_game(SimHands(world, rng, fail, wrong), SimEyes(world), ball, rng, verbose=False)

    def test_perfect_arm_always_wins_first_try(self):
        for seed in range(300):
            r = self._play(seed)
            self.assertTrue(r.success)
            self.assertTrue(r.found_first_try)
            self.assertEqual(r.interventions, 0)

    def test_clumsy_arm_still_mostly_wins(self):
        wins = sum(self._play(seed, fail=0.15, wrong=0.05).success for seed in range(500))
        self.assertGreater(wins, 0.85 * 500)

    def test_instruction_list(self):
        ins = all_instructions()
        self.assertEqual(len(ins), len(set(ins)))
        self.assertIn(config.PICK_TEXT, ins)


if __name__ == "__main__":
    unittest.main()
