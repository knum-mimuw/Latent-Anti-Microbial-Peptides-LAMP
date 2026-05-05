from __future__ import annotations

from amp_opt.core.archive import OptimizationArchive
from amp_opt.core.types import Candidate


def test_archive_dedup_keeps_best_score() -> None:
    archive = OptimizationArchive()
    archive.add([
        Candidate(sequence="AAA", score=5.0, generation=0),
        Candidate(sequence="AAA", score=3.0, generation=1),
        Candidate(sequence="AAA", score=7.0, generation=2),
    ])
    assert archive.size == 1
    assert archive.best is not None
    assert archive.best.score == 3.0


def test_archive_top_k() -> None:
    archive = OptimizationArchive()
    archive.add([
        Candidate(sequence="AAA", score=5.0, generation=0),
        Candidate(sequence="BBB", score=2.0, generation=0),
        Candidate(sequence="CCC", score=8.0, generation=0),
        Candidate(sequence="DDD", score=1.0, generation=0),
    ])
    top2 = archive.top_k(2)
    assert len(top2) == 2
    assert top2[0].sequence == "DDD"
    assert top2[1].sequence == "BBB"


def test_archive_all_sequences() -> None:
    archive = OptimizationArchive()
    archive.add([
        Candidate(sequence="XX", score=1.0, generation=0),
        Candidate(sequence="YY", score=2.0, generation=0),
    ])
    seqs = archive.all_sequences()
    assert set(seqs) == {"XX", "YY"}
