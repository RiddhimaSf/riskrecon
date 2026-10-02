"""The SQL layer must agree with the Python implementation (same gates, same queue)."""
import shutil
from pathlib import Path

import duckdb
import pytest

from src.pipeline import run

SQL = Path(__file__).resolve().parents[1] / "sql"


def _statements(sql_text):
    """Drop `--` comment lines, then split on ';' (comments may themselves contain semicolons)."""
    code = "\n".join(l for l in sql_text.splitlines() if not l.strip().startswith("--"))
    return [s.strip() for s in code.split(";") if s.strip()]


def _exec_all(con, sql_text):
    last = None
    for stmt in _statements(sql_text):
        res = con.execute(stmt)
        if res.description:
            last = res.df()
    return last


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    r = run(n=6000, seed=9, write=False)
    (tmp_path / "data").mkdir()
    r["members"].to_csv(tmp_path / "data" / "synthetic_members.csv", index=False)
    from src.pipeline import RESULT_COLUMNS
    r["classified"][RESULT_COLUMNS].to_csv(tmp_path / "data" / "reconciliation_results.csv", index=False)
    monkeypatch.chdir(tmp_path)
    return r


def test_sql_validation_gates_match_python(workdir):
    con = duckdb.connect()
    df = _exec_all(con, (SQL / "01_validation.sql").read_text()).set_index("gate")
    py = workdir["gates"].set_index("gate")
    for gate in df.index:
        assert df.loc[gate, "status"] == py.loc[gate, "status"], gate


def test_sql_headline_matches_python(workdir):
    con = duckdb.connect()
    stmts = _statements((SQL / "02_reconciliation.sql").read_text())
    for s in stmts[:2]:
        con.execute(s)
    head = con.execute(stmts[2]).df().iloc[0]
    py = workdir["metrics"]["headline"]
    assert head["pct_beyond_tolerance"] == pytest.approx(py["pct_beyond_tolerance"])
    assert head["mae"] == pytest.approx(py["mae"], rel=1e-6)
    assert head["p95_abs_delta"] == pytest.approx(py["p95_abs_delta"], rel=1e-6)


def test_sql_queue_matches_python_queue(workdir):
    con = duckdb.connect()
    q = _exec_all(con, (SQL / "03_investigation_queue.sql").read_text())
    py = workdir["queue"]
    assert len(q) == len(py) == 50
    assert q["member_id"].tolist() == py["member_id"].tolist()
    assert q["priority"].tolist() == pytest.approx(py["priority"].tolist(), abs=1e-6)
