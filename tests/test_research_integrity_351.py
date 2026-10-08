import pandas as pd
import pytest
from quanta.validation.advanced_metrics import max_drawdown
from quanta.validation.integrity import validate_research_inputs
from quanta.validation.evidence import evidence_verdict

def test_drawdown_includes_initial_capital():
    assert max_drawdown(pd.Series([-.1,.02])) == pytest.approx(-.1)

def test_pit_not_verified_by_file_existence(tmp_path):
    a=tmp_path/'dummy.csv'; a.write_text('date,ticker\n2024-01-01,X')
    dates=pd.date_range('2024-01-01',periods=100)
    prices=pd.DataFrame({'B':range(100)},index=dates)
    r=pd.DataFrame({'B':[.001]*100},index=dates)
    result=validate_research_inputs(prices,r.iloc[:50],r.iloc[50:],'B',{},False,str(a),str(a))
    assert result.status=='PASS_WITH_LIMITATIONS'
    assert not result.checks['pit_constituents']
    assert validate_research_inputs(prices,r.iloc[:50],r.iloc[50:],'B',{},True,str(a),str(a)).status=='REJECT'

def test_evidence_rejects_unverified_data():
    v=evidence_verdict({'cagr':.4}, {}, 'PASS_WITH_LIMITATIONS')
    assert v['signal']=='INSUFFICIENT_EVIDENCE'
