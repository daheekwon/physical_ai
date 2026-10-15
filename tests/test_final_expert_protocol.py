"""Protect final-test immutability and independent dense preservation measurements."""
import json
import numpy as np
import pytest
from scripts import run_followup_final_expert as protocol
from scripts.analyze_followup_final_expert import physical,energy
from zj_wam.io import sha256


def test_frozen_input_change_rejects_evaluation(tmp_path,monkeypatch):
    source=tmp_path/'rule.json';source.write_text('{"threshold":1}')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'fingerprints':{str(source):sha256(source)}}));monkeypatch.setattr(protocol,'LOCK',lock)
    protocol.verify_lock()
    source.write_text('{"threshold":2}')
    with pytest.raises(AssertionError,match='Frozen input changed'):protocol.verify_lock()


def test_midchunk_excursion_fails_even_when_endpoints_match():
    reference=np.zeros((110,10));trace=reference.copy();trace[55,2]=3.
    cfg={'tolerances':{'position':2.,'rotation':np.deg2rad(2),'coverage_loss':.02}}
    result=physical(trace,reference,5,lambda pose:0.,cfg)
    assert np.array_equal(trace[[79,-1]],reference[[79,-1]])
    assert result['object_error']==1.5


def test_orientation_wrap_and_suffix_coverage_both_count():
    reference=np.zeros((110,10));reference[:,4]=np.pi-.01;trace=reference.copy();trace[30:,4]=-np.pi+.01;trace[-1,3]=.5
    cfg={'tolerances':{'position':2.,'rotation':np.deg2rad(2),'coverage_loss':.02}}
    result=physical(trace,reference,5,lambda pose:1.-.1*pose[3],cfg)
    assert result['rotation_max']==pytest.approx(.02)
    assert result['coverage_loss']==0.
    assert result['coverage_suffix_loss']==pytest.approx(.05)
    assert result['object_error']==pytest.approx(2.5)


def test_cubic_position_jerk_has_physical_time_scaling():
    t=np.arange(20)*.01;q=np.column_stack([t**3,np.zeros(len(t))])
    assert energy(q)==pytest.approx(36*(len(t)-3)*.01,rel=1e-10)
