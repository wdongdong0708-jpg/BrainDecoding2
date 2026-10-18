"""CPU checks before freezing/reading empirical test predictions."""
import copy
import numpy as np
import pandas as pd
import pytest

from braindecoding import unified_heldout_test_v2 as worker


def test_shared_material_blocks_never_split_subjects():
    rows = []
    for subject in ('s1','s2'):
        for number in range(7):
            rows.append(dict(protocol_group=f'{subject}-{number}',subject_id=subject,recording_id=f'{subject}-run-06',
                运行编号='run-06',BIDS事件行号=number,event_id=f'{subject}-{number}',normalized_word='w'))
    groups,blocks,materials = worker.block_groups('pallier2025',pd.DataFrame(rows),5)
    assert len(blocks)==2 and len(materials)==7 and len(groups)==14
    mapping = {}
    for group in groups:
        mapping.setdefault(group['material'],set()).add(group['block'])
        assert group['partition']=='test'
    assert all(len(values)==1 for values in mapping.values())


def test_reference_words_are_not_passed_to_decoder():
    group = dict(event_ids=['a','b'],reference=['secret','reference'])
    predictions = {h:np.eye(2) for h in range(3)}
    cases = worker.cases_for_groups([group],predictions,{'a':0,'b':1},weight=.1,branch='open')
    assert set(cases[0])=={'predictions','weight','open_vocabulary'}
    assert cases[0]['weight']==.1 and cases[0]['open_vocabulary']


def test_saved_chunk_rejects_wrong_split_or_seed_or_manifest():
    groups = [dict(group='g',reference=['word'],subject='s')]
    result = dict(final={'A':['word'],'C':['word']},trace=[dict(arrival=0,A=['word'],C=['word'])])
    data = dict(manifest_sha256='frozen',dataset='d',seed=0,conditions=['A','C'],split='test',
                test_evaluated=True,selected_weight=.1,records=[dict(group='g',result=result)])
    worker.validate_chunk(data,{'manifest_sha256':'frozen'},'d',0,groups)
    for key,value in [('split','val'),('seed',1),('manifest_sha256','stale'),('selected_weight',1.5)]:
        altered = copy.deepcopy(data); altered[key]=value
        with pytest.raises(ValueError):
            worker.validate_chunk(altered,{'manifest_sha256':'frozen'},'d',0,groups)
