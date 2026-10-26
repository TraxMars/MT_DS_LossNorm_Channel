"""Run old unittest suites as imports, writing only new C0 evidence."""
import importlib.util
import io
import json
import subprocess
import time
import unittest
from support import ROOT,PROJECT,write_json


def main():
    records=[]
    for filename,cls in [('test_stage_features.py','StageFeaturesTests'),('test_stage1a_diagnostics.py','DiagnosticTests'),('test_stage1b_boundary.py','BoundaryTests')]:
        spec=importlib.util.spec_from_file_location('c0_regression_'+filename[:-3],PROJECT/'tests'/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        stream=io.StringIO();start=time.monotonic()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module,cls)))
        record=dict(suite=filename,status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
                    elapsed_seconds=time.monotonic()-start,log=stream.getvalue())
        records.append(record);print(filename,record['status'],result.testsRun,flush=True)
    core=subprocess.run(['/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python','-B',str(PROJECT/'tests/verify_core.py')],cwd=PROJECT,capture_output=True,text=True)
    records.append(dict(suite='verify_core.py',status='PASS' if core.returncode==0 else 'FAIL',exit_code=core.returncode,stdout=core.stdout,stderr=core.stderr))
    report=dict(status='PASS' if all(r['status']=='PASS' for r in records) else 'FAIL',unittest_total=sum(r.get('tests_run',0) for r in records),suites=records,historical_reports_not_overwritten=True)
    write_json(ROOT/'REGRESSION_RESULTS.json',report)
    assert report['status']=='PASS'

if __name__=='__main__':main()
