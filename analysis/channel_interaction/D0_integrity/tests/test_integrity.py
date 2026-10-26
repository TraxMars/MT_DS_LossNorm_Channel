"""Unit tests of history protection without opening real data/checkpoints."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

path=Path(__file__).resolve().parents[1]/'code/audit.py'
spec=importlib.util.spec_from_file_location('d0_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class IntegrityTests(unittest.TestCase):
    def test_history_mutation_and_addition_are_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'a';p.write_text('original')
            with mock.patch.object(audit,'protected_paths',lambda:iter([p])):
                before=audit.snapshot();self.assertEqual(audit.verify(before)['status'],'PASS')
                p.write_text('mutation');self.assertEqual(audit.verify(before)['status'],'FAIL')
                p.unlink();self.assertEqual(audit.verify(before)['status'],'FAIL')
            p.write_text('original');q=Path(temp)/'new';q.write_text('addition')
            with mock.patch.object(audit,'protected_paths',lambda:iter([p,q])):
                self.assertEqual(audit.verify(before)['new_historical_files'],[str(q)])

if __name__=='__main__': unittest.main(verbosity=2)
