from context import *
import importlib.util
import src
from v2 import des,milp
sys.modules['src.des']=des;sys.modules['src.milp']=milp;src.des=des;src.milp=milp
import src.util as util
spec=importlib.util.spec_from_file_location('test_system',BASE/'tests/test_system.py')
module=importlib.util.module_from_spec(spec);sys.modules['test_system']=module;spec.loader.exec_module(module)
original=module.test_input_fingerprint_invalidates_after_data_change
def redirected():
    previous=util.ROOT
    try:
        (ROOT/'test_io/results/tmp').mkdir(parents=True,exist_ok=True);util.ROOT=ROOT/'test_io';original()
    finally:util.ROOT=previous
module.test_input_fingerprint_invalidates_after_data_change=redirected
import pytest
class Plugin:
    def pytest_pycollect_makemodule(self,module_path,parent):
        if module_path.name=='test_system.py':
            class ExistingModule(pytest.Module):
                def _getobj(self):return module
            return ExistingModule.from_parent(parent,path=module_path)
if __name__=='__main__':
    sys.exit(pytest.main(['-q','-p','no:cacheprovider',str(BASE/'tests'),str(A2/'tests'),str(EXT/'test_spatial.py'),str(PREVIOUS/'tests'),str(ROOT/'tests'),
        '--junitxml='+str(ROOT/'audit/tests.xml')],plugins=[Plugin()]))
