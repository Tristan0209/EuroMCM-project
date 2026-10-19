import tempfile
import unittest
from pathlib import Path

import pandas as pd

from common import HERE, save_json, frozen_route, training_lengths, read_json
from carbon_loader import load_carbon, carbon_for_decision, assert_compatible_carbon
from align_time import alignment_delta, shift_times
from feasibility_precheck import precheck, isolated_earliest_start
from resource_contracts import service_at_pool, facility_energy_components, incremental_energy_kwh
from data_extensions import build_extensions


class CarbonContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_test_only_',dir=HERE)
        self.root = Path(self.temp.name)
        self.path = self.root/'synthetic_test_only.csv'
        self.meta = self.path.with_suffix('.metadata.json')
        self.frame = pd.DataFrame([{'datetime_utc':t,'region':r,'carbon_intensity_gco2_kwh':100+i*10}
            for i,t in enumerate(['2024-05-10T00:00:00Z','2024-05-10T01:00:00Z']) for r in ['DE','FR','ES']])
        self.metadata = {'data_kind':'synthetic_test_only','source':'synthetic_test_only',
            'unit':'gCO2/kWh','emissions_species':'CO2','boundary':'synthetic_test_only','timezone':'UTC','resolution':'hourly',
            'methodology':{'accounting_basis':'synthetic_test_only','scope':'synthetic_test_only','description':'synthetic_test_only interface fixture'}}

    def tearDown(self):
        self.temp.cleanup()

    def load(self,frame=None,metadata=None,allow=True,**kwargs):
        (self.frame if frame is None else frame).to_csv(self.path,index=False)
        save_json(self.meta,self.metadata if metadata is None else metadata)
        return load_carbon(self.path,self.meta,allow_synthetic=allow,**kwargs)

    def test_valid_interface_and_actual_rejected_for_decision(self):
        data = self.load(required_start='2024-05-10T00:00:00Z',required_end='2024-05-10T01:59:59Z')
        self.assertEqual(len(data),6)
        with self.assertRaises(ValueError):
            carbon_for_decision(data,'2024-05-10T00:00:00Z',['2024-05-10T01:00:00Z'])

    def test_no_synthetic_in_real_experiments(self):
        with self.assertRaises(ValueError):
            self.load(allow=False)

    def test_missing_file_clear(self):
        with self.assertRaisesRegex(FileNotFoundError,'no annual factor or synthetic fallback'):
            load_carbon(self.root/'missing.csv')

    def test_malformed_records_rejected(self):
        duplicate = pd.concat([self.frame,self.frame.iloc[[0]]],ignore_index=True)
        missing_hour = self.frame.copy()
        missing_hour.loc[3:,'datetime_utc']='2024-05-10T02:00:00Z'
        null = self.frame.copy(); null.loc[0,'carbon_intensity_gco2_kwh']=None
        negative = self.frame.copy(); negative.loc[0,'carbon_intensity_gco2_kwh']=-1
        naive = self.frame.copy(); naive['datetime_utc']=naive.datetime_utc.str.replace('Z','',regex=False)
        fractional = self.frame.copy(); fractional.loc[0,'datetime_utc']='2024-05-10T00:10:00Z'
        for frame in [duplicate,self.frame.iloc[:-1],missing_hour,null,negative,naive,fractional]:
            with self.subTest(case=frame.to_json()):
                with self.assertRaises(ValueError):
                    self.load(frame)

    def test_bad_units_and_coverage_rejected(self):
        for metadata in [{**self.metadata,'unit':'kgCO2/MWh'},{**self.metadata,'boundary':''}]:
            with self.assertRaises(ValueError):
                self.load(metadata=metadata)
        with self.assertRaises(ValueError):
            self.load(required_end='2024-05-10T03:00:00Z')

    def test_only_past_issued_forecasts_available(self):
        metadata = {**self.metadata,'data_kind':'forecast'}
        early = self.frame.copy(); early['issued_at_utc']='2024-05-09T22:00:00Z'
        later = self.frame.copy(); later['issued_at_utc']='2024-05-09T23:30:00Z'
        later['carbon_intensity_gco2_kwh']=999
        data = self.load(pd.concat([early,later],ignore_index=True),metadata)
        values = carbon_for_decision(data,'2024-05-09T23:00:00Z',['2024-05-10T01:00:00Z'])
        self.assertEqual(set(values.carbon_intensity_gco2_kwh),{110})
        with self.assertRaises(ValueError):
            carbon_for_decision(data,'2024-05-09T21:00:00Z',['2024-05-10T01:00:00Z'])

    def test_CO2e_requires_matching_column_unit_and_GWP_method(self):
        frame=self.frame.rename(columns={'carbon_intensity_gco2_kwh':'carbon_intensity_gco2e_kwh'})
        meta={**self.metadata,'emissions_species':'CO2e','unit':'gCO2e/kWh'}
        with self.assertRaisesRegex(ValueError,'gwp'):
            self.load(frame,meta)
        meta['methodology']={**meta['methodology'],'gwp_horizon_years':100,'gwp_reference':'synthetic_test_only'}
        data=self.load(frame,meta)
        self.assertEqual(data.attrs['value_column'],'carbon_intensity_gco2e_kwh')
        with self.assertRaises(ValueError):
            self.load(self.frame,meta)
        actual=self.load()
        with self.assertRaises(ValueError):
            assert_compatible_carbon(actual,data)

    def test_methodology_required_for_CO2_too(self):
        meta=dict(self.metadata);meta.pop('methodology')
        with self.assertRaises(ValueError):
            self.load(metadata=meta)

    def test_alignment_translates_only_times_and_checks_coverage(self):
        data=self.load()
        data.attrs['metadata']={**data.attrs['metadata'],'data_kind':'actual'}
        data['datetime_utc']+=pd.Timedelta(days=730)
        requests=pd.DataFrame({'request_id':['same'],'selected_model':['L'],
            'arrival_time':['2024-05-10T00:00:00Z'],'deadline':['2024-05-10T00:00:30Z']})
        delta=alignment_delta(requests,data)
        shifted=shift_times(requests,delta,['arrival_time','deadline'])
        self.assertEqual(delta,pd.Timedelta(days=730))
        self.assertEqual(shifted.selected_model.iloc[0],'L')
        self.assertEqual((pd.Timestamp(shifted.deadline.iloc[0])-pd.Timestamp(shifted.arrival_time.iloc[0])).total_seconds(),30)
        with self.assertRaises(ValueError):
            alignment_delta(requests,data,'2026-05-12T00:00:00Z')
        data.attrs['metadata']['data_kind']='synthetic_test_only'
        with self.assertRaises(ValueError):
            alignment_delta(requests,data)


class ResourceContracts(unittest.TestCase):
    def test_busy_idle_decomposition_matches_two_state_total(self):
        value=facility_energy_components(100,2,30,175,250,1.2)
        self.assertAlmostEqual(value['facility_kwh'],(250*30+175*(200-30))*1.2/3.6e6)
        self.assertAlmostEqual(value['busy_increment_kwh'],incremental_energy_kwh(30,175,250,1.2))
        with self.assertRaises(ValueError):
            facility_energy_components(10,1,11,175,250,1.2)

    def test_L_multiplier_is_not_applied_twice(self):
        self.assertEqual(service_at_pool(12,3,3),12)
        self.assertEqual(service_at_pool(12,3,1),4)
        self.assertEqual(service_at_pool(12,3,6),24)
        with self.assertRaises(ValueError):
            service_at_pool(12,0,3)


class AlignmentPipeline(unittest.TestCase):
    def test_full_translation_preserves_baseline_and_all_scenario_offsets(self):
        import shutil
        from unittest.mock import patch
        import align_time
        with tempfile.TemporaryDirectory(prefix='synthetic_test_only_',dir=HERE) as temp:
            root=Path(temp);base=root/'baseline';base.mkdir()
            requests=pd.read_csv(HERE/'output/requests.csv').iloc[:2]
            requests.to_csv(base/'requests.csv',index=False)
            for name in ['server_config.csv','network_config.csv','server_availability.csv','training_length_statistics.csv',
                         'service_random_conditions.csv','source_integrity.json','project_inventory.json',
                         'interface_test_report.json','resource_parameters.json','data_manifest.json']:
                shutil.copyfile(HERE/'output'/name,base/name)
            config=read_json(HERE/'config.json')
            save_json(base/'effective_config.json',config)
            frame=pd.DataFrame([{'datetime_utc':t,'region':r,'carbon_intensity_gco2_kwh':100}
                for t in pd.date_range('2026-05-10T00:00:00Z',periods=7,freq='h') for r in ['DE','FR','ES']])
            path=root/'synthetic_test_only.csv';frame.to_csv(path,index=False)
            meta={'data_kind':'synthetic_test_only','source':'synthetic_test_only; loader is mocked in this test'}
            save_json(path.with_suffix('.metadata.json'),meta)
            frame.attrs['metadata']={**meta,'data_kind':'actual','unit':'gCO2/kWh','emissions_species':'CO2'}
            before=(base/'requests.csv').read_bytes()


            with patch.object(align_time,'OUT',base),patch.object(align_time,'load_carbon',return_value=frame):
                target=align_time.align_dataset(path,destination=root/'aligned')
            self.assertEqual(before,(base/'requests.csv').read_bytes())
            aligned=pd.read_csv(target/'requests.csv')
            self.assertTrue(aligned.arrival_time.str.startswith('2026-05-10').all())
            pd.testing.assert_frame_equal(requests.drop(columns=['arrival_time','deadline']).reset_index(drop=True),
                aligned.drop(columns=['arrival_time','deadline']))
            events=pd.read_csv(target/'server_availability_local_failure.csv')
            self.assertEqual(pd.Timestamp(events.datetime_utc.iloc[-2]),pd.Timestamp('2026-05-10T01:00:00Z'))
            self.assertEqual(read_json(target/'effective_config.json')['time_alignment']['mode'],'carbon_coverage')


class FeasibilityContracts(unittest.TestCase):
    def setUp(self):
        self.cfg=read_json(HERE/'config.json')
        self.cfg['precheck']['router_seconds']=0
        self.cfg['precheck']['network_control_seconds']=0
        self.cfg['service_classes']=[{'name':'batch','max_queue_wait_seconds':20}]
        self.server=pd.DataFrame([{'region':'DE','model_type':a,'capacity':1,'service_time_multiplier':1 if a=='S' else 3} for a in ['S','L']])
        self.net=pd.DataFrame([{'origin_region':'DE','destination_region':'DE','allowed':1,'latency_ms':0}])
        self.avail=pd.DataFrame([{'datetime_utc':'2024-05-10T00:00:00Z','region':'DE','model_type':a,
            'available_capacity':1,'provenance':'synthetic_test_only'} for a in ['S','L']])
        self.requests=pd.DataFrame([{'request_id':'fixture','arrival_time':'2024-05-10T00:00:00Z',
            'deadline':'2024-05-10T00:00:30Z','service_class':'batch','selected_model':'L','origin_region':'DE',
            'estimated_service_time_seconds':10}])

    def run_check(self):
        return precheck(self.requests,self.server,self.net,self.avail,self.cfg,'synthetic_test_only')

    def test_available_on_arrival_does_not_imply_completion(self):
        self.requests['estimated_service_time_seconds']=31
        i,b,s=self.run_check()
        self.assertFalse(i.isolated_possible.iloc[0])
        self.assertFalse(s['necessary_checks_satisfied'])

    def test_temporary_zero_capacity_can_wait_for_recovery(self):
        self.avail.loc[self.avail.model_type.eq('L'),'available_capacity']=0
        self.avail=pd.concat([self.avail,pd.DataFrame([{'datetime_utc':'2024-05-10T00:00:10Z','region':'DE',
            'model_type':'L','available_capacity':1,'provenance':'synthetic_test_only'}])],ignore_index=True)
        i,b,s=self.run_check()
        self.assertTrue(i.isolated_possible.iloc[0])
        self.assertAlmostEqual(i.minimum_required_queue_wait_seconds.iloc[0],10)
        self.cfg['service_classes'][0]['max_queue_wait_seconds']=5
        self.assertFalse(self.run_check()[0].isolated_possible.iloc[0])

    def test_capacity_conflict_detected_even_if_each_request_can_finish(self):
        self.requests['deadline']='2024-05-10T00:00:10Z'
        self.requests['estimated_service_time_seconds']=6
        other=self.requests.copy();other['request_id']='fixture_2'
        self.requests=pd.concat([self.requests,other],ignore_index=True)
        i,b,s=self.run_check()
        self.assertTrue(i.isolated_possible.all())
        self.assertGreater(s['violated_capacity_bounds'],0)
        self.assertFalse(s['necessary_checks_satisfied'])

    def test_network_overhead_can_make_deadline_impossible(self):
        self.requests['deadline']='2024-05-10T00:00:10Z'
        self.net['latency_ms']=1000
        self.assertFalse(self.run_check()[0].isolated_possible.iloc[0])

    def test_nonpreemptive_capacity_interval_must_fit_service(self):
        self.assertIsNone(isolated_earliest_start([(0,4,1),(4,6,0),(6,10,1)],0,9,5))
        self.assertEqual(isolated_earliest_start([(0,4,2),(4,10,1)],0,9,5),0)

    def test_invalid_availability_capacity_rejected(self):
        self.avail.loc[0,'available_capacity']=2
        with self.assertRaises(ValueError):
            self.run_check()

    def test_scenarios_are_independent_and_normal_unchanged(self):
        import copy
        config=read_json(HERE/'config.json')
        with tempfile.TemporaryDirectory(prefix='synthetic_test_only_',dir=HERE) as temp:
            root=Path(temp)
            base=pd.DataFrame([{'datetime_utc':config['start_datetime_utc'],'region':r,'model_type':a,
                'available_capacity':1,'provenance':'synthetic_test_only'} for r in ['DE','FR','ES'] for a in ['S','L']])
            servers=pd.DataFrame([{'region':r,'model_type':a,'capacity':1,'service_time_multiplier':1 if a=='S' else 3}
                for r in ['DE','FR','ES'] for a in ['S','L']])
            base.to_csv(root/'server_availability.csv',index=False)
            before=(root/'server_availability.csv').read_bytes()
            build_extensions(root,self.requests,servers,self.net,config)
            self.assertEqual(before,(root/'server_availability.csv').read_bytes())
            self.assertEqual(before,(root/'server_availability_normal.csv').read_bytes())
            self.assertEqual(len(pd.read_csv(root/'server_availability_local_failure.csv')),8)
            self.assertEqual(len(pd.read_csv(root/'server_availability_high_load_capacity_reduction.csv')),10)


class InformationBoundary(unittest.TestCase):
    def test_unknown_and_risk_tasks_always_L(self):
        cost={'coefficients':{'s':[0,0],'l':[0,0]},'normalization_c0':1}
        params={'lambda':0.5,'tau':-0.617099253690552}
        for task,benchmark in [('Code','mbpp'),('Math','math'),('General','unregistered'),('Unknown','hellaswag')]:
            self.assertEqual(frozen_route(task,benchmark,1,0,100,{'hellaswag','mbpp','math'},cost,params,['Code','Math']),'L')

    def test_nontrain_output_cannot_affect_length_estimate(self):
        import common
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix='synthetic_test_only_',dir=HERE) as tmp:
            path = Path(tmp)/'outcomes.jsonl'
            requests = pd.DataFrame({'request_id':['historical','current'],'split':['train','test'],'task_type':['General','General']})
            records = [dict(request_id='historical',action_id=a,output_tokens_proxy=10) for a in ['S','L']]
            import json
            def write(current_length):
                rows=records+[dict(request_id='current',action_id=a,output_tokens_proxy=current_length,quality_true=0,response='FORBIDDEN') for a in ['S','L']]
                path.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
            with patch.object(common,'paths',return_value={'paired_outcomes':path}):
                write(1); first=training_lengths(requests)
                write(999999999); second=training_lengths(requests)
            pd.testing.assert_frame_equal(first,second)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    save_json(HERE/'output'/'interface_test_report.json',{'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
        'scope':'Synthetic temporary fixtures only; no real carbon data or physical prediction accuracy validated'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
