"""Dashboard workflow, state invalidation, exports and graceful input handling."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from edugaze.dashboard import service

REPO = Path(__file__).resolve().parents[1]
APP = REPO/'streamlit_app.py'
AOI_TEXT = 'aoi_id,name,x,y,width,height\nleft,Left,0,0,840,1050\nright,Right,840,0,840,1050\n'


class DashboardFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'fixations').mkdir();(self.root/'images').mkdir()
        trials = []
        for name in ['artificial_a.png','artificial_b.png']:
            cv2.imwrite(str(self.root/'images'/name),np.full((1050,1680,3),120,dtype=np.uint8))
            for subject in [1,2]:
                trials.append(dict(name=name,subject=subject,X=[50,850,900],Y=[20,200,300],
                                   T=[100,200,300],length=3,condition='freeview',split='train'))
        self.path = self.root/service.FIXATIONS
        self.path.write_text(json.dumps(trials))

    def app(self):
        with patch.dict(os.environ,{'EDUGAZE_DATA_ROOT':str(self.root)}):
            at = AppTest.from_file(str(APP),default_timeout=60).run()
        self.assertFalse(at.exception)
        return at

    def assert_healthy(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertFalse(at.error, [e.value for e in at.error])


class DashboardServiceTests(DashboardFixture):
    def test_adapters_preserve_registered_scanpath_and_export_bytes(self):
        data = service.open_dataset(self.root)
        matched = data.load_image('artificial_a.png',participant=1)
        scan = service.generate_scanpath(matched,participant='1',record_index=0,dpi=60)
        interactive = json.loads(scan['interactive_json'])
        self.assertEqual(interactive['data'][1]['x'],[50,850,900])
        self.assertEqual(interactive['data'][1]['y'],[20,200,300])
        self.assertEqual(interactive['layout']['yaxis']['range'],[1050,0])
        self.assertEqual(len(interactive['layout']['sliders'][0]['steps']),3)
        self.assertEqual(interactive['layout']['sliders'][0]['steps'][0]['method'],'restyle')
        heat = service.generate_heatmap(matched,weighting='count',sigma=0,dpi=60)
        self.assertEqual(heat['metadata']['input_weight'],3)
        for png in [scan['png'],heat['png']]:
            with Image.open(io.BytesIO(png)) as image:
                self.assertEqual(image.width,540)
                self.assertAlmostEqual(image.info['dpi'][0],60,places=1)
        exports = service.normalized_exports(data,'artificial_a.png','1')
        rows = pd.read_csv(io.BytesIO(exports['normalized_fixations.csv']))
        self.assertEqual(len(rows),3)
        self.assertEqual(rows.participant_id.unique().tolist(),[1])
        aois = service.parse_aois(AOI_TEXT.encode(),'.csv',(1680,1050))
        analysis = service.analyze_and_export(matched,aois,dpi=60)
        self.assertEqual(analysis['report']['summary']['fixation_count'],3)
        self.assertIsNone(analysis['report']['summary']['time_to_first_fixation_ms'])
        bundle = dict(exports,**analysis['files'],**{'scanpath.png':scan['png']})
        with zipfile.ZipFile(io.BytesIO(service.export_zip(bundle))) as archive:
            self.assertEqual(set(archive.namelist()),set(bundle))
            self.assertEqual(archive.read('scanpath.png'),scan['png'])

    def test_interactive_gaps_and_unknown_durations(self):
        matched = service.open_dataset(self.root).load_image('artificial_a.png',participant=1)
        matched.fixations.loc[matched.fixations.index[1],'inside_image'] = False
        matched.fixations.loc[matched.fixations.index[0],'duration_ms'] = np.nan
        result = service.generate_scanpath(matched,participant='1',record_index=0,dpi=60)
        figure = json.loads(result['interactive_json'])
        self.assertEqual(figure['data'][1]['text'],['1','3'])
        self.assertEqual(figure['data'][0]['x'],[])
        self.assertIn('unavailable',figure['data'][1]['customdata'][0])
        self.assertEqual(result['metadata']['line_segments'],0)

    def test_invalid_aoi_and_missing_dataset(self):
        with self.assertRaises(FileNotFoundError):service.open_dataset(self.root/'missing')
        with self.assertRaises(ValueError):service.parse_aois(b'{}','.json',(1680,1050))
        with self.assertRaises(ValueError):service.parse_aois(b'bad','.txt',(1680,1050))


class DashboardUITests(DashboardFixture):
    def test_overview_explorer_statistics_and_empty_search(self):
        at = self.app()
        self.assertEqual([m.value for m in at.metric],['2','2','4','12'])
        at.radio(key='navigation').set_value('Dataset explorer').run()
        at.text_input(key='image_search').set_value('not-an-image').run()
        self.assertIn('No image identifiers match',at.info[0].value)
        self.assert_healthy(at)
        at.radio(key='navigation').set_value('Statistics').run()
        self.assertEqual(at.metric[0].value,'6')
        self.assert_healthy(at)

    def test_statistics_export_without_aois_and_uploaded_definitions(self):
        at = self.app()
        at.radio(key='navigation').set_value('Exports').run()
        at.button(key='prepare_exports').click().run()
        self.assert_healthy(at)
        files = at.session_state['export_files']
        self.assertIn('participant_statistics.csv',files)
        self.assertEqual(json.loads(files['gaze_statistics.json'])['summary']['fixation_count'],6)
        at.radio(key='navigation').set_value('AOI analysis').run()
        at.selectbox(key='aoi_source').set_value('CSV / JSON').run()
        payload = json.dumps([dict(aoi_id='all',name='Canvas',x=0,y=0,width=1680,height=1050)]).encode()
        at.file_uploader(key='aoi_upload').set_value(('rectangles.json',payload,'application/json')).run()
        at.button(key='analyze_aois').click().run()
        self.assert_healthy(at)
        self.assertEqual(at.session_state['analysis_artifact']['report']['tables']['aoi_summary'][0]['fixation_count'],6)

    def test_missing_empty_corrupt_and_missing_photograph(self):
        at = self.app()
        at.text_input(key='data_root').set_value(str(self.root/'missing')).run()
        self.assertFalse(at.exception);self.assertTrue(at.warning);self.assertTrue(at.error)
        self.path.write_text('[]')
        at.text_input(key='data_root').set_value(str(self.root)).run()
        self.assertFalse(at.exception)
        self.assertIn('no usable scanpaths',at.info[0].value)
        self.path.write_text('{not json')
        at.run();self.assertFalse(at.exception);self.assertTrue(at.error)

    def test_missing_photograph_shows_actionable_error(self):
        (self.root/'images/artifical_nonexistent.png').write_text('irrelevant')
        (self.root/'images/artificial_a.png').unlink()
        at = self.app()
        at.radio(key='navigation').set_value('Gaze workspace').run()
        self.assertFalse(at.exception)
        self.assertIn('No exact photograph match',at.error[0].value)

    def test_invalid_aoi_errors_and_selection_invalidation(self):
        at = self.app()
        at.selectbox(key='participant').set_value('1').run()
        at.radio(key='navigation').set_value('Gaze workspace').run()
        at.button(key='generate_scanpath').click().run()
        at.button(key='generate_heatmap').click().run()
        self.assertIn('scan_artifact',at.session_state)
        self.assertIn('heat_artifact',at.session_state)
        at.slider(key='sigma').set_value(16.).run()
        self.assertNotIn('scan_artifact',at.session_state)
        self.assertNotIn('heat_artifact',at.session_state)
        at.radio(key='navigation').set_value('AOI analysis').run()
        at.selectbox(key='aoi_source').set_value('CSV / JSON').run()
        at.selectbox(key='aoi_format').set_value('JSON').run()
        overlap=[dict(aoi_id='a',name='A',x=0,y=0,width=100,height=100),
                 dict(aoi_id='b',name='B',x=50,y=0,width=100,height=100)]
        at.text_area(key='aoi_text_JSON').set_value(json.dumps(overlap)).run()
        self.assertFalse(at.exception)
        self.assertIn('Overlapping AOIs',at.error[0].value)
        self.assertNotIn('analysis_artifact',at.session_state)
        at.text_area(key='aoi_text_JSON').set_value('not json').run()
        self.assertFalse(at.exception);self.assertTrue(at.error)

    def test_full_fixture_workflow_and_image_change(self):
        at = self.app()
        at.selectbox(key='participant').set_value('1').run()
        at.radio(key='navigation').set_value('Gaze workspace').run()
        at.button(key='generate_scanpath').click().run()
        at.button(key='generate_heatmap').click().run()
        at.radio(key='navigation').set_value('AOI analysis').run()
        at.selectbox(key='aoi_source').set_value('CSV / JSON').run()
        at.text_area(key='aoi_text_CSV').set_value(AOI_TEXT).run()
        at.button(key='analyze_aois').click().run()
        self.assert_healthy(at)
        at.radio(key='navigation').set_value('Exports').run()
        at.button(key='prepare_exports').click().run()
        self.assert_healthy(at)
        files=at.session_state['export_files']
        self.assertIn('scanpath.png',files);self.assertIn('heatmap.png',files);self.assertIn('report.json',files)
        self.assertEqual(json.loads(files['report.json'])['summary']['fixation_count'],3)
        self.assertIsNotNone(at.get('download_button'))
        at.selectbox(key='image').set_value('artificial_b.png').run()
        for key in ['scan_artifact','heat_artifact','analysis_artifact','export_files']:
            self.assertNotIn(key,at.session_state)
        self.assert_healthy(at)


class RealDashboardWorkflowTests(unittest.TestCase):
    def test_real_image_participant_scanpath_heatmap_aoi_exports(self):
        root=REPO/'data/raw/coco_freeview'
        if not (root/service.FIXATIONS).exists() or not (root/'images/000000128398.jpg').exists():
            self.skipTest('Official local COCO-FreeView example unavailable')
        with patch.dict(os.environ,{'EDUGAZE_DATA_ROOT':str(root)}):
            at=AppTest.from_file(str(APP),default_timeout=60).run()
        at.selectbox(key='participant').set_value('101').run()
        at.radio(key='navigation').set_value('Gaze workspace').run()
        at.button(key='generate_scanpath').click().run()
        self.assertEqual(at.session_state['scan_artifact']['metadata']['plotted_fixations'],15)
        at.button(key='generate_heatmap').click().run()
        self.assertEqual(at.session_state['heat_artifact']['metadata']['input_weight'],4238)
        at.radio(key='navigation').set_value('AOI analysis').run()
        at.button(key='analyze_aois').click().run()
        at.radio(key='navigation').set_value('Exports').run()
        at.button(key='prepare_exports').click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertFalse(at.error,[e.value for e in at.error])
        files=at.session_state['export_files']
        self.assertEqual(len(pd.read_csv(io.BytesIO(files['normalized_fixations.csv']))),15)
        report=json.loads(files['report.json'])
        self.assertEqual(report['summary']['total_fixation_duration_ms'],4238)
        self.assertIsNone(report['summary']['time_to_first_fixation_ms'])
        self.assertEqual(len(report['tables']['aoi_summary']),4)
        for name in ['scanpath.png','heatmap.png','aoi_overlay.png']:
            self.assertIsNotNone(cv2.imdecode(np.frombuffer(files[name],np.uint8),cv2.IMREAD_COLOR))
        # Pooled selection must invalidate the individual artifacts and regenerate.
        at.selectbox(key='participant').set_value('All participants').run()
        self.assertNotIn('export_files',at.session_state)
        at.radio(key='navigation').set_value('Gaze workspace').run()
        at.button(key='generate_heatmap').click().run()
        self.assertEqual(at.session_state['heat_artifact']['metadata']['used_fixations'],148)
        self.assertFalse(at.exception)
