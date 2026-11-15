"""Artificial arithmetic fixtures; main integration example uses real human gaze."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from edugaze.analysis import AOI, analyze_aois, analyze_gaze, load_aois, saccade_distances, validate_aois
from edugaze.analysis.reports import export_analysis
from edugaze.datasets import CocoFreeView, ImageGeometry, MatchedImage


def records(points=None, durations=None):
    points = [(0,0),(3,4),(6,8)] if points is None else points
    n = len(points)
    return pd.DataFrame({'image_name':['artificial.png']*n, 'participant_id':['p1']*n,
                         'record_index':[0]*n,'fixation_index':list(range(n)),
                         'x_image':[p[0] for p in points], 'y_image':[p[1] for p in points],
                         'duration_ms':[100,200,300] if durations is None and n == 3 else ([100]*n if durations is None else durations),
                         'image_width':[10]*n,'image_height':[10]*n,'inside_image':[True]*n})


class GazeStatisticsTests(unittest.TestCase):
    def test_known_distances_durations_distribution(self):
        result = analyze_gaze(records(), image_size=(10,10), grid=(2,2))
        self.assertEqual(result.summary['fixation_count'],3)
        self.assertEqual(result.summary['mean_fixation_duration_ms'],200)
        self.assertEqual(result.summary['total_fixation_duration_ms'],600)
        self.assertEqual(result.summary['total_scanpath_length_px'],10)
        np.testing.assert_array_equal(result.saccades.distance_px,[5,5])
        self.assertEqual(result.spatial.fixation_count.tolist(),[2,0,0,1])
        self.assertAlmostEqual(result.spatial.percentage_of_valid_fixations.sum(),100)
        self.assertIsNone(result.summary['time_to_first_fixation_ms'])
        self.assertEqual(result.trials.ttff_status.iloc[0],'timing_unavailable')

    def test_missing_invalid_durations_do_not_become_zero(self):
        df = records(durations=[100,None,-20])
        summary = analyze_gaze(df,image_size=(10,10)).summary
        self.assertEqual(summary['mean_fixation_duration_ms'],100)
        self.assertEqual(summary['observed_fixation_duration_ms'],100)
        self.assertIsNone(summary['total_fixation_duration_ms'])
        self.assertEqual(summary['duration_status'],'partial')
        df = df.drop(columns='duration_ms')
        summary = analyze_gaze(df,image_size=(10,10)).summary
        self.assertIsNone(summary['mean_fixation_duration_ms'])
        self.assertIsNone(summary['observed_fixation_duration_ms'])
        self.assertEqual(summary['duration_status'],'unavailable')

    def test_no_cross_participant_or_trial_distances(self):
        first = records()
        df = pd.concat([first, first.assign(participant_id='p2'), first.assign(record_index=1)], ignore_index=True)
        result = analyze_gaze(df,image_size=(10,10))
        self.assertEqual(result.summary['total_scanpath_length_px'],30)
        self.assertEqual(result.summary['saccade_count'],6)
        self.assertEqual(result.summary['trial_count'],3)
        per = result.participants.set_index('participant_id')
        self.assertEqual(per.loc['p1','total_scanpath_length_px'],20)
        self.assertEqual(per.loc['p2','total_scanpath_length_px'],10)

    def test_invalid_coordinates_and_index_gaps_not_bridged(self):
        df = records()
        df.loc[1,'x_image'] = np.nan
        result = analyze_gaze(df,image_size=(10,10))
        self.assertEqual(len(result.saccades),0)
        self.assertEqual(result.summary['valid_coordinate_count'],2)
        self.assertIsNone(result.summary['total_scanpath_length_px'])
        self.assertEqual(result.summary['observed_scanpath_length_px'],0)
        self.assertEqual(result.spatial.fixation_count.sum(),2)
        df = records().iloc[[0,2]]
        self.assertTrue(saccade_distances(df,image_size=(10,10)).empty)
        self.assertIsNone(analyze_gaze(df,image_size=(10,10)).summary['total_scanpath_length_px'])

    def test_observed_timing_and_origin_required(self):
        df = records().assign(fixation_onset_ms=[1010,1130,1370],trial_start_ms=1000)
        result = analyze_gaze(df,image_size=(10,10))
        self.assertEqual(result.summary['time_to_first_fixation_ms'],10)
        self.assertEqual(result.summary['timed_trial_count'],1)
        zero = analyze_gaze(df.assign(fixation_onset_ms=[1000,1130,1370]), image_size=(10,10))
        self.assertEqual(zero.summary['time_to_first_fixation_ms'],0)
        summary = analyze_gaze(df.drop(columns='trial_start_ms'),image_size=(10,10)).summary
        self.assertIsNone(summary['time_to_first_fixation_ms'])
        # Multiple trials are summarized as separate latencies, not a shared clock.
        other = df.assign(record_index=1,fixation_onset_ms=[2020,2130,2370],trial_start_ms=2000)
        combined = analyze_gaze(pd.concat([df,other]),image_size=(10,10))
        self.assertIsNone(combined.summary['time_to_first_fixation_ms'])
        self.assertEqual(combined.summary['mean_trial_time_to_first_fixation_ms'],15)

    def test_invalid_and_incomplete_timing_never_estimated(self):
        base = records().assign(fixation_onset_ms=[1010,1130,1370],trial_start_ms=1000)
        changes = [base.assign(fixation_onset_ms=[999,1130,1370]),
                   base.assign(fixation_onset_ms=[1010,1370,1130]),
                   base.assign(fixation_onset_ms=[1010,None,1370]),
                   base.assign(trial_start_ms=[1000,1001,1000]),base.iloc[[1,2]]]
        invalid = base.copy(); invalid.loc[1,'x_image']=np.nan; changes.append(invalid)
        for df in changes:
            with self.subTest(df=df.to_dict()):
                result = analyze_gaze(df,image_size=(10,10))
                self.assertIsNone(result.summary['time_to_first_fixation_ms'])
                self.assertNotEqual(result.trials.ttff_status.iloc[0],'available')

    def test_bad_schema_dimensions_duplicates_and_no_input_mutation(self):
        original = records(); before=original.copy(deep=True)
        analyze_gaze(original,image_size=(10,10))
        pd.testing.assert_frame_equal(before,original)
        for df, size in [(original,(20,10)),(original.iloc[:0],(10,10)),
                         (pd.concat([original,original]),(10,10)),
                         (original.assign(fixation_index=[0,1.5,2]),(10,10)),
                         (original.assign(fixation_index=[0,np.bool_(True),2]),(10,10)),
                         (original.drop(columns='fixation_index'),(10,10)),
                         (original.assign(participant_id=None),(10,10))]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                analyze_gaze(df,image_size=size)
        with self.assertRaises(ValueError):
            analyze_gaze(original,image_size=(10,10),grid=(0,2))

    def test_singleton_and_all_invalid_coordinates(self):
        self.assertEqual(analyze_gaze(records([(1,1)]),image_size=(10,10)).summary['total_scanpath_length_px'],0)
        df=records([(np.inf,0),(-1,0),(10,0)])
        result=analyze_gaze(df,image_size=(10,10))
        self.assertEqual(result.summary['valid_coordinate_count'],0)
        self.assertIsNone(result.summary['mean_x_px'])
        self.assertTrue(result.spatial.percentage_of_valid_fixations.isna().all())


class AOITests(unittest.TestCase):
    def setUp(self):
        self.aois=[AOI('A','Left',0,0,5,10),AOI('B','Right',5,0,5,10)]

    def test_boundary_counts_and_both_percentage_denominators(self):
        df=records([(0,0),(4.999,9.999),(5,0),(9.999,9.999),(10,0),(-.1,0),(0,10)])
        result=analyze_aois(df,self.aois,image_size=(10,10))
        self.assertEqual(result.summary.fixation_count.tolist(),[2,2])
        self.assertEqual(result.summary.total_fixation_duration_ms.tolist(),[200,200])
        np.testing.assert_allclose(result.summary.percentage_of_fixations,[200/7,200/7])
        np.testing.assert_allclose(result.summary.percentage_of_valid_fixations,[50,50])
        self.assertEqual(result.assignments.aoi_id.tolist(),['A','A','B','B',None,None,None])

    def test_viewed_missing_dwell_and_never_viewed_aoi(self):
        df=records([(0,0),(3,4)],[None,None])
        result=analyze_aois(df,self.aois,image_size=(10,10))
        left,right=result.summary.iloc[0],result.summary.iloc[1]
        self.assertTrue(pd.isna(left.total_fixation_duration_ms))
        self.assertTrue(pd.isna(left.observed_fixation_duration_ms))
        self.assertEqual(right.fixation_count,0)
        self.assertEqual(right.total_fixation_duration_ms,0)
        self.assertEqual(result.participants.viewed.tolist(),[True,False])
        self.assertTrue(result.trials.time_to_first_fixation_ms.isna().all())

    def test_direct_transitions_repeats_outside_gaps_and_trial_boundaries(self):
        aois=[AOI('A','A',0,0,3,10),AOI('B','B',6,0,3,10)]
        df=records([(1,1),(4,1),(7,1),(1,1),(2,1)])
        result=analyze_aois(df,aois,image_size=(10,10))
        self.assertEqual(result.transitions.from_aoi_id.tolist(),['B'])
        self.assertEqual(result.transitions.to_aoi_id.tolist(),['A'])
        self.assertEqual(result.transition_counts.transition_count.tolist(),[1])
        gap=records([(1,1),(7,1)]).assign(fixation_index=[0,2])
        self.assertTrue(analyze_aois(gap,aois,image_size=(10,10)).transitions.empty)
        separate=pd.concat([records([(1,1)]),records([(7,1)]).assign(record_index=1)])
        self.assertTrue(analyze_aois(separate,aois,image_size=(10,10)).transitions.empty)

    def test_actual_timing_aoi_ttff_and_absent_clock(self):
        df=records().assign(fixation_onset_ms=[1010,1130,1370],trial_start_ms=1000)
        result=analyze_aois(df,self.aois,image_size=(10,10))
        self.assertEqual(result.trials.time_to_first_fixation_ms.tolist(),[10,370])
        missing=analyze_aois(df.drop(columns='trial_start_ms'),self.aois,image_size=(10,10))
        self.assertTrue(missing.trials.time_to_first_fixation_ms.isna().all())
        self.assertEqual(missing.trials.ttff_status.tolist(),['timing_unavailable']*2)

    def test_participants_viewed_and_timing_aggregated_by_trial(self):
        df=records().assign(fixation_onset_ms=[1010,1130,1370],trial_start_ms=1000)
        other=records([(1,1)]).assign(participant_id='p2',fixation_onset_ms=1050,trial_start_ms=1000)
        result=analyze_aois(pd.concat([df,other]),self.aois,image_size=(10,10))
        self.assertEqual(result.summary.participants_viewing.tolist(),[2,1])
        self.assertEqual(result.summary.mean_trial_time_to_first_fixation_ms.tolist(),[30,370])
        self.assertIsNone(result.summary.iloc[0].time_to_first_fixation_ms)
        right=result.participants[(result.participants.participant_id=='p2') & (result.participants.aoi_id=='B')].iloc[0]
        self.assertFalse(right.viewed)
        self.assertEqual(right.ttff_status,'not_viewed')

    def test_csv_json_loading_leading_zero_ids_and_bad_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            csv=path/'a.csv'; csv.write_text('aoi_id,name,x,y,width,height\n001,Left,0,0,5,10\n002,Right,5,0,5,10\n')
            loaded=load_aois(csv,image_size=(10,10))
            self.assertEqual(loaded[0].aoi_id,'001')
            js=path/'a.json';js.write_text(json.dumps([asdict(a) for a in loaded]))
            self.assertEqual(load_aois(js,image_size=(10,10)),loaded)
            for payload in [{},[{'aoi_id':'A'}], [dict(asdict(loaded[0]),width='bad')]]:
                js.write_text(json.dumps(payload))
                with self.assertRaises(ValueError): load_aois(js,image_size=(10,10))

    def test_invalid_geometry_overlap_duplicate_and_touching(self):
        validate_aois(self.aois,image_size=(10,10))
        for aois in [[],[self.aois[0],self.aois[0]], [AOI('x','x',0,0,11,2)],
                     [self.aois[0],AOI('C','C',4,0,2,2)]]:
            with self.subTest(aois=aois),self.assertRaises(ValueError): validate_aois(aois,image_size=(10,10))
        for values in [('', 'a',0,0,1,1),('a','',0,0,1,1),('a','a',-1,0,1,1),
                       ('a','a',0,0,0,1),('a','a',0,0,np.nan,1),('a','a',False,0,1,1)]:
            with self.subTest(values=values),self.assertRaises(ValueError): AOI(*values)

    def test_aligned_original_image_coordinates_used_for_aois(self):
        # Artificial trial tests the same loader registration used for diagrams.
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);cv2.imwrite(str(p/'artificial.png'),np.zeros((600,800,3),dtype=np.uint8))
            (p/'test.json').write_text(json.dumps([dict(name='artificial.png',subject=1,X=[840],Y=[525],T=[100])]))
            matched=CocoFreeView(p/'test.json',p).load_image('artificial.png',geometry=ImageGeometry(140,0,1400,1050))
            result=analyze_aois(matched.fixations,[AOI('center','Center',390,290,20,20)],image_size=(800,600))
            self.assertEqual(result.summary.fixation_count.iloc[0],1)
            self.assertEqual(result.assignments.x_image.iloc[0],400)

    def test_reports_json_nulls_csv_and_png_generation(self):
        df=records(durations=[None,None,None])
        matched=MatchedImage(Path('artificial.png'),np.zeros((10,10,3),dtype=np.uint8),df,ImageGeometry(0,0,10,10))
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/'report'
            export_analysis(matched,self.aois,destination,dpi=60)
            report=json.loads((destination/'report.json').read_text(),parse_constant=lambda x: self.fail(f'Nonstandard JSON {x}'))
            self.assertIsNone(report['summary']['total_fixation_duration_ms'])
            self.assertIsNone(report['summary']['time_to_first_fixation_ms'])
            self.assertTrue(pd.read_csv(destination/'trial_statistics.csv').time_to_first_fixation_ms.isna().all())
            self.assertEqual(len(list(destination.glob('*.png'))),5)
            for path in destination.glob('*.png'): self.assertIsNotNone(cv2.imread(str(path)))
            with self.assertRaises(FileExistsError): export_analysis(matched,self.aois,destination,dpi=60)


class RealAOIIntegrationTests(unittest.TestCase):
    def test_real_coco_statistics_and_manual_aoi_definitions(self):
        repo=Path(__file__).resolve().parents[1]; root=repo/'data/raw/coco_freeview'
        source=root/'fixations/COCOFreeView_fixations_trainval.json'
        if not source.exists() or not (root/'images/000000128398.jpg').exists():
            self.skipTest('Official local real example unavailable')
        matched=CocoFreeView(source,root/'images').load_image('000000128398.jpg')
        stats=analyze_gaze(matched.fixations,image_size=(1680,1050))
        self.assertEqual(stats.summary['fixation_count'],len(matched.fixations))
        self.assertEqual(stats.summary['total_fixation_duration_ms'],matched.fixations.duration_ms.sum())
        self.assertIsNone(stats.summary['time_to_first_fixation_ms'])
        self.assertEqual(stats.summary['timed_trial_count'],0)
        self.assertEqual(stats.spatial.fixation_count.sum(),len(matched.fixations))
        csv=load_aois(repo/'configs/aois/coco_000000128398.csv',image_size=(1680,1050))
        js=load_aois(repo/'configs/aois/coco_000000128398.json',image_size=(1680,1050))
        self.assertEqual(csv,js)
        result=analyze_aois(matched.fixations,csv,image_size=(1680,1050))
        self.assertTrue(result.trials.time_to_first_fixation_ms.isna().all())
        self.assertLessEqual(result.summary.fixation_count.sum(),len(matched.fixations))
        self.assertEqual(len(result.participants),len(csv)*matched.fixations.participant_id.nunique())
