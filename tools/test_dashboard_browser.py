#!/usr/bin/env python3
"""Optional local Chrome end-to-end check; requires requirements-dev.txt.

Start Streamlit first. This test uses the installed Chrome, real local records,
actual download buttons, and a new output folder. It never publishes data.
"""
import argparse
from contextlib import contextmanager
import subprocess
import sys
import time
from urllib.parse import urlparse
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import pandas as pd
from PIL import Image
from playwright.sync_api import sync_playwright


@contextmanager
def local_server(args):
    if not args.start_server:
        yield
        return
    target=urlparse(args.url)
    if target.hostname not in {'127.0.0.1','localhost'}:
        raise ValueError('--start-server requires a loopback URL')
    with (args.out/'streamlit.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','streamlit','run',str(Path(__file__).resolve().parents[1]/'streamlit_app.py'),
                                  '--server.address','127.0.0.1','--server.port',str(target.port or 8501),
                                  '--server.headless','true','--server.fileWatcherType','none'],stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError('Streamlit exited at startup; see streamlit.log')
                try:
                    with urllib.request.urlopen(args.url.rstrip('/')+'/_stcore/health',timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError('Streamlit did not become ready')
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8501')
    parser.add_argument('--out',type=Path,default=Path('outputs/dashboard_browser_validation'))
    parser.add_argument('--start-server',action='store_true',help='Start and stop a local Streamlit server for this test')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    with local_server(args):
        run(args)


def run(args):
    with urllib.request.urlopen(args.url.rstrip('/')+'/_stcore/health',timeout=10) as response:
        assert response.status == 200
    chrome=shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:
        raise RuntimeError('Install Chrome/Chromium or run the Streamlit AppTest suite instead')
    errors=[]; console_errors=[]; stage={"name":"startup"}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=chrome,headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
        page=context.new_page();page.set_default_timeout(60000)
        page.on('pageerror',lambda error:errors.append({'message':str(error),'stack':getattr(error,'stack',None),'stage':stage['name']}))
        page.on('console',lambda message:console_errors.append({'message':message.text,'stage':stage['name']}) if message.type=='error' else None)
        page.goto(args.url,wait_until='domcontentloaded')
        page.get_by_text('Dataset overview',exact=True).wait_for()
        page.screenshot(path=str(args.out/'overview.png'))
        # Select the known example explicitly, even if the initial defaults change.
        page.get_by_role('combobox',name='Image',exact=True).click()
        page.get_by_role('option',name='000000128398.jpg',exact=True).click()
        page.get_by_role('combobox',name='Analysis participant',exact=True).click()
        page.get_by_role('option',name='101',exact=True).click()
        stage['name']='scanpath'
        page.get_by_text('Gaze workspace',exact=True).click()
        page.get_by_role('tab',name='Scanpath',exact=True).click()
        page.get_by_role('button',name='Generate scanpath',exact=True).click()
        scan_download=page.get_by_role('button',name='Download scanpath PNG',exact=True)
        scan_download.wait_for()
        stage['name']='fixation_order'
        page.get_by_text('First fixation',exact=True).click()
        page.wait_for_function("Array.from(document.querySelectorAll('.js-plotly-plot')).some(p=>p.layout?.title?.text?.includes('Recorded scanpath') && p.data?.[1]?.x?.length===1 && p._fullLayout?.sliders?.[0]?.active===0)")
        page.get_by_text('Full trial',exact=True).click()
        page.wait_for_function("Array.from(document.querySelectorAll('.js-plotly-plot')).some(p=>p.layout?.title?.text?.includes('Recorded scanpath') && p.data?.[1]?.x?.length===15 && p._fullLayout?.sliders?.[0]?.active===14)")
        with page.expect_download() as pending:
            scan_download.click()
        pending.value.save_as(args.out/'scanpath.png')
        page.screenshot(path=str(args.out/'interactive_scanpath.png'))
        stage['name']='heatmap'
        page.get_by_role('tab',name='Attention heatmap',exact=True).click()
        page.get_by_role('button',name='Generate heatmap',exact=True).click()
        heat_download=page.get_by_role('button',name='Download heatmap PNG',exact=True)
        heat_download.wait_for()
        with page.expect_download() as pending:
            heat_download.click()
        pending.value.save_as(args.out/'heatmap.png')
        page.screenshot(path=str(args.out/'heatmap_workspace.png'))
        stage['name']='aois'
        page.get_by_text('AOI analysis',exact=True).click()
        page.get_by_role('button',name='Analyze AOIs',exact=True).click()
        page.get_by_text('AOI analysis complete. Tables and charts are ready to export.',exact=True).wait_for()
        stage['name']='exports'
        page.get_by_text('Exports',exact=True).click()
        page.get_by_role('button',name='Prepare exports',exact=True).click()
        download=page.get_by_role('button',name='Download complete bundle',exact=True)
        download.wait_for()
        with page.expect_download() as pending:
            download.click()
        pending.value.save_as(args.out/'edugaze_results.zip')
        # Exercise the individual CSV/JSON/PNG controls as well as the ZIP.
        for label,target in [('Download normalized CSV','normalized_fixations.csv'),
                             ('Download JSON report','report.json'),('Download PNG','selected_figure.png')]:
            with page.expect_download() as pending:
                page.get_by_role('button',name=label,exact=True).click()
            pending.value.save_as(args.out/target)
        page.screenshot(path=str(args.out/'exports.png'))
        (args.out/'browser_diagnostics.json').write_text(json.dumps({'errors':errors,'console_errors':console_errors},indent=2)+'\n')
        assert not errors, errors
        browser.close()
    with zipfile.ZipFile(args.out/'edugaze_results.zip') as archive:
        files=archive.namelist()
        frame=pd.read_csv(io.BytesIO(archive.read('normalized_fixations.csv')))
        report=json.loads(archive.read('report.json'))
        assert len(frame)==15 and set(frame.participant_id)=={101}
        assert report['summary']['total_fixation_duration_ms']==4238
        assert report['summary']['time_to_first_fixation_ms'] is None
        assert len(report['tables']['aoi_summary'])==4
        for name in ['scanpath.png','heatmap.png','aoi_overlay.png']:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                assert image.width==2700
    with Image.open(args.out/'scanpath.png') as image:
        assert json.loads(image.info['Description'])['plotted_fixations']==15
        assert abs(image.info['dpi'][0]-300)<.01
    with Image.open(args.out/'heatmap.png') as image:
        assert json.loads(image.info['Description'])['input_weight']==4238
    result={'passed':True,'browser':'installed Chrome/Chromium','image':'000000128398.jpg',
            'participant':'101','fixations_exported':15,'aoi_count':4,'files_in_zip':len(files),
            'workflow':['select image','select participant','generate scanpath','step through fixation order',
                        'generate heatmap','analyze AOIs','download CSV','download JSON','download PNG','download ZIP'],
            'browser_errors':errors}
    (args.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
