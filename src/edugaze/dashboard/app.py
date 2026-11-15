"""Local Streamlit research workspace. Existing modules own all scientific logic."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from edugaze.analysis import analyze_gaze
from . import service

ROOT = Path(__file__).resolve().parents[3]
PAGES = ['Overview', 'Dataset explorer', 'Gaze workspace', 'Statistics', 'AOI analysis', 'Exports']
ARTIFACTS = ['scan_artifact', 'heat_artifact', 'analysis_artifact', 'export_files']


@st.cache_resource(show_spinner='Reading validated fixation records…', max_entries=2)
def _dataset(root, stamp):
    return service.open_dataset(root)


@st.cache_data(show_spinner=False, max_entries=2)
def _inventory(root, stamp):
    return _dataset(root, stamp).inventory()


def _select(label, options, key, default=None):
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]
    return st.selectbox(label, options, key=key)


def _invalidate(keys=ARTIFACTS):
    for key in keys:
        st.session_state.pop(key, None)


def _overview(dataset, inventory):
    st.subheader('Dataset overview')
    st.caption('COCO-FreeView · recorded viewing of natural photographs')
    columns = st.columns(4)
    for column, label, value in zip(columns, ['Images', 'Participants', 'Scanpaths', 'Fixations'],
            [len(inventory), dataset.fixations.participant_id.nunique(), len(dataset.metadata), len(dataset.fixations)]):
        column.metric(label, f'{value:,}')
    left, right = st.columns([2,1])
    with left:
        st.markdown('#### Collection availability')
        statuses = inventory.image_status.value_counts().rename_axis('Photograph status').reset_index(name='Images')
        st.dataframe(statuses, hide_index=True, width='stretch')
        st.markdown('#### Recorded data quality')
        st.dataframe(pd.DataFrame({'Measure':['Valid display coordinates','Positive measured durations'],
                                  'Fixations':[int(dataset.fixations.valid_display.sum()), int(dataset.fixations.valid_duration.sum())]}), hide_index=True, width='stretch')
    with right:
        st.info('Explore an image, inspect a participant’s recorded scanpath, generate a heatmap, then define AOIs and export your results.')
        st.caption('Fixation duration is available. Observed fixation-onset and trial-start timestamps are absent in COCO-FreeView, so time-to-first-fixation remains unavailable.')
    st.markdown('#### Research workflow')
    st.markdown('**Explore → Select participant → Visualize → Analyze AOIs → Export**')
    st.caption('These records describe natural-image viewing, not educational outcomes. No attention model is running.')


def _explorer(dataset, inventory, image):
    st.subheader('Dataset explorer')
    search = st.text_input('Search image identifiers', key='image_search', placeholder='For example: 128398')
    filtered = inventory[inventory.image_name.str.contains(search, case=False, regex=False)]
    st.caption(f'{len(filtered):,} matching images · choose the active image in the sidebar')
    if filtered.empty:
        st.info('No image identifiers match your search.')
    else:
        st.dataframe(filtered,hide_index=True,width='stretch',height=300)
    matched = dataset.load_image(image)
    st.markdown('#### Original photograph')
    st.image(matched.pixels_bgr,channels='BGR',caption=f'{image} · verified fixation canvas',width='stretch')
    st.dataframe(matched.fixations[['participant_id','record_index','fixation_index','x_image','y_image','duration_ms','valid_for_image']],
                 hide_index=True,width='stretch',height=240)


def _gaze(matched, scan_participant, record, settings):
    st.subheader('Human gaze workspace')
    st.caption('Hover for measurements; drag to pan and scroll to zoom. All positions are recorded human fixations.')
    fixations, scanpath, heatmap = st.tabs(['Fixation positions', 'Scanpath', 'Attention heatmap'])
    with fixations:
        st.plotly_chart(service.fixation_figure(matched),width='stretch',config={'scrollZoom':True},key='fixation_chart')
        with st.expander('Original photograph'):
            st.image(matched.pixels_bgr,channels='BGR',width='stretch')
    with scanpath:
        st.caption(f'One trial only · participant {scan_participant} · source record {record}')
        if st.button('Generate scanpath',key='generate_scanpath',type='primary'):
            with st.spinner('Rendering recorded scanpath…'):
                st.session_state.scan_artifact = service.generate_scanpath(matched,participant=scan_participant,
                    record_index=record,show_durations=settings['durations'],dpi=settings['dpi'])
                st.session_state.pop('export_files',None)
        artifact = st.session_state.get('scan_artifact')
        if artifact:
            st.plotly_chart(pio.from_json(artifact['interactive_json']),width='stretch',
                            config={'scrollZoom':True},key='scanpath_chart')
            st.caption('Step through fixation order using the slider or First/Full buttons. These controls do not represent measured elapsed time. Exported PNG includes the full selected trial.')
            st.download_button('Download scanpath PNG',artifact['png'],'scanpath.png','image/png',key='scan_png',on_click='ignore')
            with st.expander('Publication figure and recorded durations'):
                st.image(artifact['png'],width='stretch')
    with heatmap:
        st.caption('Uses the analysis participant selection. “All participants” pools recorded fixations without equal-participant balancing.')
        if st.button('Generate heatmap',key='generate_heatmap',type='primary'):
            with st.spinner('Rendering empirical fixation density…'):
                st.session_state.heat_artifact = service.generate_heatmap(matched,weighting=settings['weighting'],
                    sigma=settings['sigma'],alpha=settings['alpha'],dpi=settings['dpi'])
                st.session_state.pop('export_files',None)
        artifact = st.session_state.get('heat_artifact')
        if artifact:
            st.image(artifact['png'],width='stretch')
            st.caption('Relative density: each map has peak 1. Colors across different maps do not imply equal absolute dwell/counts. This is not model-predicted attention.')
            st.download_button('Download heatmap PNG',artifact['png'],'heatmap.png','image/png',key='heat_png',on_click='ignore')


def _statistics(matched):
    st.subheader('Gaze statistics')
    height,width = matched.pixels_bgr.shape[:2]
    result = analyze_gaze(matched.fixations,image_size=(width,height))
    values = result.summary
    cols = st.columns(4)
    cols[0].metric('Recorded fixations',values['fixation_count'])
    mean = values['mean_fixation_duration_ms']
    cols[1].metric('Mean measured duration',f'{mean:.2f} ms' if mean is not None else 'Unavailable')
    total = values['total_fixation_duration_ms']
    cols[2].metric('Complete fixation duration',f'{total:,.0f} ms' if total is not None else 'Unavailable')
    path = values['total_scanpath_length_px']
    cols[3].metric('Within-trial path length',f'{path:,.2f} px' if path is not None else 'Incomplete')
    if values['timed_trial_count'] == 0:
        st.info('Time to first fixation is unavailable: no complete observed onset/trial-start timing is present. Durations are never used to invent timestamps.')
    st.dataframe(result.participants,hide_index=True,width='stretch')
    left,right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(result.participants,x='participant_id',y='fixation_count',
                        title='Recorded fixations by participant',color_discrete_sequence=['#147d83']),width='stretch',key='participant_counts')
    with right:
        if values['valid_coordinate_count']:
            st.plotly_chart(px.scatter(result.spatial,x='x',y='y',size='fixation_count',
                                      hover_data=['row','column','percentage_of_valid_fixations'],
                                      title='Spatial grid counts (top-left cell coordinates)').update_yaxes(autorange='reversed'),
                            width='stretch',key='spatial_counts')
        else:
            st.info('No valid registered coordinates are available for a spatial chart.')
    with st.expander('Per-trial statistics and consecutive-fixation displacements'):
        st.dataframe(result.trials,hide_index=True,width='stretch')
        st.dataframe(result.saccades,hide_index=True,width='stretch')
    st.caption('Distances are pixel displacements between consecutive valid fixation endpoints, not measured continuous saccade trajectories. Trial boundaries and invalid gaps are never joined.')


def _aoi(matched, settings, provenance):
    st.subheader('Area of Interest analysis')
    height,width = matched.pixels_bgr.shape[:2]
    st.caption('Rectangles use registered image pixels. Left/top edges are included; right/bottom edges are excluded. Overlapping rectangles are rejected.')
    options = ['Table editor','CSV / JSON']
    bundled = ROOT/'configs/aois/coco_000000128398.csv'
    if matched.path.name == '000000128398.jpg' and bundled.is_file():
        options.insert(0,'Bundled example')
    source = _select('AOI source',options,'aoi_source')
    if source == 'Bundled example':
        content,suffix = bundled.read_bytes(),'.csv'
        st.caption('Illustrative manual regions for this photograph, not semantic ground truth or model segmentation.')
    elif source == 'Table editor':
        initial = pd.read_csv(bundled) if matched.path.name == '000000128398.jpg' and bundled.exists() else pd.DataFrame(columns=['aoi_id','name','x','y','width','height'])
        edited = st.data_editor(initial,num_rows='dynamic',hide_index=True,width='stretch',key=f'aoi_editor_{matched.path.name}')
        content,suffix = edited.to_csv(index=False).encode(),'.csv'
    else:
        fmt = st.selectbox('Definition format',['CSV','JSON'],key='aoi_format')
        uploaded = st.file_uploader('Upload AOI definitions',type=['csv','json'],key='aoi_upload')
        text = st.text_area('Or paste definitions',value='aoi_id,name,x,y,width,height\n' if fmt == 'CSV' else '[]',height=170,key=f'aoi_text_{fmt}')
        content = uploaded.getvalue() if uploaded else text.encode()
        suffix = Path(uploaded.name).suffix.lower() if uploaded else '.'+fmt.lower()
    signature = hashlib.sha256(content+suffix.encode()).hexdigest()
    if signature != st.session_state.get('aoi_signature'):
        _invalidate(['analysis_artifact','export_files'])
        st.session_state.aoi_signature = signature
    aois = service.parse_aois(content,suffix,(width,height))
    st.dataframe(pd.DataFrame([asdict(a) for a in aois]),hide_index=True,width='stretch')
    if st.button('Analyze AOIs',key='analyze_aois',type='primary'):
        with st.spinner('Computing validated statistics and export charts…'):
            st.session_state.analysis_artifact = service.analyze_and_export(matched,aois,dpi=settings['dpi'],
                provenance=dict(provenance,aoi_definition_sha256=hashlib.sha256(content).hexdigest()))
            st.session_state.pop('export_files',None)
    artifact = st.session_state.get('analysis_artifact')
    if artifact:
        report = artifact['report']
        st.success('AOI analysis complete. Tables and charts are ready to export.')
        st.dataframe(pd.DataFrame(report['tables']['aoi_summary']),hide_index=True,width='stretch')
        st.image(artifact['files']['aoi_overlay.png'],width='stretch')
        first,second = st.columns(2)
        first.image(artifact['files']['aoi_fixation_counts.png'],width='stretch')
        second.image(artifact['files']['aoi_durations.png'],width='stretch')
        with st.expander('Transitions and participant-level AOI results'):
            st.image(artifact['files']['aoi_transitions.png'],width='stretch')
            st.dataframe(pd.DataFrame(report['tables']['participant_aoi_statistics']),hide_index=True,width='stretch')
        st.caption('“Viewed” means a recorded fixation hit. Time to first fixation remains unavailable without valid observed onset and trial-start timestamps.')


def _exports(dataset, matched, image, participant, provenance):
    st.subheader('Export research results')
    st.caption('Exports include only the current selection. Generate plots and AOI analysis first to include them. Files download locally; raw datasets are not modified.')
    if st.button('Prepare exports',key='prepare_exports',type='primary'):
        with st.spinner('Preparing normalized records and result files…'):
            files = service.normalized_exports(dataset,image,participant)
            files.update(service.statistics_exports(matched,provenance=provenance))
            for key,name in [('scan_artifact','scanpath.png'),('heat_artifact','heatmap.png')]:
                if key in st.session_state:
                    artifact = st.session_state[key]
                    files[name] = artifact['png']
                    files[name+'.metadata.json'] = json.dumps(artifact['metadata'],indent=2,allow_nan=False).encode()
            if 'analysis_artifact' in st.session_state:
                files.update(st.session_state.analysis_artifact['files'])
            st.session_state.export_files = files
    files = st.session_state.get('export_files')
    if not files:
        st.info('Prepare exports to download normalized CSV records, JSON provenance, available PNG figures and a combined ZIP.')
        return
    st.success(f'{len(files)} files ready for the current selection.')
    st.download_button('Download normalized CSV',files['normalized_fixations.csv'],'normalized_fixations.csv','text/csv',key='download_csv',on_click='ignore')
    json_name = 'report.json' if 'report.json' in files else 'gaze_statistics.json'
    st.download_button('Download JSON report',files[json_name],json_name,'application/json',key='download_json',on_click='ignore')
    pngs = sorted(name for name in files if name.endswith('.png'))
    if pngs:
        name = st.selectbox('PNG figure',pngs,key='png_choice')
        st.download_button('Download PNG',files[name],name,'image/png',key='download_png',on_click='ignore')
    csvs = sorted(name for name in files if name.endswith('.csv'))
    name = st.selectbox('CSV table',csvs,key='csv_choice')
    st.download_button('Download selected table',files[name],name,'text/csv',key='download_table',on_click='ignore')
    st.download_button('Download complete bundle',service.export_zip(files),'edugaze_results.zip','application/zip',key='download_zip',on_click='ignore')
    st.dataframe(pd.DataFrame({'File':list(files),'Bytes':[len(v) for v in files.values()]}),hide_index=True,width='stretch')


def main():
    st.set_page_config(page_title='EduGaze · Research workspace',page_icon='◉',layout='wide')
    st.markdown('''<style>.block-container{max-width:1400px;padding-top:2rem} [data-testid="stMetric"]{background:white;border:1px solid #dce5eb;border-radius:12px;padding:16px} h1{letter-spacing:-.04em} [data-testid="stSidebar"]{border-right:1px solid #dce5eb}</style>''',unsafe_allow_html=True)
    st.title('EduGaze')
    st.caption('VISUAL ATTENTION RESEARCH WORKSPACE · Recorded human gaze')
    with st.sidebar:
        st.markdown('### Research navigation')
        page = st.radio('Page',PAGES,key='navigation',label_visibility='collapsed')
        with st.expander('Dataset location'):
            root = st.text_input('COCO-FreeView directory',value=os.environ.get('EDUGAZE_DATA_ROOT',str(ROOT/'data/raw/coco_freeview')),key='data_root')
            if st.button('Refresh dataset',key='refresh_dataset'):
                _dataset.clear();_inventory.clear();_invalidate()
    try:
        root,stamp = service.dataset_stamp(root)
        dataset = _dataset(root,stamp)
        inventory = _inventory(root,stamp)
    except (OSError,ValueError,TypeError) as exc:
        st.warning('The dataset could not be loaded. Set the local COCO-FreeView directory in the sidebar.')
        st.error(str(exc))
        st.code('python tools/download_coco_freeview.py --all-images',language='bash')
        st.caption('Expected: fixations/COCOFreeView_fixations_trainval.json and images/ under the selected directory. No synthetic data is substituted.')
        return
    if inventory.empty:
        st.info('This dataset contains no usable scanpaths. Choose a nonempty fixation JSON.')
        return
    with st.sidebar:
        image = _select('Image',inventory.image_name.tolist(),'image',default='000000128398.jpg')
        raw = dataset.select(image=image)
        people = sorted(raw.participant_id.unique().tolist())
        selected = _select('Analysis participant',['All participants']+people,'participant')
        participant = None if selected == 'All participants' else selected
        scan_participant = _select('Scanpath participant',people,'scan_participant',default='101') if participant is None else participant
        trials = sorted(int(x) for x in raw[raw.participant_id.eq(scan_participant)].record_index.unique())
        record = _select('Recorded trial',trials,'scan_record')
        with st.expander('Visualization settings'):
            weighting = st.selectbox('Heatmap weighting',['duration','count'],key='weighting',help='Duration uses positive measured milliseconds; count uses one per valid position.')
            sigma = st.slider('Gaussian smoothing (image pixels)',0.,100.,32.,1.,key='sigma')
            alpha = st.slider('Maximum heatmap opacity',0.,1.,.65,.05,key='alpha')
            dpi = st.selectbox('PNG resolution (DPI)',[150,300,600],index=1,key='dpi')
            durations = st.checkbox('Show scanpath duration table',True,key='durations')
        st.caption('Local workspace · No machine learning')
    settings = {'weighting':weighting,'sigma':sigma,'alpha':alpha,'dpi':dpi,'durations':durations}
    context = (root,stamp,image,participant,scan_participant,record,tuple(settings.items()))
    if context != st.session_state.get('selection_context'):
        _invalidate();st.session_state.selection_context = context
    if page == 'Overview':
        _overview(dataset,inventory);return
    try:
        if page == 'Dataset explorer':
            _explorer(dataset,inventory,image);return
        matched = dataset.load_image(image,participant=participant)
        st.caption(f'{image} · {selected} · {len(matched.fixations)} recorded fixations')
        provenance = {'source_sha256':dataset.source_sha256,'image_name':image,
                      'image_sha256':hashlib.sha256(matched.path.read_bytes()).hexdigest(),
                      'participant_filter':participant,'geometry':vars(matched.geometry)}
        if page == 'Gaze workspace': _gaze(matched,scan_participant,record,settings)
        elif page == 'Statistics': _statistics(matched)
        elif page == 'AOI analysis': _aoi(matched,settings,provenance)
        elif page == 'Exports': _exports(dataset,matched,image,participant,provenance)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        st.error(f'Unable to complete this selection: {exc}')
        st.caption('Check photograph availability, participant selection and AOI coordinates. A differently sized photograph requires explicit registration through the Phase 2 loader.')
