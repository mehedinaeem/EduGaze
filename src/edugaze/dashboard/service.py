"""Dashboard adapters. Scientific calculations stay in the existing modules."""
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from edugaze.analysis import analyze_gaze, load_aois
from edugaze.analysis.reports import _json_safe, _table_records, export_analysis
from edugaze.datasets import CocoFreeView
from edugaze.visualization import plot_heatmap, plot_scanpath, save_png
from edugaze.visualization._common import coordinate_mask, select_rows


FIXATIONS = Path('fixations/COCOFreeView_fixations_trainval.json')


def dataset_stamp(root):
    root = Path(root).expanduser().resolve()
    source = root/FIXATIONS
    if not source.is_file():
        raise FileNotFoundError(f'Fixation JSON not found: {source}')
    paths = [source, root/'images', root/'download_manifest.json']
    return str(root), tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists())


def open_dataset(root):
    root, _ = dataset_stamp(root)
    return CocoFreeView(Path(root)/FIXATIONS, Path(root)/'images')


def parse_aois(content, suffix, image_size):
    if suffix.lower() not in {'.csv', '.json'}:
        raise ValueError('Use a CSV or JSON AOI definition')
    if len(content) > 2*1024*1024:
        raise ValueError('AOI definition exceeds 2 MB')
    with tempfile.TemporaryDirectory(prefix='edugaze-aois-') as folder:
        path = Path(folder)/f'aois{suffix.lower()}'
        path.write_bytes(content)
        return load_aois(path, image_size=image_size)


def _png(plot, dpi):
    try:
        with tempfile.TemporaryDirectory(prefix='edugaze-figure-') as folder:
            path = save_png(plot, Path(folder)/'figure.png', dpi=dpi)
            return path.read_bytes()
    finally:
        plot.figure.clear()


def fixation_figure(matched):
    """Hoverable recorded positions using aligned values; no trajectory inference."""
    df = select_rows(matched)
    points = df[coordinate_mask(df,matched.pixels_bgr.shape)]
    if points.empty:
        raise ValueError('No valid recorded positions for this selection')
    figure = go.Figure()
    for participant, group in points.groupby('participant_id', sort=False):
        figure.add_trace(go.Scatter(x=group.x_image.tolist(), y=group.y_image.tolist(),
                                   name=str(participant), mode='markers', marker={'size':9,'opacity':.8},
                                   customdata=[[int(row.fixation_index), f'{row.duration_ms:g} ms' if np.isfinite(row.duration_ms) and row.duration_ms > 0 else 'unavailable'] for row in group.itertuples()],
                                   hovertemplate='Participant '+str(participant)+'<br>x=%{x:.1f}, y=%{y:.1f}<br>Source index=%{customdata[0]}<br>Duration=%{customdata[1]}<extra></extra>'))
    return _on_photo(figure, matched, 'Recorded fixation positions')


def _on_photo(figure, matched, title):
    height, width = matched.pixels_bgr.shape[:2]
    figure.add_layout_image(source=Image.fromarray(cv2.cvtColor(matched.pixels_bgr, cv2.COLOR_BGR2RGB)),
                            xref='x', yref='y', x=0, y=0, sizex=width, sizey=height,
                            xanchor='left', yanchor='top', sizing='stretch', layer='below')
    figure.update_layout(title=title, height=620, margin={'l':20,'r':20,'t':50,'b':30},
                         paper_bgcolor='white', plot_bgcolor='white', dragmode='pan',
                         xaxis={'range':[0,width],'title':'Image x (pixels)','constrain':'domain','showgrid':False,'zeroline':False},
                         yaxis={'range':[height,0],'title':'Image y (pixels)','scaleanchor':'x','scaleratio':1,'showgrid':False,'zeroline':False})
    return figure


def generate_scanpath(matched, *, participant, record_index, show_durations=True, dpi=300):
    """Adapt tested Matplotlib scanpath topology into interactive ordinal controls."""
    plot = plot_scanpath(matched, participant=participant, record_index=record_index,
                         show_durations=show_durations)
    xy = np.asarray(plot.axes.collections[1].get_offsets())
    segments = list(plot.axes.collections[0].get_segments())
    numbers = [int(t.get_text()) for t in plot.axes.texts]
    selected = matched.fixations[(matched.fixations.participant_id.eq(str(participant)))
                                  & matched.fixations.record_index.eq(record_index)].set_index('fixation_index')
    hover = []
    for number in numbers:
        duration = selected.loc[number-1,'duration_ms']
        text = f'{duration:g} ms' if np.isfinite(duration) and duration > 0 else 'unavailable'
        hover.append(f'Fixation {number}<br>Duration: {text}')

    def traces(count):
        # The prefix uses the existing plot's segments, including its gap policy.
        segment_count = sum(b == a+1 for a,b in zip(numbers[:count],numbers[1:count]))
        xs, ys = [], []
        for segment in segments[:segment_count]:
            xs.extend([segment[0,0],segment[1,0],None]);ys.extend([segment[0,1],segment[1,1],None])
        return [go.Scatter(x=xs,y=ys,mode='lines',line={'color':'#f8d35d','width':3},
                           hoverinfo='skip',showlegend=False),
                go.Scatter(x=xy[:count,0].tolist(),y=xy[:count,1].tolist(),mode='markers+text',
                           text=[str(n) for n in numbers[:count]],textposition='top center',
                           marker={'size':13,'color':'#147d83','line':{'width':1,'color':'white'}},
                           customdata=hover[:count],hovertemplate='%{customdata}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>',
                           showlegend=False)]
    def update_args(count):
        line, points = traces(count)
        return [{'x':[line.x,points.x], 'y':[line.y,points.y],
                 'text':[None,points.text], 'customdata':[None,points.customdata]},[0,1]]
    figure = go.Figure(data=traces(len(xy)))
    _on_photo(figure,matched,f'Recorded scanpath · participant {participant}')
    figure.update_layout(updatemenus=[{'type':'buttons','direction':'left','x':0,'y':-.16,
        'buttons':[{'label':'First fixation','method':'update','args':[update_args(1)[0],{'sliders[0].active':0},[0,1]]},
                   {'label':'Full trial','method':'update','args':[update_args(len(xy))[0],{'sliders[0].active':len(xy)-1},[0,1]]}]}],
        sliders=[{'active':len(xy)-1,'x':.32,'len':.65,'y':-.12,'currentvalue':{'prefix':'Through fixation '},
                  'steps':[{'label':str(numbers[n-1]),'method':'restyle','args':update_args(n)} for n in range(1,len(xy)+1)]}],
        margin={'l':20,'r':20,'t':50,'b':110},height=700)
    metadata = plot.metadata
    png = _png(plot,dpi)
    return {'png':png,'interactive_json':figure.to_json(),'metadata':metadata}


def generate_heatmap(matched, *, weighting='duration', sigma=32., alpha=.65, dpi=300):
    plot = plot_heatmap(matched,weighting=weighting,sigma=sigma,alpha=alpha)
    metadata = plot.metadata
    return {'png':_png(plot,dpi),'metadata':metadata}


def analyze_and_export(matched, aois, *, dpi=300, provenance=None):
    """Use the Phase 4 report writer and return its exact generated file bytes."""
    with tempfile.TemporaryDirectory(prefix='edugaze-analysis-') as folder:
        destination = Path(folder)/'report'
        export_analysis(matched,aois,destination,dpi=dpi,provenance=provenance)
        files = {p.name:p.read_bytes() for p in destination.iterdir() if p.is_file()}
    files['aoi_definitions.json'] = json.dumps([asdict(a) for a in aois],indent=2,allow_nan=False).encode()
    return {'files':files,'report':json.loads(files['report.json'])}


def normalized_exports(dataset, image, participant):
    with tempfile.TemporaryDirectory(prefix='edugaze-normalized-') as folder:
        path = Path(folder)/'normalized_fixations.csv'
        dataset.export_csv(path,image=image,participant=participant)
        return {p.name:p.read_bytes() for p in Path(folder).iterdir() if p.is_file()}



def statistics_exports(matched, *, provenance=None):
    height,width = matched.pixels_bgr.shape[:2]
    result = analyze_gaze(matched.fixations,image_size=(width,height))
    tables = {'gaze_summary':pd.DataFrame([result.summary]), 'participant_statistics':result.participants,
              'trial_statistics':result.trials, 'saccade_distances':result.saccades,
              'spatial_distribution':result.spatial}
    files = {name+'.csv':table.to_csv(index=False,na_rep='').encode() for name,table in tables.items()}
    report = {'metadata':{'data_kind':'recorded_human_gaze','image_name':matched.path.name,
                           'image_size':[width,height],'provenance':provenance or {}},
              'summary':result.summary,'tables':{name:_table_records(table) for name,table in tables.items()}}
    files['gaze_statistics.json'] = json.dumps(_json_safe(report),indent=2,allow_nan=False).encode()
    return files


def export_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(Path(name).name,content)
    return buffer.getvalue()
