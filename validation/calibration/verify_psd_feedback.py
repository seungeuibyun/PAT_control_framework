"""Verify actual FSM/PSD two-axis response, center acquisition and feedback tracking."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import time
import numpy as np
from xy_motion import XYMotion
from laser_termite import TermiteLaser
from find_pd_maximum import SearchCancelled


def main(args):
    folder=Path(__file__).resolve().parent
    motion=XYMotion();report={'started':dt.datetime.now().astimezone().isoformat(),
        'state':'starting','geometry_samples':[],'center_calibration_samples':[],
        'feedback_samples':[],'source_or_stage_commands':False,
        'test_controller':'measured_two_axis_ray_geometry_feedback',
        'PINN_active':False,'PD_alignment_required_for_this_test':False}
    output=Path(args.output) if args.output else folder/(
        'PSD_FSM_feedback_verification_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    source=TermiteLaser().initial
    if source['current_mA']!=32:raise RuntimeError('Fixed 32 mA source is required')
    report['source']=source;last_source_check=time.monotonic();G=None;cancelled=False
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    def fresh():
        nonlocal last_source_check
        s=motion.state();p=s['psd']
        if not p.get('connected') or p.get('age_s',9)>.5 or not p.get('position_available') \
                or not 2<=p['relative_percent']<=90 or p['edge_margin_mm']<.5:
            raise RuntimeError('Fresh unsaturated interior PSD measurement is unavailable')
        if time.monotonic()-last_source_check>25:
            if TermiteLaser().initial['current_mA']!=32:raise RuntimeError('Source current changed during the test')
            last_source_check=time.monotonic()
        return s
    def sample(label,destination):
        states=[];last_count=None;deadline=time.monotonic()+1.5
        while len(states)<5 and time.monotonic()<deadline:
            s=fresh();count=s['psd']['sample_count']
            if count!=last_count:states.append(s);last_count=count
            time.sleep(.04)
        if len(states)<5:raise RuntimeError('Not enough independent PSD samples')
        p=np.array([[s['psd']['x_mm'],s['psd']['y_mm']] for s in states])
        if p.std(0).max()>.02:raise RuntimeError('The optical spot was not stable during sampling')
        row={'label':label,'time':dt.datetime.now().astimezone().isoformat(),
             'independent_PSD_sample_count':len(states),
             'PSD_mm':p.mean(0).tolist(),'PSD_std_mm':p.std(0).tolist(),
             'FSM_feedback_xy':np.mean([s['fsm']['feedback_xy'] for s in states],axis=0).tolist(),
             'FSM_target_xy':states[-1]['fsm']['target_xy'],
             'FSM_current_mA':states[-1]['fsm']['actual_current_mA'],
             'PSD_percent':float(np.mean([s['psd']['relative_percent'] for s in states]))}
        report[destination].append(row);save()
        print(json.dumps({'stage':destination,'point':len(report[destination]),
                          'PSD_mm':row['PSD_mm']}),flush=True)
        return row
    def geometry(rows):
        q=np.array([r['FSM_feedback_xy'] for r in rows]);p=np.array([r['PSD_mm'] for r in rows])
        d=np.column_stack((q-q.mean(0),np.ones(len(q))))
        fit=np.linalg.lstsq(d,p,rcond=None)[0];g=fit[:2].T
        rms=float(np.sqrt(np.mean((d@fit-p)**2)))
        if np.linalg.cond(g)>10 or rms>.03 or np.ptp(q,axis=0).min()<.001:
            raise RuntimeError('Actual two-axis geometry failed condition, span or residual checks')
        return g,rms
    def feedback(target,label,tolerance=.025):
        target=np.array(target,float)
        for i in range(45):
            s=fresh();p=np.array([s['psd']['x_mm'],s['psd']['y_mm']]);err=target-p
            row={'label':label,'time':dt.datetime.now().astimezone().isoformat(),
                 'reference_PSD_mm':target.tolist(),'actual_PSD_mm':p.tolist(),
                 'error_norm_mm':float(np.linalg.norm(err)),'FSM_feedback_xy':s['fsm']['feedback_xy']}
            report['feedback_samples'].append(row);save()
            if np.linalg.norm(err)<=tolerance:
                print(json.dumps({'closed_loop_target':target.tolist(),'measured_PSD':p.tolist(),
                                  'error_mm':row['error_norm_mm']}),flush=True)
                return row
            dq=.75*np.linalg.solve(G,err);dq*=min(1,.0015/max(np.linalg.norm(dq),1e-12))
            next_xy=np.array(s['fsm']['feedback_xy'])+dq
            row['next_xy']=next_xy.tolist();motion.command(next_xy.tolist())
        raise RuntimeError('Actual PSD error failed to converge')
    try:
        fresh()
        if motion.state()['fsm'].get('automatic_search_active'):
            raise RuntimeError('Another automatic controller already owns the mirror')
        motion.start();report['initial_state']=motion.state()
        reference=np.array(motion.state()['fsm']['target_xy'])
        for offset in ([0,0],[.001,0],[-.001,0],[0,0],[0,.001],[0,-.001],[0,0]):
            motion.command((reference+offset).tolist());sample('local_axis_probe','geometry_samples')
        G,rms=geometry(report['geometry_samples'])
        report.update(initial_geometry_mm_per_xy=G.tolist(),initial_geometry_rms_mm=rms)
        feedback([0,0],'center_acquisition')
        center=np.array(fresh()['fsm']['feedback_xy']);report['center_feedback_xy']=center.tolist()
        # Measure both axes around the actual sensor center, independently of PD.
        offsets=np.linspace(-.003,.003,5)
        for j,y in enumerate(offsets):
            for x in offsets if j%2==0 else offsets[::-1]:
                motion.command((center+[x,y]).tolist());sample('center_ray_map','center_calibration_samples')
        G,rms=geometry(report['center_calibration_samples'])
        report.update(center_geometry_mm_per_xy=G.tolist(),center_geometry_rms_mm=rms)
        feedback([0,0],'center_return')
        errors=[]
        for cycle in range(0 if args.skip_circles else 2):
            for angle in np.linspace(0,2*np.pi,12,endpoint=False):
                r=feedback(.3*np.array([np.cos(angle),np.sin(angle)]),f'circle_{cycle+1}')
                errors.append(r['error_norm_mm']);time.sleep(.12)
        feedback([0,0],'final_center_hold')
        report.update(state='real_PSD_FSM_closed_loop_verified',
            reference_tracking_RMS_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            maximum_settled_tracking_error_mm=max(errors) if errors else None,
            circle_verification_performed=not args.skip_circles,physical_PSD_center_hold_verified=True,
            source_stage_position_used=False,absolute_PSD_coordinate_accuracy_verified=False)
    except SearchCancelled as exc:cancelled=True;report.update(state='cancelled',error=str(exc))
    except Exception as exc:report.update(state='aborted',error=str(exc))
    finally:
        if motion.active:
            if not cancelled and report['center_calibration_samples'] and report['state']!='real_PSD_FSM_closed_loop_verified':
                try:motion.command(report['center_calibration_samples'][len(report['center_calibration_samples'])//2]['FSM_target_xy'])
                except Exception as exc:report['return_error']=str(exc)
            try:report['handoff']=motion.close()
            except Exception as exc:report['handoff_error']=str(exc)
        report['final_state']=motion.broker.state();report['finished']=dt.datetime.now().astimezone().isoformat();save()
    if args.train_pinn and report['state']=='real_PSD_FSM_closed_loop_verified':
        import sys
        sys.path.insert(0,str(folder.parent))
        from psd_position_pinn import train_psd_position_pinn
        try:
            model=train_psd_position_pinn(report)
            model.record['dataset_artifact']=str(output.resolve())
            model.record['calibration_time']=report['finished']
            model_path=output.with_name(output.stem+'_PINN_model.json')
            model.save(model_path)
            report.update(PINN_model_path=str(model_path.resolve()),PINN_validation=model.record)
            save()
        except Exception as exc:
            report.update(state='physical_geometry_verified_PINN_training_failed',training_error=str(exc));save()
    print(json.dumps({k:report.get(k) for k in ('state','error','center_geometry_rms_mm',
          'reference_tracking_RMS_mm','maximum_settled_tracking_error_mm','PINN_model_path','training_error')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--skip-circles',action='store_true',help='Collect the real center map without the optional geometric circle test')
    parser.add_argument('--train-pinn',action='store_true',help='Fit and validate a Frozen-PINN on this measured center map')
    main(parser.parse_args())
