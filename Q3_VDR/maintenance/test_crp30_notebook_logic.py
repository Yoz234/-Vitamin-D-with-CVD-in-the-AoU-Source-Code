"""Synthetic-data checks only. Never loads participant data or runs an AoU model."""
import ast
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_crp30_notebooks as build


class CRP30Tests(unittest.TestCase):
    def setUp(self):
        self.ns = {'np': np, 'pd': pd}
        tree = ast.parse(build.PREP)
        keep = [n for n in tree.body if
                isinstance(n, ast.FunctionDef) and n.name == 'select_mask' or
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'rows' for t in n.targets)]
        exec(compile(ast.Module(body=keep, type_ignores=[]), '<synthetic-mask-test>', 'exec'), self.ns)
        self.spec = pd.DataFrame(self.ns['rows'], columns=['model_dataset','max_days','exclude_gt10','exclude_ambiguous','analysis_role'])
        self.data = pd.DataFrame({'days_before':[-1,0,1,30,31,365,366,1825,1826,np.nan],
                                  'crp_mgL':[1,11,4,10,3,20,5,3,4,1],
                                  'ambiguous_unit':[0,1,0,0,0,0,1,0,np.nan,0]})

    def test_boundaries_and_legacy_retention(self):
        masks = {r.model_dataset:self.ns['select_mask'](self.data,r) for r in self.spec.itertuples(index=False)}
        self.assertEqual(list(self.data.index[masks['crp_30day_primary']]), [1,2,3])
        self.assertEqual(int(masks['all_crp_model_base'].sum()), 8)
        self.assertEqual(int(masks['same_day_or_365d_before'].sum()), 5)
        self.assertEqual(int(masks['same_day_or_5y_before'].sum()), 7)
        self.assertEqual(len(masks)*4,40)
        for key in ['crp_30day_exclude_gt10','crp_30day_exclude_ambiguous','crp_30day_exclude_both']:
            self.assertTrue((~masks[key] | masks['crp_30day_primary']).all())
        self.assertFalse(masks['exclude_ambiguous_unit'].iloc[8])

    def test_flag_partition_with_unknown_overlap(self):
        d = pd.DataFrame({'crp_mgL':[12,12,12,1,1], 'ambiguous_unit':[1,0,np.nan,0,1]})
        high,amb = d.crp_mgL.gt(10),d.ambiguous_unit.eq(1)
        counts = [int((high & d.ambiguous_unit.eq(0)).sum()),int((~high & amb).sum()),
                  int((high & amb).sum()),int((~high & d.ambiguous_unit.eq(0)).sum()),int(d.ambiguous_unit.isna().sum())]
        self.assertEqual(sum(counts),len(d))
        self.assertEqual(counts,[1,1,1,1,1])

    def test_primary_adjustment_keeps_family_size_four(self):
        p = pd.Series([.001,.02,np.nan])
        adjusted = np.minimum(4*p,1)
        self.assertAlmostEqual(adjusted.iloc[0],.004)
        self.assertAlmostEqual(adjusted.iloc[1],.08)
        self.assertTrue(pd.isna(adjusted.iloc[2]))

    def test_python_cells_compile(self):
        for code in [build.CONFIG,build.PREP,build.RUN_R,build.FINALIZE,build.FIG_CONFIG,build.FIG_PLOTS]:
            ast.parse(code)
        self.assertIn("spec <- read.csv",build.R_CODE)
        self.assertIn("confint(fit",build.R_CODE)
        self.assertNotIn("coef - 1.96",build.R_CODE)

    def test_synthetic_end_to_end_summary_and_bundle(self):
        # Exercise packaging/count checks with synthetic inputs and mock coefficient rows;
        # no R process, real cohort, or regression is executed.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output=Path(directory)
            date=pd.Timestamp('2020-06-01')
            days=np.tile([0,1,30,31,365,366,1825,1826],12)
            n=len(days)
            fake=pd.DataFrame(dict(person_id=[str(i) for i in range(n)],index_date=[str(date.date())]*n,
                crp_date=[str((date-pd.Timedelta(days=int(x))).date()) for x in days],days_before=days,
                crp_mgL=np.tile([2.,12.],n//2),vitd_main=np.tile([10.,25.,40.],n//3),
                age_at_index=[50.]*n,female=np.tile([0,1],n//2),winter=np.tile([0,1],n//2),
                bmi_full=[25.]*n,ambiguous_unit=np.tile([0,0,1],n//3)))
            fake['log_crp']=np.log(fake.crp_mgL)
            fake['log_vitd']=np.log(fake.vitd_main)
            for rs in ['rs731236','rs7975232','rs1544410','rs2228570']:
                fake[rs+'_alt_dosage']=np.tile([0.,1.,2.],n//3)
            source=output/'synthetic_input.csv'
            fake.to_csv(source,index=False)
            ns={'pd':pd,'np':np,'Path':Path,'hashlib':__import__('hashlib'),'zipfile':zipfile,
                'CRP_PATH':source,'OUTPUT_DIR':output,'RUN_ID':'SYNTHETIC',
                'RSIDS':['rs731236','rs7975232','rs1544410','rs2228570'],'WRITTEN':[]}
            functions=[node for node in ast.parse(build.CONFIG).body if isinstance(node,ast.FunctionDef)]
            exec(compile(ast.Module(body=functions,type_ignores=[]),'<synthetic-helpers>','exec'),ns)
            exec(build.PREP,ns)
            flow=pd.DataFrame(ns['flow'])
            status_rows=[]; term_rows=[]
            for row in flow.itertuples(index=False):
                info=ns['spec'].set_index('model_dataset').loc[row.model_dataset]
                status_rows.append(dict(run_id='SYNTHETIC',model_dataset=row.model_dataset,rsid=row.rsid,
                    status='pass',n=row.complete_case_n))
                term_rows.append(dict(run_id='SYNTHETIC',model_dataset=row.model_dataset,rsid=row.rsid,
                    priority=info.priority,term=row.rsid+'_x_log_vitd_additive',interaction=row.rsid+'_x_log_vitd_additive',
                    coef=.02,se=.01,ci_lower_coef=0.,ci_upper_coef=.04,p=.02,n=row.complete_case_n))
            ns['write_csv'](pd.DataFrame(status_rows),'16_crp_sensitivity_model_fit_status.csv')
            ns['write_csv'](pd.DataFrame(term_rows),'16_crp_sensitivity_all_term_results.csv')
            exec(build.FINALIZE,ns)
            self.assertEqual(ns['run_status']['status'],'pass')
            self.assertEqual(len(ns['primary']),4)
            self.assertTrue(ns['primary'].p_bonferroni_4.eq(.08).all())
            with zipfile.ZipFile(output/'16_q3_crp30_return_bundle.zip') as package:
                self.assertNotIn('_server_only_crp_model_input.csv',package.namelist())
                self.assertNotIn('synthetic_input.csv',package.namelist())
                self.assertIn('16_crp_30day_primary_interactions.csv',package.namelist())

    def test_figure_uses_full_ci_range_and_marks_failed_models(self):
        ns = {'pd':pd,'np':np,'plt':plt,'RSIDS':['s1','s2'],'COLORS':['#20746e','#bd5b29'],
              'LABELS':{'crp_30day_primary':'30-day primary','old':'Old sensitivity'}}
        ns['results']=pd.DataFrame([
            dict(model_dataset='crp_30day_primary',rsid='s1',coef=.1,ci_lower_coef=-2.,ci_upper_coef=3.),
            dict(model_dataset='crp_30day_primary',rsid='s2',coef=0.,ci_lower_coef=-1.,ci_upper_coef=1.)])
        ns['fit_status']=pd.DataFrame([dict(model_dataset='old',rsid='s1',status='rank_deficient')])
        captures=[]
        def capture(fig,stem,caption):
            captures.append((fig,caption))
        ns['save_figure']=capture
        tree=ast.parse(build.FIG_CONFIG)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forest')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<synthetic-plot-test>','exec'),ns)
        ns['forest'](['crp_30day_primary','old'],'test','Synthetic test only')
        fig,caption=captures[0]
        low,high=fig.axes[0].get_xlim()
        self.assertLess(low,-2.)
        self.assertGreater(high,3.)
        self.assertIn('rank_deficient',caption)
        buffer=io.BytesIO()
        fig.savefig(buffer,format='png',dpi=100)
        self.assertGreater(buffer.tell(),1000)
        buffer.seek(0)
        pixels=plt.imread(buffer)
        self.assertGreater(float(pixels.std()),.01)
        plt.close(fig)


if __name__=='__main__':
    unittest.main(verbosity=2)
