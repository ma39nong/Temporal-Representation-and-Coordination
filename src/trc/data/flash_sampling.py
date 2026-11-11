"""Small FLASH-specific index over already proven continuous segments."""
import numpy as np


class ContinuousFlashIndex:
    def __init__(self,records,cut_manifest,seed):
        by_id={r['id']:r for r in cut_manifest['records']}
        if len(by_id)!=len(cut_manifest['records']) or len(by_id)!=len(records) or set(by_id)!={r['id'] for r in records}:
            raise ValueError('Continuous proof must cover every training source exactly')
        self.ranges=[];self.ends=[];total=0
        for r in records:
            proof=by_id[r['id']]
            if proof['rows']!=r['rows'] or proof['source_recording_id']!=r['source_recording_id']:
                raise ValueError('Source proof differs')
            cuts=proof['cuts'];n=r['rows']
            if cuts!=sorted(set(cuts)) or any(not isinstance(c,int) or not 0<c<n for c in cuts):
                raise ValueError('Sorted internal integer cuts required')
            bounds=[0,*cuts,n]
            count=0
            for left,right in zip(bounds[:-1],bounds[1:]):
                size=max(0,right-left-58)
                if size:
                    self.ranges.append((r['id'],left+7,size));total+=size;self.ends.append(total);count+=size
            if count!=proof['legal_windows']:raise ValueError('Legal window count differs')
        if not total or total!=cut_manifest['legal_windows']:raise ValueError('Empty/inconsistent sampler')
        self.total=total;self.seed=seed;self.epoch=None;self.permutation=None

    def index(self,position):
        if isinstance(position,(bool,np.bool_)) or not isinstance(position,(int,np.integer)) or position<0:raise ValueError('Nonnegative integer sample index required')
        epoch,offset=divmod(int(position),self.total)
        if epoch!=self.epoch:
            self.permutation=np.random.default_rng(np.random.SeedSequence([self.seed,epoch])).permutation(self.total)
            self.epoch=epoch
        value=int(self.permutation[offset]);i=int(np.searchsorted(self.ends,value,side='right'))
        ident,first,_=self.ranges[i];start=first+value-(self.ends[i-1] if i else 0)
        return {'id':ident,'start':start}
