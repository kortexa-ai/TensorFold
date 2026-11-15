"""Bounded replacement/equality and cold-buffer timing panel for Inkling decode attention."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
from inkling2_probe import Attention, Driver


def guarded(shape):
    count = 1
    for n in shape:
        count *= n
    storage = torch.full((count + 256,), 0x5A5A, dtype=torch.int16, device='cuda')
    return storage, storage[128:-128].view(torch.bfloat16).reshape(shape)


def fixture(length, window, seed, plant=None, heads=24):
    torch.manual_seed(seed)
    pos, capacity = length-1, min(length, 1280) if window else length
    shapes = ((1,heads,128),(capacity,heads//6,128),(capacity,heads//6,128),
              (1,heads,1024),(1,heads,128),(1,heads))
    buffers = [guarded(s) for s in shapes]
    for i, (_, x) in enumerate(buffers):
        x.copy_(torch.randn(x.shape, device='cuda') * (.3 if i < 4 else 1))
    tensors = [x for _, x in buffers]
    q,k,v,relative,gate,sink = tensors
    first = max(0,pos-window) if window else 0
    if plant == 'q-nan':
        q[0,0,0] = torch.nan
    elif plant == 'k-inf':
        k[first%capacity,0,0] = torch.inf
    elif plant == 'v-inf':
        v[first%capacity,0,0] = torch.inf
    elif plant == 'v-both-inf':
        v[first%capacity,0,0],v[(first+1)%capacity,0,0] = torch.inf,-torch.inf
    elif plant == 'relative-nan':
        relative[0,0,0] = torch.nan
    elif plant == 'gate-nan':
        gate[0,0,0] = torch.nan
    elif plant == 'sink-inf':
        sink[0,0] = torch.inf
    elif plant == 'q-subnormal':
        q[0,0,0] = 1e-40
    elif plant == 'hidden-v-nan':
        if not window or first == 0 or first % 64 == 0:
            raise ValueError('The hidden-value fixture needs a masked key in the first tile')
        v[(first-1)%capacity,0,0] = torch.nan
    output, out = guarded((1,heads,128))
    args = Attention(*(x.data_ptr() for x in (*tensors,out)),1,pos,heads,heads//6,capacity,window)
    return buffers, output, out, args


def equal_bytes(x, y):
    return torch.equal(x.view(torch.uint8),y.view(torch.uint8))


def check_guards(storage):
    return bool((storage[:128] == 0x5A5A).all() and (storage[-128:] == 0x5A5A).all())


def reference(buffers, args, want):
    q,k,v,relative,gate,sink = [x.double() for _,x in buffers]
    first = max(0,args.pos-args.window) if args.window else 0
    keys = torch.arange(first,args.pos+1,device='cuda')
    scores = torch.einsum('hd,khd->hk',q[0],k[keys%args.capacity].repeat_interleave(6,1))*128**-.5
    distance = args.pos-keys
    scores += torch.where(distance[None]<1024,relative[0,:,distance.clamp_max(1023)],0)
    raw = torch.einsum('hk,khd->hd',scores.softmax(-1),v[keys%args.capacity].repeat_interleave(6,1))
    expected = raw*gate[0].sigmoid()*(scores.logsumexp(-1)-sink[0]).sigmoid()[:,None]
    error = (want[0].double()-expected).abs()
    envelope = expected.abs()*2**-8+2e-6
    if (error > envelope).any():
        raise AssertionError('Original attention exceeds its existing FP64 rounding envelope')
    return {'max_abs_error':error.max().item(),'outside_existing_envelope':0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--symbols',default='tf_ik_attention_decode')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--canary',action='store_true')
    a = p.parse_args()
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0) not in ((12,0),(12,1)):
        raise RuntimeError('Pin one authorized sm_120/sm_121 GPU')
    torch.cuda.set_per_process_memory_fraction(.10)
    if torch.cuda.mem_get_info()[0] < 8*1024**3:
        raise RuntimeError('Leave at least 8 GiB free before this bounded probe')
    torch.empty(1,device='cuda')
    baseline,candidate = Driver(a.baseline),Driver(a.candidate)
    symbols = a.symbols.split(',')
    report = {'status':'running','gpu':torch.cuda.get_device_name(0),'compute_capability':torch.cuda.get_device_capability(0),
              'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
              'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
              'timing_method':'CUDA events, 256 MiB L2 scrub before each timed kernel, median of seven', 'cases':[]}
    scrub = torch.empty(256*1024**2,device='cuda',dtype=torch.uint8)
    cases = [(n,w,None) for n in ((65,) if a.canary else (1,63,64,65,1023,1024,1025,8192,32768,65536))
             for w in (0,1023)]
    if not a.canary:
        cases += [(2050,1023,'hidden-v-nan')]
        cases += [(65,w,plant) for w in (0,1023) for plant in
                  ('q-nan','k-inf','v-inf','v-both-inf','relative-nan','gate-nan','sink-inf','q-subnormal')]
    try:
        for i,(length,window,plant) in enumerate(cases):
            buffers,storage,out,args = fixture(length,window,20261114+i,plant)
            before = [s.clone() for s,_ in buffers]
            baseline.launch('tf_ik_attention',(1,args.hq),128,args)
            torch.cuda.synchronize()
            want = out.clone()
            item = {'length':length,'window':window,'plant':plant,'symbols':{}}
            if plant is None and length in (1,65,1025):
                item['fp64_reference'] = reference(buffers,args,want)
            for symbol in symbols:
                splits = 2 if symbol.endswith('2') else 4
                for _ in range(5):
                    out.fill_(torch.nan)
                    candidate.launch(symbol,(1,args.hq,splits),128,args)
                    torch.cuda.synchronize()
                    if not equal_bytes(out,want) or not check_guards(storage):
                        raise AssertionError(f'Output bytes or guard changed: {symbol}, {length}, {window}, {plant}')
                    if any(not equal_bytes(s,b) or not check_guards(s) for (s,_),b in zip(buffers,before)):
                        raise AssertionError('An input or its guard changed')
                item['symbols'][symbol] = {'different_bytes':0,'identical_repeats':5}
            if plant is None and length>=65:
                for symbol,driver,splits in [('tf_ik_attention',baseline,1)]+[(s,candidate,2 if s.endswith('2') else 4) for s in symbols]:
                    times=[]
                    for _ in range(9):
                        scrub.zero_()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        driver.launch(symbol,(1,args.hq,splits),128,args)
                        end.record(); end.synchronize()
                        times.append(start.elapsed_time(end))
                    item['symbols'].setdefault(symbol,{})['median_kernel_ms']=statistics.median(times[2:])
            report['cases'].append(item)
            print(json.dumps(item),flush=True)
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
