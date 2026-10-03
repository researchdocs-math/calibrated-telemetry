import numpy as np
from protocol import simulate,link_sequence
def main():
    rng=np.random.default_rng(62817)
    worst=0.
    # Test extreme, bounded, drifting and stochastic paths plus blackout links.
    for seed in range(40):
        n=2001
        y=np.clip(np.cumsum(rng.normal(0,.09,n))+.5,0,1)
        if seed%4==0: y=np.arange(n)%2+0.
        if seed%4==1: y[:]=.5
        for mode in (3,4,5,7):
            for H in (1,8,64):
                s,p,u=link_sequence(n-1,seed%4,seed)
                out,_=simulate(y,s,p,u,mode,.1,H,.003,.1,keep=False)
                assert out[11]<1e-8,(seed,mode,H,out[11])
                assert abs((1-out[0])-.1)<=out[12]+1e-12
                D=out[9]
                assert -.003*.1*D-1e-10<=out[13]<=1+.003*.9*D+1e-10
                worst=max(worst,out[11])
    # Identical event thresholds imply identical point predictions in all modes.
    y=rng.random(1001);s,p,u=link_sequence(1000,2,91)
    outs=[simulate(y,s,p,u,m,.2,12,.001,.3)[0] for m in range(9)]
    assert np.ptp([o[3] for o in outs])<1e-14
    print('PASS: 480 pathwise identity/bound/state checks; shared prediction check; worst residual',worst)
if __name__=='__main__':main()
