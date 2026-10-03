"""Causal scalar telemetry simulator. Modes and accounting are documented in README."""
import numpy as np
from numba import njit
MODES = ["Frozen", "Received", "IPW", "Periodic", "Cumulative", "Full feedback", "Last only", "Timer cumulative", "Mirror IPW"]
METRICS = ["coverage","width","quantile_loss","mae","bytes","attempt_rate","feedback_rate","empty_rate","full_rate","max_gap","terminal_gap","identity_error","bound","q_end","q0","eta","window_error","max_debt","terminal_debt","repair_success_rate","max_failure_run"]
@njit(cache=True)
def simulate(y, success, propensity, audit_u, mode, h, H, eta, q0, alpha=.1, header=20, keep=False, predictor=0):
    # y[0] is one common initial delivered reading; evaluation starts at y[1].
    n=len(y)-1
    q=q0; pred=y[0]
    last_value=y[0]; slope=0.; last_data=0
    nerr=0; acknowledged_errors=0; last_feedback=0
    widths=0.; scores=0.; mae=0.; byte_count=0.
    attempts=0; deliveries=0; empty=0; full=0; maxgap=0
    window_sum=0.; window_count=0; window_err=0
    latched=False; maxdebt=0.
    repair_success=0; failure_run=0; maxfailure=0
    trace=np.zeros((n,8)) if keep else np.empty((0,8))
    for j in range(n):
        t=j+1; obs=y[t]
        if predictor==1: pred=min(1.,max(0.,last_value+slope*(1.-.9**(t-last_data))/.1))
        s=abs(obs-pred)
        e=1 if s>q else 0
        nerr+=e; mae+=s
        if q<0:
            lo=pred; hi=pred; width=0.
            empty+=1
        else:
            lo=max(0.,pred-q); hi=min(1.,pred+q); width=hi-lo
        if q>=0 and lo==0. and hi==1.: full+=1
        # Pinball loss remains defined for empty/full prediction sets.
        score=(1.-alpha)*(s-q) if s>=q else alpha*(q-s)
        widths+=width; scores+=score
        window_err+=e
        if t%256==0:
            window_sum+=abs(window_err/256.-alpha)
            window_count+=1; window_err=0
        event=s>h
        debt=(nerr-acknowledged_errors)-alpha*(t-last_feedback)
        if mode==4 and abs(debt)>=alpha*H: latched=True
        due=latched if mode==4 else t-last_feedback>=H
        audit=audit_u[j]<1./H
        want_feedback=(mode==4 or mode==6 or mode==7) or (mode==3 and due)
        attempted=event or ((mode==4 or mode==6 or mode==3 or mode==7) and due) or ((mode==2 or mode==8) and audit)
        feedback=False
        if attempted:
            attempts+=1
            # Header 20B includes sequence and address; data 8B, counter 4B;
            # one-bit feedback encoded in 4B field for aligned packet comparison.
            byte_count+=header+(8 if event else 0)+(4 if (want_feedback or mode==2 or mode==8 or mode==6) else 0)
            if success[j]:
                failure_run=0
                if not event: repair_success+=1
                byte_count+=8 # acknowledged link; ACK loss excluded by model
                if event:
                    slope=(obs-last_value)/(t-last_data);last_value=obs;last_data=t;pred=obs
                if mode==1:
                    q+=eta*(e-alpha); feedback=True
                elif mode==2 or mode==8:
                    pi=propensity[j]*(1. if event else 1./H)
                    if mode==2: q+=eta*(e-alpha)/pi
                    else:
                        # IM-OCP with uniform prior [0,a] and sigma=1/q0.
                        a=min(1.,q0/(1.-alpha)); sigma=1./q0
                        cdf=min(1.,max(0.,q/a))
                        z=cdf-(1.-alpha)+sigma*q+(eta/q0)*(e-alpha)/pi
                        if z<-(1.-alpha): q=(z+(1.-alpha))/sigma
                        elif z<=alpha+sigma*a: q=(z+(1.-alpha))/(1./a+sigma)
                        else: q=(z-alpha)/sigma
                    feedback=True
                elif mode==3 or mode==4 or mode==7:
                    if want_feedback:
                        q+=eta*((nerr-acknowledged_errors)-alpha*(t-last_feedback))
                        maxdebt=max(maxdebt,abs(debt))
                        acknowledged_errors=nerr; feedback=True; latched=False
                elif mode==6:
                    q+=eta*(t-last_feedback)*(e-alpha); feedback=True
        if attempted and not success[j]:
            failure_run+=1; maxfailure=max(maxfailure,failure_run)
        if mode==5:
            q+=eta*(e-alpha); maxdebt=max(maxdebt,abs(e-alpha)); acknowledged_errors=nerr; feedback=True
        if feedback:
            deliveries+=1
            maxgap=max(maxgap,t-last_feedback)
            last_feedback=t
        if keep:
            trace[j,0]=obs;trace[j,1]=lo;trace[j,2]=hi;trace[j,3]=e
            trace[j,4]=attempted;trace[j,5]=feedback;trace[j,6]=q;trace[j,7]=byte_count
    gap=n-last_feedback
    identity=abs((nerr-alpha*n)-((q-q0)/eta+(nerr-acknowledged_errors)-alpha*gap)) if mode in (3,4,5,7) else np.nan
    terminal_debt=(nerr-acknowledged_errors)-alpha*gap
    bound=(max(q0,1.-q0)/eta+maxdebt+abs(terminal_debt))/n
    result=np.array([1-nerr/n,widths/n,scores/n,mae/n,byte_count/n,attempts/n,deliveries/n,empty/n,full/n,maxgap,gap,identity,bound,q,q0,eta,window_sum/max(1,window_count),maxdebt,terminal_debt,repair_success/n,maxfailure])
    return result,trace

@njit(cache=True)
def link_sequence(n, kind, seed):
    """Common random channel states; states evolve on every observed-reading slot."""
    np.random.seed(seed)
    success=np.empty(n,np.bool_); p=np.empty(n)
    audit=np.random.random(n)
    bad=False
    for t in range(n):
        if kind==0: pt=.9
        elif kind==1: pt=.7
        elif kind==2:
            # P(G->B)=.02, P(B->G)=.08; stationary success .8*.95+.2*.10=.78.
            if bad:
                if np.random.random()<.08: bad=False
            else:
                if np.random.random()<.02: bad=True
            pt=.1 if bad else .95
        elif kind==3:
            # Source-independent outage of 128 reading slots in every 2048.
            pt=0. if (t%2048)>=960 and (t%2048)<1088 else .9
        else: pt=1.
        p[t]=pt; success[t]=np.random.random()<pt
    return success,p,audit

@njit(cache=True)
def initial_radius(y, success, h, alpha=.1, predictor=0):
    """Offline calibration of the identical causal event-driven predictor."""
    scores=np.empty(len(y)-1)
    pred=y[0];last_value=y[0];slope=0.;last_data=0
    for j in range(len(scores)):
        if predictor==1: pred=min(1.,max(0.,last_value+slope*(1.-.9**(j+1-last_data))/.1))
        scores[j]=abs(y[j+1]-pred)
        if scores[j]>h and success[j]:
            slope=(y[j+1]-last_value)/(j+1-last_data);last_value=y[j+1];last_data=j+1;pred=y[j+1]
    return max(1e-5,np.quantile(scores,1.-alpha))
