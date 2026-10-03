# Cumulative Calibration for Lossy IoT Telemetry

Reproducible experiments for cumulative error telemetry (CET), an acknowledged sensor–gateway protocol that carries cumulative prediction-error counts and triggers repair traffic from outstanding calibration imbalance.

**Author:** Cemil Emre Yavas, School of Computing, Georgia Southern University  
**Contact:** cemil@fairesresearch.org  
**ORCID:** https://orcid.org/0009-0006-9552-1731

## Reproduce the results

Use Python 3.12. The recorded environment is an Apple M3 Pro with 12 CPU cores and 18 GB unified memory. The core experiments run on CPUs; an optional Apple Metal check verifies the bootstrap matrix products on the GPU.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py --workers 12
```

The command runs protocol checks, chronological validation, the 30-seed main evaluation, sensitivity and alternative-predictor experiments, controlled shifts, statistical analysis, and output verification. It overwrites the generated files in results/ with the reproduced outputs. The included processed data support an entirely offline experiment run after dependency installation.

For the optional Metal GPU cross-check:

```sh
python -m pip install -r requirements-gpu.txt
python run_all.py --workers 12 --check-mps
```

To validate the included results immediately:

```sh
python verify_results.py
```

The verifier checks the recorded SHA-256 fingerprints of uncompressed CSV contents, parsed tuning configuration, and numerical arrays. Compression timestamps and elapsed run times are excluded from fingerprints. Exact matching was tested with the pinned environment. Numerical libraries on other architectures can change final floating-point digits; the original environment and baseline summaries are recorded for comparison.

## Contents

- **src/protocol.py:** causal telemetry simulator, loss processes, and initial calibration.
- **src/test_protocol.py:** pathwise identity, bound, state, and shared-predictor checks.
- **src/run_experiments.py:** validation, frozen gain selection, and main evaluation.
- **src/extra_experiments.py:** gain–budget sensitivity and the damped-trend predictor.
- **src/synthetic.py:** controlled distribution shifts and saved traces.
- **src/analyze.py:** equal-stream aggregation, paired bootstrap intervals, and calibration-certificate checks.
- **src/prepare_data.py:** deterministic preprocessing of source datasets.
- **download_data.py:** checksum-verified acquisition and preprocessing of the original datasets.
- **data/streams.npz:** normalized observed readings and original timestamps.
- **data/streams.json:** stream metadata, chronological split boundaries, scale, offset, and training statistics.
- **results/:** individual runs, selected gains, aggregate statistics, paired comparisons, and synthetic traces.
- **config.json:** experiment settings and method identifiers.
- **reproduction_manifest.json:** fingerprints of reproducible numerical outputs.
- **environment.json:** recorded software and hardware environment.
- **DATA_SOURCES.md:** dataset provenance, preprocessing, attribution, and permissions.

The large individual-run tables use CSV with gzip compression. Pandas reads them directly with pd.read_csv("results/test.csv.gz").

## Experimental scope

There are 12 Beijing PM2.5 streams and 52 Intel temperature streams, totaling 724,351 eligible observations. Each stream is split in chronological order into 50% training, 20% validation, and 30% testing. Missing observations are excluded without interpolation. The simulator advances on eligible observed-reading rounds; it does not convert source-data gaps into invented observations.

The main evaluation contains 552,960 runs and 1,877,774,400 prediction evaluations over 217,335 distinct test readings. Validation contains 112,896 runs. The additional sensitivity and predictor evaluation contains 24,960 runs, and the controlled-shift evaluation contains 540 runs. All configurations, including weaker-performing cases, are retained.

## Methods

| ID | Name | Calibration feedback |
|---:|---|---|
| 0 | Frozen | Initial radius retained |
| 1 | Received only | Current error on successful measurement events |
| 2 | I-ACI | Inverse-probability update with random audits |
| 3 | Periodic counts | Cumulative counts after H readings, with retries |
| 4 | CET | Counts on measurement packets and debt-triggered repair |
| 5 | Full feedback | Every error through an ideal, unpriced side channel |
| 6 | Last-only repair | Latest error multiplied by elapsed block length |
| 7 | Timer CET | Counts on measurement packets and timer-triggered repair |
| 8 | IM-OCP | Uniform-prior mirror update with inverse-probability audits |

Names in protocol.py are short internal labels; numeric IDs define the mapping. The full-feedback method is a calibration reference: its data-link traffic excludes the ideal feedback channel.

The target coverage is 90%. A data event occurs when absolute reconstruction error exceeds h. The primary predictor holds the last delivered reading. Only successful measurement packets change its state. Counter-only feedback changes the calibration radius without changing the predictor.

CET latches repair at absolute outstanding error debt b = 0.1 H. It attempts a packet every subsequent observed-reading round until success. Each attempt carries the current counter. A simultaneous measurement event and repair use one combined packet. A reliable acknowledgement commits the same state at sensor and gateway.

The counter update is applied after scoring the current interval. It is not projected onto a nonnegative domain: negative radii yield empty sets, while intervals spanning the bounded domain yield full sets. Both frequencies are included in results.

## Channels and byte accounting

Channel IDs are 0: independent 10% loss; 1: independent 30% loss; 2: a two-state burst process; 3: periodic complete outages.

The burst process uses good-to-bad probability 0.02, bad-to-good probability 0.08, and success probabilities 0.95 and 0.10. Its state evolves even during silence. The outage process blocks zero-based slots 960–1087 of each 2048-round cycle and uses independent 10% loss elsewhere.

The nominal header is 20 bytes, including sequence information. A reading uses 8 bytes, a cumulative counter or aligned current-error field uses 4, and a successful acknowledgement uses 8. Failed attempts incur their full uplink cost. Channels and packets are simulated; these are explicit application-level byte assumptions.

## Initialization, tuning, and seeds

For each stream, s0 is the training 90th percentile of absolute one-step changes, floored at 1e-5. Measurement thresholds are h/s0 = 0.5, 1, 2, 4. Initial radii use a causal replay of the final 20% of the training portion under the corresponding threshold and channel.

Validation compares gain multipliers eta/s0 = 0.003, 0.01, 0.03, 0.1, 0.3, 1 for every dataset, method, and H in {8, 32}, using thresholds {1, 4}, channels {0, 1, 2}, and three seeds. Among gains with mean absolute coverage gap at most 0.02, the minimum normalized quantile loss is selected; otherwise the smallest gap is selected, breaking ties by loss. Selections are frozen before testing.

For stream index i starting at zero, test channel seeds are seed + 100003(i+1). Validation adds 7,000,000. Initialization uses 910117+i. Stream order is stored in data/streams.json. Main test seed indices are 0–29. Extra experiments use indices 0–9. Synthetic observations use seed 109912; their channels use 22001+seed. Bootstrap resampling uses seed 774109. Main-method comparisons share channel realizations within each stream and seed.

## Metrics and outputs

- **coverage:** fraction of issued intervals containing the target.
- **width:** mean interval width in normalized units. Multiply by the stream scale (1000 for Beijing, 80 for Intel) for physical units.
- **quantile_loss:** pinball loss at quantile level 0.9.
- **bytes:** attempted uplink bytes plus successful acknowledgements, per reading.
- **window_error:** mean absolute miscoverage deviation in complete, nonoverlapping 256-reading windows.
- **empty_rate / full_rate:** frequencies of empty sets and full-domain intervals.
- **identity_error:** absolute residual of the cumulative-feedback accounting identity.
- **bound:** the pathwise time-average miscoverage bound.
- **max_debt / terminal_debt:** maximum completed-block debt magnitude and outstanding terminal debt.
- **max_failure_run:** longest run of failed attempts, counting consecutive attempts even across silence.
- **repair_success_rate:** successful counter-only packets per reading.

Summary rows average streams equally within each dataset and then average channel seeds. Absolute gaps are calculated per stream before aggregation. Confidence intervals use 10,000 paired bootstrap resamples of the 30 seed indices. They describe channel-randomization uncertainty conditional on these traces.

At burst losses, h/s0 = 2 and H = 32, CET reaches 90.05% coverage on Beijing and 88.42% on Intel. Relative to periodic cumulative feedback, Beijing's mean 256-reading error is 30.7% lower and its traffic is 6.4% lower. Against Timer CET, the Beijing improvement costs additional width and a small increase in bytes; Intel shows essentially equal local error. The complete result tables support examination of these trade-offs.

## Individual stages

```sh
python src/test_protocol.py
python src/run_experiments.py --phase all --workers 12 --seeds 30
python src/extra_experiments.py --workers 12
python src/synthetic.py
python src/analyze.py
python verify_results.py
```

To reconstruct the processed arrays from the original sources:

```sh
python download_data.py
python verify_results.py
```

Downloaded raw files are saved in data/raw/. Dataset descriptions and attribution are in DATA_SOURCES.md.

## License and citation

Original experiment code is provided under the MIT license in LICENSE. Dataset permissions and attribution are recorded separately in DATA_SOURCES.md. CITATION.cff supplies the author, title, version, and release date for citing this research software.
