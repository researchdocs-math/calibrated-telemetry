# Dataset provenance and attribution

## Beijing Multi-Site Air Quality

Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data  
Persistent identifier: https://doi.org/10.24432/C5RK5G  
Dataset creator: Song Chen. Citation: Chen, S. (2017). Beijing Multi-Site Air Quality. UCI Machine Learning Repository.  
License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

The source dataset contains hourly observations from 12 monitoring sites between March 2013 and February 2017. This package uses finite PM2.5 observations in [0,1000] micrograms per cubic meter, divided by 1000. Missing and out-of-domain values are excluded. All 12 sites are retained. Each stream is split chronologically into 50% training, 20% validation, and 30% testing.

The included arrays are processed versions of the source measurements. Original timestamps are retained. The use and processing of these data do not imply endorsement by their creators.

Related source study: S. Zhang, B. Guo, A. Dong, J. He, Z. Xu, and S. X. Chen, "Cautionary tales on air-quality improvement in Beijing," Proceedings of the Royal Society A, vol. 473, no. 2205, article 20170457, 2017. https://doi.org/10.1098/rspa.2017.0457

Download: https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip  
SHA-256: b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4

## Intel Berkeley Research Lab

Source: https://db.csail.mit.edu/labdata/labdata.html  
Dataset contributors: Peter Bodik, Wei Hong, Carlos Guestrin, Sam Madden, Mark Paskin, and Romain Thibaux (2004).

The source website permits use for any purpose and reproduction with acknowledgement of the original source. The original data and the processed arrays are attributed to the contributors and Intel Berkeley Research Lab.

This package retains valid timestamps, mote identifiers 1–54, and temperature measurements in [-20,60] degrees Celsius. Within each mote, temperatures are aggregated by five-minute medians. Empty bins are excluded and motes with at least 2000 nonempty bins are retained, yielding 52 streams. Values are normalized as (temperature + 20)/80. Chronological 50/20/30 splits are recorded per stream.

The original source recorded readings using TinyDB. The associated system study is S. R. Madden, M. J. Franklin, J. M. Hellerstein, and W. Hong, "TinyDB: an acquisitional query processing system for sensor networks," ACM Transactions on Database Systems, vol. 30, no. 1, pp. 122–173, 2005. https://doi.org/10.1145/1061318.1061322

Download: https://db.csail.mit.edu/labdata/data.txt.gz  
SHA-256: d99288c8f406ca6604d359ceaa0d8adfffa79e7095061a1e27dc4399f48c7225

## Processed files

data/streams.npz stores each reading array under its stream key and timestamps under key__time. Timestamps are Unix seconds representing the source's local clock labels; no time-zone conversion is applied. data/streams.json describes the split boundaries, physical scale and offset, original gaps, and training scale s0.

The simulation advances in observed-reading order, preserving chronological ordering. Neither source's missing targets are interpolated. Original software in this package is licensed under MIT; the datasets retain the permissions described above.
