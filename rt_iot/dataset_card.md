# RT-IoT2022 Network Traffic Dataset

## Overview

Bidirectional network flow records captured from a real IoT testbed at the National Institute of Engineering, Mysuru (India). The setup had a mix of commodity IoT devices -- ThingSpeak-connected LEDs, Wipro smart bulbs, MQTT temperature sensors -- running alongside simulated attack traffic. Each row is one network flow, characterized by 83 features extracted by the Zeek network monitor and CICFlowMeter plugin.

The attacks cover a range of common IoT threats: SYN floods (hping3), ARP spoofing, various Nmap scan techniques (UDP, TCP, Xmas tree, FIN, OS detection), Slowloris DDoS, and Metasploit SSH brute force. Normal traffic comes from the three IoT devices doing their regular thing.

The original dataset is extremely lopsided -- SYN flood traffic alone accounts for about 77% of all 123K flows. For this task the large classes were downsampled to keep the file size manageable (about 27K flows total), but the class imbalance is still significant. DOS_SYN_Hping is around 29% of the data, and two attack types (SSH brute force and FIN scan) have fewer than 40 samples combined. This is realistic for network security data -- the dangerous subtle stuff is always buried in the noise.

Domain: IoT network security / intrusion detection systems.

## Source

B. S. Sharmila, Rohini Nagapadma (2023). "RT-IoT2022." UCI Machine Learning Repository.

Original data: https://archive.ics.uci.edu/dataset/942/rt-iot2022

Published alongside the paper: "Quantized autoencoder (QAE) intrusion detection system for anomaly detection in resource-constrained IoT devices using RT-IoT2022 dataset" (Cybersecurity, 2023).

## License

CC-BY-4.0

Full license text: https://creativecommons.org/licenses/by/4.0/

Original dataset by B. S. Sharmila and Rohini Nagapadma, The National Institute of Engineering, Mysuru. Published at UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/942/rt-iot2022), DOI: 10.24432/C5P338.

This derivative version includes the following modifications from the original, per CC-BY-4.0 Section 3(b):

- Dropped the original `id` column and added a new sequential `id`.
- Renamed `id.orig_p` to `src_port` and `id.resp_p` to `dst_port` since dots in column names cause headaches with some tools.
- Replaced all remaining dots in column names with underscores (e.g. `fwd_pkts_payload.min` became `fwd_pkts_payload_min`).
- Downsampled the larger classes (DOS_SYN_Hping capped at 8,000, Thing_Speak and ARP_poisioning at 4,000, MQTT_Publish at 3,000) to reduce file size. Rare classes kept all samples.
- Did an 80/20 stratified split on `Attack_type`. Even the tiny classes (37 SSH brute force, 28 FIN scan) got proportional representation in both splits.
- No rows with missing values existed in the original; none were dropped for that reason.

## Features

83 features total. 81 numeric, 2 categorical (`proto`, `service`).

The features fall into a few groups:

**Connection metadata** (5 cols): source/destination ports, protocol, detected service, and flow duration. The `service` column is "-" for most flows because Zeek couldn't identify the application protocol -- this is normal for attack traffic and some IoT protocols.

**Packet counts and rates** (9 cols): forward/backward/total packet counts, rates, and the download-to-upload ratio. SYN floods have very high forward packet counts with almost nothing coming back.

**Header sizes** (6 cols): min/max/total header sizes for each direction. Useful for spotting scan traffic where packets are minimal.

**TCP flag counts** (10 cols): counts of each TCP flag type within the flow. These are some of the most discriminative features. Xmas tree scans set FIN+PSH+URG, SYN floods hit SYN hard, etc. Each scan type has a distinctive flag signature.

**Payload statistics** (15 cols): min/max/total/mean/std of payload bytes per packet, for forward, backward, and combined directions. Attack traffic typically has very different payload patterns from legitimate IoT communication.

**Inter-arrival times (IAT)** (15 cols): packet timing statistics. Automated attacks like SYN floods have very regular, fast IATs. Normal device communication is more bursty.

**Throughput** (1 col): `payload_bytes_per_second`.

**Subflow and bulk stats** (10 cols): packet/byte counts for subflows and bulk transfers.

**Activity timing** (10 cols): statistics on active and idle periods during the flow.

**TCP window sizes** (3 cols): initial and last window sizes. Different OS stacks and attack tools use different default window sizes, so these act as a soft fingerprint.

## Splitting & Leakage

Stratified random split (80/20) on the `Attack_type` column. No time-based or group-based ordering.

Leakage notes:

Some features are very close to directly encoding the attack type. The TCP flag counts, for example, nearly uniquely identify certain scans (Xmas tree = high URG+PSH+FIN, FIN scan = FIN only). This isn't really "leakage" since flag information is available in real-time network monitoring too, but it does mean a simple decision tree on flag features alone gets surprisingly far.

The `service` column correlates strongly with normal traffic classes (MQTT traffic gets detected as service="mqtt", ThingSpeak as service="http"). If you're building a model for real deployment, relying on this is risky because attackers can mimic service signatures, but for this dataset the correlation is real.

Port numbers (`src_port`, `dst_port`) are noisy but not useless. MQTT traffic uses port 1883, DNS uses 53, etc. These are useful for some classes but won't help distinguish between, say, different Nmap scan types.

The dataset doesn't include IP addresses or timestamps, so there's no risk of memorizing specific hosts or time windows.
