# RT-IoT2022 Network Traffic Classification

## Objective

Given bidirectional network flow features captured from a real IoT infrastructure, classify each flow into one of 12 traffic categories. Some are normal device communication, others are specific attack patterns.

The 12 classes:

| Label | What it is |
|-------|-----------|
| `DOS_SYN_Hping` | SYN flood DoS via hping3. By far the most common attack in this dataset. |
| `ARP_poisioning` | ARP spoofing/cache poisoning |
| `NMAP_UDP_SCAN` | Nmap UDP port scan |
| `NMAP_XMAS_TREE_SCAN` | Nmap Xmas tree scan (FIN+PSH+URG flags) |
| `NMAP_OS_DETECTION` | Nmap OS fingerprinting probes |
| `NMAP_TCP_scan` | Nmap TCP connect/SYN scan |
| `DDOS_Slowloris` | Slowloris DDoS (holds connections open with partial headers) |
| `Metasploit_Brute_Force_SSH` | SSH brute force via Metasploit (very rare, only ~37 samples) |
| `NMAP_FIN_SCAN` | Nmap FIN scan (even rarer, ~28 samples) |
| `Thing_Speak` | Normal ThingSpeak IoT platform traffic |
| `MQTT_Publish` | Normal MQTT publish messages |
| `Wipro_bulb` | Normal Wipro smart bulb traffic |

The class distribution is skewed. The original dataset had DOS_SYN_Hping at 77%, but the large classes were downsampled to keep files manageable. DOS_SYN_Hping is still the biggest at about 29% of training data, and two classes (SSH brute force, FIN scan) have fewer than 40 samples total. A model that just predicts the majority class everywhere will score terribly on macro F1.

## Inputs

| File | Contents |
|------|----------|
| `train.csv` | 21,963 network flows with all features + `Attack_type` label |
| `test.csv`  | 5,491 flows, same features but no `Attack_type` |

### Features

83 features, mostly numeric. Two categorical (`proto`, `service`). All derived from bidirectional flow analysis using Zeek + Flowmeter.

**Connection identifiers:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID |
| `src_port` | int | Source port |
| `dst_port` | int | Destination port |
| `proto` | string | Protocol: tcp, udp, or icmp |
| `service` | string | Detected application protocol (mqtt, dns, http, ssl, ssh, etc). "-" means none detected |

**Flow-level stats:**

| Column | Type | Description |
|--------|------|-------------|
| `flow_duration` | float | Total flow duration in microseconds |
| `fwd_pkts_tot` | int | Packets sent forward |
| `bwd_pkts_tot` | int | Packets sent backward |
| `fwd_data_pkts_tot` | int | Forward data packets (excludes control) |
| `bwd_data_pkts_tot` | int | Backward data packets |
| `fwd_pkts_per_sec` | float | Forward packet rate |
| `bwd_pkts_per_sec` | float | Backward packet rate |
| `flow_pkts_per_sec` | float | Total packet rate |
| `down_up_ratio` | float | Download/upload ratio |

**Header sizes:**

| Column | Type | Description |
|--------|------|-------------|
| `fwd_header_size_tot` | int | Total forward header bytes |
| `fwd_header_size_min` | int | Min forward header size |
| `fwd_header_size_max` | int | Max forward header size |
| `bwd_header_size_tot` | int | Total backward header bytes |
| `bwd_header_size_min` | int | Min backward header size |
| `bwd_header_size_max` | int | Max backward header size |

**TCP flags (counts per flow):**

| Column | Type | Description |
|--------|------|-------------|
| `flow_FIN_flag_count` | int | FIN flags |
| `flow_SYN_flag_count` | int | SYN flags |
| `flow_RST_flag_count` | int | RST flags |
| `fwd_PSH_flag_count` | int | Forward PSH flags |
| `bwd_PSH_flag_count` | int | Backward PSH flags |
| `flow_ACK_flag_count` | int | ACK flags |
| `fwd_URG_flag_count` | int | Forward URG flags |
| `bwd_URG_flag_count` | int | Backward URG flags |
| `flow_CWR_flag_count` | int | CWR flags |
| `flow_ECE_flag_count` | int | ECE flags |

**Payload statistics (bytes):**

Forward, backward, and flow-level payload stats. Each direction has min, max, total, average, and standard deviation columns. 15 columns total following the naming pattern `{direction}_pkts_payload_{stat}`, e.g. `fwd_pkts_payload_min`, `bwd_pkts_payload_avg`, `flow_pkts_payload_std`.

**Inter-arrival time (IAT) statistics (microseconds):**

Same structure as payload stats but for time gaps between packets. 15 columns following `{direction}_iat_{stat}`, e.g. `fwd_iat_min`, `bwd_iat_max`, `flow_iat_std`.

**Throughput:**

| Column | Type | Description |
|--------|------|-------------|
| `payload_bytes_per_second` | float | Payload throughput |

**Subflow and bulk transfer stats:**

| Column | Type | Description |
|--------|------|-------------|
| `fwd_subflow_pkts` | float | Forward subflow packet count |
| `bwd_subflow_pkts` | float | Backward subflow packet count |
| `fwd_subflow_bytes` | float | Forward subflow bytes |
| `bwd_subflow_bytes` | float | Backward subflow bytes |
| `fwd_bulk_bytes` | float | Forward bulk transfer bytes |
| `bwd_bulk_bytes` | float | Backward bulk transfer bytes |
| `fwd_bulk_packets` | float | Forward bulk packets |
| `bwd_bulk_packets` | float | Backward bulk packets |
| `fwd_bulk_rate` | float | Forward bulk rate |
| `bwd_bulk_rate` | float | Backward bulk rate |

**Connection activity timing:**

| Column | Type | Description |
|--------|------|-------------|
| `active_min` | float | Min active period |
| `active_max` | float | Max active period |
| `active_tot` | float | Total active time |
| `active_avg` | float | Average active period |
| `active_std` | float | Active period std dev |
| `idle_min` | float | Min idle period |
| `idle_max` | float | Max idle period |
| `idle_tot` | float | Total idle time |
| `idle_avg` | float | Average idle period |
| `idle_std` | float | Idle period std dev |

**TCP window sizes:**

| Column | Type | Description |
|--------|------|-------------|
| `fwd_init_window_size` | int | Initial forward TCP window |
| `bwd_init_window_size` | int | Initial backward TCP window |
| `fwd_last_window_size` | int | Last forward TCP window |

## Output

Produce a `submission.csv` with 5,491 rows (plus header).

```
id,Attack_type
3,DOS_SYN_Hping
7,MQTT_Publish
11,NMAP_UDP_SCAN
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match IDs in `test.csv` |
| `Attack_type` | string | One of the 12 valid class labels, spelled exactly as shown above |

## Metric

Macro F1 score. Higher is better.

Macro F1 computes F1 for each class independently, then takes the unweighted average. This means every class counts equally regardless of how many samples it has. A model that nails the big classes but ignores the rare ones will score poorly.

For reference, predicting `DOS_SYN_Hping` for everything gives a macro F1 of about 0.04. That's the floor (only 1 of 12 classes gets any F1 credit). A decent model should get above 0.85, and a well-tuned one should push past 0.95.

Score locally with:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Constraints

- `id` is a row identifier, not a feature.
- Class labels must be spelled exactly as they appear in the training data (case-sensitive).
- `src_port` and `dst_port` are network ports, not features you'd use directly in most models (they're noisy identifiers, not stable predictors). But your call.
- The dataset has no missing values.
