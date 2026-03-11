# How I'd work through this

## 1. Get oriented with the data

About 22K training rows, 83 features, 12 classes. The large classes were downsampled from the original 123K-row dataset to keep file sizes reasonable, but class imbalance is still the main challenge. DOS_SYN_Hping is about 29% of training data, and two classes have fewer than 40 samples. That's going to drive almost every decision downstream.

What I'd check in an initial pass:
- `df.isna().sum()` -- the original dataset claims zero missing values, verify that survived the split.
- Look at `proto` and `service` value counts. `proto` has 3 values (tcp/udp/icmp) and `service` has 10, but most flows are service="-" (unknown). Check if some classes map almost 1:1 to a service value.
- `df.describe()` on the numeric columns. Some of these network stats span many orders of magnitude (flow durations from 0 to 20M microseconds, packet counts from 0 to 10K). This matters for scaling.
- Check for zero-variance columns. With this many features extracted from network flows, some might be constant or near-constant (e.g. URG flags are probably zero for 99.9% of flows).
- Look at the rare classes specifically. Pull out the 30 Metasploit_Brute_Force_SSH rows and the 22 NMAP_FIN_SCAN rows. What do their feature distributions look like? Are they clearly separable from the majority class, or buried in the noise?
- Plot a few key features by class: `flow_SYN_flag_count` should light up for DOS_SYN_Hping. TCP flag distributions should separate the Nmap variants. IAT patterns should separate automated scans from normal IoT chatter.

## 2. Validation strategy

Stratified K-fold is the only reasonable option here. Without stratification, some folds might have zero samples of the rare classes, and macro F1 would collapse.

I'd use 5-fold stratified CV. Even with stratification, the rare classes will have only 4-6 samples per fold, so expect high variance in per-class F1 for those. The overall macro F1 should be more stable but don't be surprised if it fluctuates by 0.02-0.05 across folds.

RepeatedStratifiedKFold (3 repeats x 5 folds = 15 runs) would give a more reliable estimate at the cost of training time, but with 22K rows and simple models it's fast enough to be worth doing.

Track per-class F1 alongside macro F1. If the model gets 0.99 F1 on the big classes but 0.0 on FIN_SCAN, that's a problem even if macro F1 looks decent.

## 3. Preprocessing

**Categoricals:**
- `proto`: 3 values (tcp/udp/icmp). One-hot encode or leave as-is for tree models.
- `service`: 10 values, but "-" dominates. One-hot encode. The service feature is very informative -- MQTT traffic is literally labeled "mqtt" in this column -- so don't drop it.

**Numeric features:**
- Many features have very different scales. Packet counts, byte totals, and timing values span different ranges. Trees don't care, but anything distance-based (SVM, KNN) needs scaling.
- Some features might be near-constant for the majority class but informative for rare classes. Don't use variance thresholds to drop features without checking per-class distributions.
- Several features are arithmetic combinations of others (e.g. `flow_pkts_per_sec` = total packets / duration). Tree models handle this fine but linear models might benefit from having both.

**Feature engineering ideas:**
- `syn_to_ack_ratio = flow_SYN_flag_count / (flow_ACK_flag_count + 1)`. SYN floods have very high SYN with almost no ACK. Normal TCP has roughly balanced SYN and ACK.
- `fwd_bwd_pkt_ratio = fwd_pkts_tot / (bwd_pkts_tot + 1)`. Scans are very one-directional. Normal traffic has more back-and-forth.
- `xmas_flags = fwd_URG_flag_count + fwd_PSH_flag_count + flow_FIN_flag_count`. Directly captures the Xmas tree scan signature.
- `payload_to_header_ratio = flow_pkts_payload_tot / (fwd_header_size_tot + bwd_header_size_tot + 1)`. Scan traffic has large headers relative to payload. Normal data transfer is the opposite.
- `iat_regularity = flow_iat_std / (flow_iat_avg + 1e-6)`. Automated attacks have very regular timing (low CV), normal traffic is bursty (high CV).

**Handling imbalance:**
This is the key challenge. Options:
- SMOTE on the rare classes. Risky with only 22-30 samples -- the synthetic examples might not be representative.
- Class weights in the loss function. Simpler to implement and usually works well enough. `class_weight="balanced"` in sklearn or equivalent.
- Undersampling the majority class. Throwing away 90% of your SYN flood data is wasteful but can help some models focus on the minority classes.
- For tree-based models, `class_weight="balanced_subsample"` in RandomForest handles this natively.

I'd start with class weights and only try SMOTE if the rare-class F1 is still bad.

## 4. Baselines

Start simple:
- **Majority class baseline**: predict DOS_SYN_Hping for everything. Macro F1 ~ 0.04. Only 1 of 12 classes gets any F1 credit, so the average is terrible.
- **Decision tree (max_depth=5)**: TCP flag features alone should get surprisingly far. The Nmap variants each have a distinctive flag signature. Expect macro F1 around 0.70-0.80.
- **Random forest (100 trees, balanced class weights)**: should push to 0.90+ without much tuning. The data has strong signal in the flag and IAT features.

Check the confusion matrix after each baseline. The interesting question is which classes get confused with each other. I'd bet NMAP_TCP_scan and NMAP_OS_DETECTION share a lot of feature overlap because OS detection includes TCP probes.

## 5. Model iteration

**What I'd try:**

1. **LightGBM or XGBoost with class weights.** These handle imbalanced multi-class well out of the box. Start with max_depth 6-8, learning rate 0.1, 300-500 trees. Use the `is_unbalance` flag in LightGBM or pass `sample_weight` computed from class frequencies.

2. **Feature selection.** With 83 features, many of which are correlated (fwd/bwd/flow versions of the same metric), reducing to the top 30-40 features by importance or mutual information might help the rare classes by reducing the search space.

3. **Separate binary classifiers for rare classes.** Train a dedicated Metasploit_Brute_Force_SSH vs. everything-else model. With only 30 positive examples, a well-tuned binary classifier might do better than a 12-way model that barely sees these samples.

4. **Stacking.** An RF + LightGBM + logistic regression stack could squeeze out another point or two.

5. **Threshold tuning on predicted probabilities.** Macro F1 is non-differentiable, so the optimal decision threshold for each class isn't necessarily 1/12 or even the default argmax. After getting class probabilities, tune per-class thresholds on the validation set to directly maximize macro F1. This can matter a lot for the rare classes.

**What I'd skip:**
- Neural networks. 22K tabular rows with class imbalance -- gradient boosting will almost certainly win here. A neural net might work with careful architecture but the tuning cost isn't justified.
- KNN. Too many features, huge majority class will dominate neighborhoods.

## 6. Error analysis

After the best model is trained, look at where it fails.

- **Confusion matrix.** Which attack types get confused? I'd expect: NMAP_TCP_scan misclassified as NMAP_OS_DETECTION (both are TCP-based Nmap probes). Wipro_bulb mixed up with Thing_Speak or MQTT_Publish (all normal IoT devices, similar traffic patterns). The 7 test Metasploit_SSH samples -- at least some of them will probably be wrong.
- **Per-class F1.** If any class has F1 below 0.5, that's dragging the macro average hard. Focus tuning effort there.
- **Rare class review.** For SSH brute force and FIN scan, look at the misclassified examples individually. Are they edge cases that look ambiguous even to a human, or is the model systematically wrong? With 6-7 test examples per class, even one mistake is a 14-17% hit.
- **False positives.** Check if normal traffic (Thing_Speak, MQTT_Publish, Wipro_bulb) is being mislabeled as attacks. In real IDS deployment, false positives are extremely costly because they trigger alerts that analysts have to investigate.

## 7. Watch for overfitting

22K rows is decent, but the rare classes effectively make the useful training size much smaller. A model can easily overfit to the 22 FIN_SCAN training examples.

Signs to watch for:
- Training accuracy 100% but validation macro F1 stuck at 0.85. Classic overfit on the rare classes.
- Per-class F1 that varies wildly across CV folds for the same rare class. If fold 1 gets 1.0 on FIN_SCAN but fold 3 gets 0.0, the model hasn't learned a robust pattern.
- Feature importances that don't make network sense. If `bwd_bulk_rate` is the top predictor for SSH brute force, that's suspicious -- SSH brute force should be characterized by repeated connection attempts to port 22 with short flows, not bulk transfer patterns.

## 8. Submission checklist

1. 5,491 data rows + header.
2. Two columns: `id` and `Attack_type`.
3. IDs match test.csv exactly.
4. All `Attack_type` values are one of the 12 valid labels, spelled exactly right ("ARP_poisioning" not "ARP_poisoning" -- yes, the misspelling is in the original data, keep it).
5. Class distribution in predictions should roughly mirror training set distribution. If your submission predicts 50% NMAP_FIN_SCAN, something went wrong.
6. Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv`
7. Double-check: the most common prediction should be DOS_SYN_Hping. If it's anything else, the model is broken.
