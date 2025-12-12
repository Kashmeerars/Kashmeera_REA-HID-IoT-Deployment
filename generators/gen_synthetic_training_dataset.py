#!/usr/bin/env python3
"""
gen_synthetic_training_dataset.py

Generates:
  - synthetic_training.pcap
  - synthetic_training_13feat.csv  (13 columns, canonical order)
  - manifest.json

Designed to produce IoT-like benign traffic + attack flows (DoS-like, TCP SYN bursts, low-and-slow).
Every flow includes originator + responder packets so durations/bytes/pkts > 0.
Features computed directly from generated packets (no Zeek required) to match the
project canonical 13-feature schema.
"""

import os
import random
import time
import json
import argparse
from datetime import datetime
from scapy.all import Ether, IP, UDP, TCP, Raw, PcapWriter
import numpy as np
import csv
from statistics import mean

# -----------------------
# Utilities
# -----------------------
def ms_to_sec(ms): return float(ms) / 1000.0

def safe_port(v):
    try:
        iv = int(v)
    except:
        iv = random.randint(1025, 65535)
    iv = iv % 65535
    if iv == 0: iv = 1025
    return iv

def mk_udp_pkt(src_ip, dst_ip, sport, dport, payload_len, src_mac="02:00:00:00:00:01", dst_mac="02:00:00:00:00:02", ts=None, ttl=64):
    sport = safe_port(sport); dport = safe_port(dport)
    payload = Raw(bytes([random.randint(0,255) for _ in range(max(0,int(payload_len)))]))
    pkt = Ether(src=src_mac, dst=dst_mac)/IP(src=src_ip, dst=dst_ip, ttl=ttl)/UDP(sport=sport,dport=dport)/payload
    if ts is not None: pkt.time = ts
    return pkt

def mk_tcp_syn(src_ip, dst_ip, sport, dport, ts=None, ttl=64):
    sport = safe_port(sport); dport = safe_port(dport)
    tcp = TCP(sport=sport, dport=dport, flags="S", seq=random.randint(0,2**32-1))
    pkt = Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")/IP(src=src_ip, dst=dst_ip, ttl=ttl)/tcp
    if ts is not None: pkt.time = ts
    return pkt

def mk_tcp_synack(dst_ip, src_ip, sport, dport, ts=None, ttl=64):
    # server->client SYN+ACK (note swapped src/dst)
    sport = safe_port(sport); dport = safe_port(dport)
    tcp = TCP(sport=sport, dport=dport, flags="SA", seq=random.randint(0,2**32-1), ack=1)
    pkt = Ether(src="02:00:00:00:00:02", dst="02:00:00:00:00:01")/IP(src=src_ip, dst=dst_ip, ttl=ttl)/tcp
    if ts is not None: pkt.time = ts
    return pkt

def mk_tcp_ack(src_ip, dst_ip, sport, dport, ts=None, ttl=64):
    sport = safe_port(sport); dport = safe_port(dport)
    tcp = TCP(sport=sport, dport=dport, flags="A", seq=random.randint(0,2**32-1), ack=1)
    pkt = Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")/IP(src=src_ip, dst=dst_ip, ttl=ttl)/tcp
    if ts is not None: pkt.time = ts
    return pkt

# -----------------------
# Flow -> feature computation (canonical)
# -----------------------
def compute_flow_features(pkt_records, orig_ip, resp_ip, orig_port, resp_port, proto, label):
    """
    pkt_records: list of tuples (timestamp, direction, pkt_len, tcp_flags)
       direction: 'orig' or 'resp' (origator -> responder or reverse)
    proto: 6 = TCP, 17 = UDP
    Returns dict of 13 canonical features in required order.
    """
    # Sort by time
    pkt_records.sort(key=lambda x: x[0])
    times = [t for (t, d, l, tf) in pkt_records]
    lens = [l for (t, d, l, tf) in pkt_records]
    dirs = [d for (t, d, l, tf) in pkt_records]
    tcp_flags_list = [tf for (t,d,l,tf) in pkt_records]

    first_ts = times[0]
    last_ts = times[-1]
    flow_duration_s = max(0.0, last_ts - first_ts)
    flow_duration_ms = flow_duration_s * 1000.0 if flow_duration_s>=0 else 0.0

    # split orig/resp timings
    orig_times = [t for (t,d,l,tf) in pkt_records if d=='orig']
    resp_times = [t for (t,d,l,tf) in pkt_records if d=='resp']
    duration_in_s = (max(orig_times) - min(orig_times)) if len(orig_times)>=2 else (flow_duration_s if len(orig_times)==1 else 0.0)
    duration_out_s = (max(resp_times) - min(resp_times)) if len(resp_times)>=2 else (flow_duration_s if len(resp_times)==1 else 0.0)

    # bytes
    in_bytes = sum(l for (t,d,l,tf) in pkt_records if d=='orig')
    out_bytes = sum(l for (t,d,l,tf) in pkt_records if d=='resp')

    # min ttl fixed design: use 64 (generator sets it), but keep generic in case changed upstream
    min_ttl = 64

    longest_pkt = max(lens) if lens else 0
    shortest_pkt = min(lens) if lens else 0

    # TCP_FLAGS: approximate aggregate numeric representation of flags seen in flow (bitwise OR)
    # For UDP flows we set 0.
    agg_tcp_flags = 0
    for tf in tcp_flags_list:
        if tf is None: continue
        agg_tcp_flags |= int(tf)

    # CLIENT_TCP_FLAGS: flags seen in originator packets
    client_tcp_flags = 0
    for (t,d,l,tf) in pkt_records:
        if d=='orig' and tf is not None:
            client_tcp_flags |= int(tf)

    # IAT_mean: mean inter-arrival across all packets (seconds)
    iat_vals = []
    for i in range(len(times)-1):
        diff = times[i+1] - times[i]
        if diff >= 0:
            iat_vals.append(diff)
    if len(iat_vals) >= 1:
        iat_mean = mean(iat_vals)
    else:
        # single packet case: use flow duration in seconds (matching methodology)
        iat_mean = flow_duration_s

    # Compose row in canonical order
    row = {
        'L4_SRC_PORT': int(orig_port),
        'IN_BYTES': int(in_bytes),
        'OUT_BYTES': int(out_bytes),
        'FLOW_DURATION_MILLISECONDS': float(flow_duration_ms),
        'PROTOCOL': int(proto),
        'TCP_FLAGS': int(agg_tcp_flags),
        'DURATION_IN': float(duration_in_s * 1000.0),
        'DURATION_OUT': float(duration_out_s * 1000.0),
        'MIN_TTL': int(min_ttl),
        'LONGEST_FLOW_PKT': int(longest_pkt),
        'SHORTEST_FLOW_PKT': int(shortest_pkt),
        'CLIENT_TCP_FLAGS': int(client_tcp_flags),
        'IAT_mean': float(iat_mean),
        'Label': int(label)
    }
    return row

# -----------------------
# Main synth routine
# -----------------------
def synth(out_dir, pcap_name, csv_name, total_flows=24000, attack_frac=0.5, seed=42):
    random.seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    pcap_path = os.path.join(out_dir, pcap_name)
    csv_path = os.path.join(out_dir, csv_name)
    manifest_path = os.path.join(out_dir, 'manifest.json')

    pw = PcapWriter(pcap_path, sync=True)

    # hosts (IoT-like devices behind gateway)
    clients = [f"10.0.0.{i}" for i in range(2,20)]
    servers = ["192.168.1.10", "192.168.1.11"]  # local servers/cloud endpoints
    dst_ports_iot = [53, 123, 1883, 8883, 8000, 80]  # DNS, NTP, MQTT, alt HTTP
    timestamp = time.time()
    current_time = timestamp

    stats = {'total_flows': total_flows, 'attack_frac': attack_frac, 'attack_flows':0, 'benign_flows':0, 'seed':seed}

    rows = []
    for fid in range(total_flows):
        # small spacing between flow starts
        current_time += random.uniform(0.001, 0.05)  # 1ms-50ms between flow starts

        is_attack = (random.random() < attack_frac)
        if is_attack:
            stats['attack_flows'] += 1
            label = 1
            # attack mode selection (weighted)
            mode = random.choices(['udp_flood','tcp_syn_flood','low_and_slow'], weights=[0.4,0.3,0.3])[0]
        else:
            stats['benign_flows'] += 1
            label = 0
            mode = 'iot_benign'

        src = random.choice(clients)
        dst = random.choice(servers)
        sport = random.randint(1025, 65535)
        dport = random.choice(dst_ports_iot)

        pkt_records = []  # (ts, dir, pkt_len, tcp_flags)

        if mode == 'iot_benign':
            # IoT-like: small multi-packet, quick replies, small IAT
            pkt_count = random.randint(2,6)
            t = current_time
            for i in range(pkt_count):
                size = random.randint(60,900)  # payload sizes
                p = mk_udp_pkt(src, dst, sport+i, dport, payload_len=size, ts=t)
                pw.write(p)
                pkt_records.append((p.time, 'orig', len(p), None))
                # reply almost immediately
                t_reply = t + ms_to_sec(random.uniform(1,10))
                r = mk_udp_pkt(dst, src, dport, sport+i, payload_len=random.randint(20,300), src_mac="02:00:00:00:00:02", dst_mac="02:00:00:00:00:01", ts=t_reply)
                pw.write(r)
                pkt_records.append((r.time, 'resp', len(r), None))
                t = t + ms_to_sec(random.uniform(5,30))

        elif mode == 'udp_flood':
            # attack: multiple orig packets with short spacing; server replies can be absent or minimal
            num = random.randint(4, 20)
            t = current_time
            for i in range(num):
                size = random.randint(200,1500)
                p = mk_udp_pkt(src, dst, sport+i, dport, payload_len=size, ts=t)
                pw.write(p)
                pkt_records.append((p.time, 'orig', len(p), None))
                # occasional reply (to keep bidirectional)
                if random.random() < 0.2:
                    t_r = t + ms_to_sec(random.uniform(10,200))
                    r = mk_udp_pkt(dst, src, dport, sport+i, payload_len=random.randint(10,200), src_mac="02:00:00:00:00:02", dst_mac="02:00:00:00:00:01", ts=t_r)
                    pw.write(r)
                    pkt_records.append((r.time, 'resp', len(r), None))
                t += ms_to_sec(random.uniform(1,10))

        elif mode == 'tcp_syn_flood':
            # many SYNs spaced small; some SYN-ACKs included occasionally
            num_syn = random.randint(3,12)
            t = current_time
            for i in range(num_syn):
                syn = mk_tcp_syn(src, dst, sport+i, dport, ts=t)
                pw.write(syn)
                pkt_records.append((syn.time, 'orig', len(syn), int(syn[TCP].flags)))
                if random.random() < 0.25:
                    t_reply = t + ms_to_sec(random.uniform(10,200))
                    synack = mk_tcp_synack(dst, src, dport, sport+i, ts=t_reply)
                    pw.write(synack)
                    pkt_records.append((synack.time, 'resp', len(synack), int(synack[TCP].flags)))
                    if random.random() < 0.4:
                        t_ack = t_reply + ms_to_sec(random.uniform(5,50))
                        ack = mk_tcp_ack(src, dst, sport+i, dport, ts=t_ack)
                        pw.write(ack)
                        pkt_records.append((ack.time, 'orig', len(ack), int(ack[TCP].flags)))
                        t = t_ack + ms_to_sec(random.uniform(5,200))
                    else:
                        t = t_reply + ms_to_sec(random.uniform(5,200))
                else:
                    t = t + ms_to_sec(random.uniform(5,50))

        elif mode == 'low_and_slow':
            # spaced orig->resp pairs or few-packet multi flows with large IATs (the low-and-slow we want)
            submode = random.choice(['udp_drip','slow_multi'])
            if submode == 'udp_drip':
                t0 = current_time
                p = mk_udp_pkt(src, dst, sport, dport, payload_len=random.randint(40,400), ts=t0)
                pw.write(p); pkt_records.append((p.time,'orig',len(p),None))
                # long reply delay
                reply_delay = ms_to_sec(random.uniform(200, 5000))  # 0.2s - 5s
                t1 = t0 + reply_delay
                r = mk_udp_pkt(dst, src, dport, sport, payload_len=random.randint(10,120), src_mac="02:00:00:00:00:02", dst_mac="02:00:00:00:00:01", ts=t1)
                pw.write(r); pkt_records.append((r.time,'resp',len(r),None))
            else:
                pkt_count = random.randint(2,6)
                t = current_time
                for i in range(pkt_count):
                    p = mk_udp_pkt(src, dst, sport+i, dport, payload_len=random.randint(40,500), ts=t)
                    pw.write(p); pkt_records.append((p.time,'orig',len(p),None))
                    # reply longish
                    t_r = t + ms_to_sec(random.uniform(100,3000))
                    r = mk_udp_pkt(dst, src, dport, sport+i, payload_len=random.randint(10,200), src_mac="02:00:00:00:00:02", dst_mac="02:00:00:00:00:01", ts=t_r)
                    pw.write(r); pkt_records.append((r.time,'resp',len(r),None))
                    t = t_r + ms_to_sec(random.uniform(200,2000))

        # Compute features from this flow
        # Ensure there is at least one orig and one resp; if not, create tiny reply to enforce bidirectionality
        if not any(d=='resp' for (_,d,_,_) in pkt_records):
            t_reply = current_time + ms_to_sec(random.uniform(1,50))
            small = mk_udp_pkt(dst, src, dport, sport, payload_len=20, src_mac="02:00:00:00:00:02", dst_mac="02:00:00:00:00:01", ts=t_reply)
            pw.write(small); pkt_records.append((small.time,'resp',len(small),None))
        if not any(d=='orig' for (_,d,_,_) in pkt_records):
            # create orig small
            t_orig = current_time
            small = mk_udp_pkt(src, dst, sport, dport, payload_len=20, ts=t_orig)
            pw.write(small); pkt_records.append((small.time,'orig',len(small),None))

        # compute features with default proto=17 (UDP); real protocol inferred below
        row = compute_flow_features(pkt_records, src, dst, sport, dport, 17, label=label)

        # Detect if any TCP packet existed in the flow
        has_tcp = any(tf is not None for (_,d,l,tf) in pkt_records)
        row['PROTOCOL'] = 6 if has_tcp else 17
        rows.append(row)

    pw.close()

    # write CSV
    columns = ['L4_SRC_PORT','IN_BYTES','OUT_BYTES','FLOW_DURATION_MILLISECONDS','PROTOCOL','TCP_FLAGS','DURATION_IN','DURATION_OUT','MIN_TTL','LONGEST_FLOW_PKT','SHORTEST_FLOW_PKT','CLIENT_TCP_FLAGS','IAT_mean','Label']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k:r[k] for k in columns})

    manifest = {
        'pcap': os.path.basename(pcap_path),
        'csv': os.path.basename(csv_path),
        'generated_at': datetime.utcfromtimestamp(time.time()).isoformat()+"Z",
        'total_flows': total_flows,
        'attack_flows': stats['attack_flows'],
        'benign_flows': stats['benign_flows'],
        'attack_fraction': stats['attack_flows']/max(1,stats['total_flows']),
        'seed': seed,
        'notes': 'IoT-like benigns (UDP-based) + attacks (udp_flood, tcp_syn_flood, low_and_slow). Each flow contains originator + responder packets so that duration/bytes/pkts are non-zero.'
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Wrote pcap: {pcap_path}")
    print(f"[+] Wrote csv : {csv_path}")
    print(f"[+] Wrote manifest: {manifest_path}")
    return pcap_path, csv_path, manifest_path

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IoT training dataset (pcap + 13-feature csv).")
    parser.add_argument('--out-dir', default='../zeek_logs/synthetic_training', help='output directory (default: ../zeek_logs/synthetic_training)')
    parser.add_argument('--total-flows', type=int, default=24000, help='number of flows to generate (default 24000)')
    parser.add_argument('--attack-frac', type=float, default=0.5, help='fraction of flows that are attack (default 0.5)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default 42)')
    parser.add_argument('--pcap-name', default='synthetic_training.pcap')
    parser.add_argument('--csv-name', default='synthetic_training_13feat.csv')
    args = parser.parse_args()

    synth(args.out_dir, args.pcap_name, args.csv_name, total_flows=args.total_flows, attack_frac=args.attack_frac, seed=args.seed)

if __name__ == '__main__':
    main()
