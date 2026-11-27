from pathlib import Path
import csv, yaml, json, hashlib
ROOT=Path.cwd()
sample=Path(r"C:\NOVEL_IDS_runs\phaseD_cic_partial_20261126\cic_sample.csv")
with sample.open("r",encoding="utf-8-sig",newline="") as f: cols=next(csv.reader(f))
cols=[c for c in cols if c.strip().lower() not in {"unnamed: 0","unnamed:0","label"}]
descriptions={
"Flow ID":("Bidirectional flow identifier derived from endpoint tuple and protocol","identifier","Flow key; not an attack feature"),
"Src IP":("Source IP address of the forward direction","address","Origin endpoint address"),
"Source IP":("Source IP address of the forward direction","address","Origin endpoint address"),
"Src Port":("Source transport port of the forward direction","port","Origin transport endpoint"),
"Source Port":("Source transport port of the forward direction","port","Origin transport endpoint"),
"Destination IP":("Destination IP address of the forward direction","address","Destination endpoint address"),
"Dst IP":("Destination IP address of the forward direction","address","Destination endpoint address"),
"Destination Port":("Destination transport port of the forward direction","port","Destination transport endpoint"),
"Dst Port":("Destination transport port of the forward direction","port","Destination transport endpoint"),
"Protocol":("IANA IP protocol number carried by the flow","iana_protocol_number","Transport protocol identifier"),
"Timestamp":("Timestamp associated with the start of the flow","datetime","Flow start time"),
"Flow Duration":("Duration of the bidirectional flow","microseconds","Elapsed time between first and last observed packet"),
"Total Fwd Packets":("Total number of packets in the forward direction","packets","Forward packet volume"),
"Total Backward Packets":("Total number of packets in the backward direction","packets","Backward packet volume"),
"Total Length of Fwd Packets":("Total bytes carried by forward-direction packets","bytes","Forward byte volume"),
"Total Length of Bwd Packets":("Total bytes carried by backward-direction packets","bytes","Backward byte volume"),
"Flow Bytes/s":("Number of flow bytes transferred per second","bytes_per_second","Bidirectional byte rate"),
"Flow Packets/s":("Number of flow packets transferred per second","packets_per_second","Bidirectional packet rate"),
"Fwd Packets/s":("Forward-direction packet rate","packets_per_second","Forward packet rate"),
"Bwd Packets/s":("Backward-direction packet rate","packets_per_second","Backward packet rate"),
"Flow IAT Mean":("Mean inter-arrival time between consecutive flow packets","microseconds","Flow timing distribution"),
"Flow IAT Std":("Standard deviation of flow packet inter-arrival times","microseconds","Flow timing distribution"),
"Flow IAT Max":("Maximum flow packet inter-arrival time","microseconds","Flow timing distribution"),
"Flow IAT Min":("Minimum flow packet inter-arrival time","microseconds","Flow timing distribution"),
"Fwd IAT Total":("Total inter-arrival time in the forward direction","microseconds","Forward timing distribution"),
"Fwd IAT Mean":("Mean inter-arrival time in the forward direction","microseconds","Forward timing distribution"),
"Fwd IAT Std":("Standard deviation of forward inter-arrival times","microseconds","Forward timing distribution"),
"Fwd IAT Max":("Maximum forward inter-arrival time","microseconds","Forward timing distribution"),
"Fwd IAT Min":("Minimum forward inter-arrival time","microseconds","Forward timing distribution"),
"Bwd IAT Total":("Total inter-arrival time in the backward direction","microseconds","Backward timing distribution"),
"Bwd IAT Mean":("Mean inter-arrival time in the backward direction","microseconds","Backward timing distribution"),
"Bwd IAT Std":("Standard deviation of backward inter-arrival times","microseconds","Backward timing distribution"),
"Bwd IAT Max":("Maximum backward inter-arrival time","microseconds","Backward timing distribution"),
"Bwd IAT Min":("Minimum backward inter-arrival time","microseconds","Backward timing distribution"),
"Fwd PSH Flags":("Count of forward packets with TCP PSH flag set","count","TCP control flag frequency"),
"Bwd PSH Flags":("Count of backward packets with TCP PSH flag set","count","TCP control flag frequency"),
"Fwd URG Flags":("Count of forward packets with TCP URG flag set","count","TCP control flag frequency"),
"Bwd URG Flags":("Count of backward packets with TCP URG flag set","count","TCP control flag frequency"),
"FIN Flag Count":("Number of packets with TCP FIN flag","count","TCP control flag frequency"),
"SYN Flag Count":("Number of packets with TCP SYN flag","count","TCP control flag frequency"),
"RST Flag Count":("Number of packets with TCP RST flag","count","TCP control flag frequency"),
"PSH Flag Count":("Number of packets with TCP PSH flag","count","TCP control flag frequency"),
"ACK Flag Count":("Number of packets with TCP ACK flag","count","TCP control flag frequency"),
"URG Flag Count":("Number of packets with TCP URG flag","count","TCP control flag frequency"),
"CWE Flag Count":("Number of packets with TCP CWR flag","count","TCP control flag frequency"),
"CWR Flag Count":("Number of packets with TCP CWR flag","count","TCP control flag frequency"),
"ECE Flag Count":("Number of packets with TCP ECE flag","count","TCP control flag frequency"),
"Fwd Header Length":("Total header bytes in forward direction","bytes","Forward protocol header size"),
"Fwd Header Length.1":("Duplicate CICFlowMeter forward header length field","bytes","Forward protocol header size"),
"Bwd Header Length":("Total header bytes in backward direction","bytes","Backward protocol header size"),
"Min Packet Length":("Minimum packet length observed in the flow","bytes","Packet-size distribution"),
"Max Packet Length":("Maximum packet length observed in the flow","bytes","Packet-size distribution"),
"Packet Length Mean":("Mean packet length in the flow","bytes","Packet-size distribution"),
"Packet Length Std":("Standard deviation of packet lengths in the flow","bytes","Packet-size distribution"),
"Packet Length Variance":("Variance of packet lengths in the flow","bytes_squared","Packet-size distribution"),
"Average Packet Size":("Average packet size over the flow","bytes","Packet-size distribution"),
"Avg Fwd Segment Size":("Average segment size in the forward direction","bytes","Forward segment-size distribution"),
"Avg Bwd Segment Size":("Average segment size in the backward direction","bytes","Backward segment-size distribution"),
"Fwd Packet Length Max":("Maximum forward packet length","bytes","Forward packet-size distribution"),
"Fwd Packet Length Min":("Minimum forward packet length","bytes","Forward packet-size distribution"),
"Fwd Packet Length Mean":("Mean forward packet length","bytes","Forward packet-size distribution"),
"Fwd Packet Length Std":("Standard deviation of forward packet lengths","bytes","Forward packet-size distribution"),
"Bwd Packet Length Max":("Maximum backward packet length","bytes","Backward packet-size distribution"),
"Bwd Packet Length Min":("Minimum backward packet length","bytes","Backward packet-size distribution"),
"Bwd Packet Length Mean":("Mean backward packet length","bytes","Backward packet-size distribution"),
"Bwd Packet Length Std":("Standard deviation of backward packet lengths","bytes","Backward packet-size distribution"),
"Down/Up Ratio":("Ratio between download and upload traffic","dimensionless","Directional traffic balance"),
"Fwd Avg Bytes/Bulk":("Average forward bytes per bulk transfer","bytes_per_bulk","Forward bulk transfer statistic"),
"Fwd Avg Packets/Bulk":("Average forward packets per bulk transfer","packets_per_bulk","Forward bulk transfer statistic"),
"Fwd Avg Bulk Rate":("Average forward bulk transfer rate","bulk_rate","Forward bulk transfer statistic"),
"Bwd Avg Bytes/Bulk":("Average backward bytes per bulk transfer","bytes_per_bulk","Backward bulk transfer statistic"),
"Bwd Avg Packets/Bulk":("Average backward packets per bulk transfer","packets_per_bulk","Backward bulk transfer statistic"),
"Bwd Avg Bulk Rate":("Average backward bulk transfer rate","bulk_rate","Backward bulk transfer statistic"),
"Subflow Fwd Packets":("Average number of forward packets in a subflow","packets","Forward subflow volume"),
"Subflow Fwd Bytes":("Average number of forward bytes in a subflow","bytes","Forward subflow volume"),
"Subflow Bwd Packets":("Average number of backward packets in a subflow","packets","Backward subflow volume"),
"Subflow Bwd Bytes":("Average number of backward bytes in a subflow","bytes","Backward subflow volume"),
"Init_Win_bytes_forward":("Initial TCP window size in forward direction","bytes","Forward TCP receive-window size"),
"Init_Win_bytes_backward":("Initial TCP window size in backward direction","bytes","Backward TCP receive-window size"),
"act_data_pkt_fwd":("Count of forward packets with at least one byte of TCP payload","packets","Forward data-bearing packet count"),
"min_seg_size_forward":("Minimum TCP segment size observed in forward direction","bytes","Forward segment-size statistic"),
"Active Mean":("Mean active period before a flow becomes idle","microseconds","Flow active/idle timing"),
"Active Std":("Standard deviation of active periods","microseconds","Flow active/idle timing"),
"Active Max":("Maximum active period","microseconds","Flow active/idle timing"),
"Active Min":("Minimum active period","microseconds","Flow active/idle timing"),
"Idle Mean":("Mean idle period before a flow becomes active","microseconds","Flow active/idle timing"),
"Idle Std":("Standard deviation of idle periods","microseconds","Flow active/idle timing"),
"Idle Max":("Maximum idle period","microseconds","Flow active/idle timing"),
"Idle Min":("Minimum idle period","microseconds","Flow active/idle timing"),
"SimillarHTTP":("HTTP request URI or host-like application string recorded by the extractor","text","Application-layer request identifier"),
"Inbound":("Direction indicator for inbound traffic","binary","Traffic direction flag"),
}
level1={
"Flow Duration":("flow_duration","duration","seconds","microseconds_to_seconds"),
"Protocol":("transport_protocol","protocol_type","symbolic","iana_protocol_to_name"),
"Total Length of Fwd Packets":("source_to_destination_bytes","src_bytes","bytes",None),
"Total Length of Bwd Packets":("destination_to_source_bytes","dst_bytes","bytes",None),
}
groups={
"Total Fwd Packets":"packet_volume","Total Backward Packets":"packet_volume",
"Subflow Fwd Packets":"packet_volume","Subflow Bwd Packets":"packet_volume","act_data_pkt_fwd":"packet_volume",
"Flow Bytes/s":"bandwidth_load","Flow Packets/s":"bandwidth_load","Fwd Packets/s":"bandwidth_load","Bwd Packets/s":"bandwidth_load",
"Fwd Avg Bulk Rate":"bandwidth_load","Bwd Avg Bulk Rate":"bandwidth_load",
"Flow IAT Mean":"packet_interarrival_time","Flow IAT Std":"packet_interarrival_time","Flow IAT Max":"packet_interarrival_time","Flow IAT Min":"packet_interarrival_time",
"Fwd IAT Total":"packet_interarrival_time","Fwd IAT Mean":"packet_interarrival_time","Fwd IAT Std":"packet_interarrival_time","Fwd IAT Max":"packet_interarrival_time","Fwd IAT Min":"packet_interarrival_time",
"Bwd IAT Total":"packet_interarrival_time","Bwd IAT Mean":"packet_interarrival_time","Bwd IAT Std":"packet_interarrival_time","Bwd IAT Max":"packet_interarrival_time","Bwd IAT Min":"packet_interarrival_time",
"Fwd PSH Flags":"connection_state","Bwd PSH Flags":"connection_state","Fwd URG Flags":"connection_state","Bwd URG Flags":"connection_state",
"FIN Flag Count":"connection_state","SYN Flag Count":"connection_state","RST Flag Count":"connection_state","PSH Flag Count":"connection_state","ACK Flag Count":"connection_state","URG Flag Count":"connection_state","CWE Flag Count":"connection_state","CWR Flag Count":"connection_state","ECE Flag Count":"connection_state",
"Init_Win_bytes_forward":"tcp_window_measurement","Init_Win_bytes_backward":"tcp_window_measurement",
"Fwd Header Length":"packet_size_measurement","Fwd Header Length.1":"packet_size_measurement","Bwd Header Length":"packet_size_measurement",
"Min Packet Length":"packet_size_measurement","Max Packet Length":"packet_size_measurement","Packet Length Mean":"packet_size_measurement","Packet Length Std":"packet_size_measurement","Packet Length Variance":"packet_size_measurement","Average Packet Size":"packet_size_measurement",
"Avg Fwd Segment Size":"packet_size_measurement","Avg Bwd Segment Size":"packet_size_measurement","Fwd Packet Length Max":"packet_size_measurement","Fwd Packet Length Min":"packet_size_measurement","Fwd Packet Length Mean":"packet_size_measurement","Fwd Packet Length Std":"packet_size_measurement","Bwd Packet Length Max":"packet_size_measurement","Bwd Packet Length Min":"packet_size_measurement","Bwd Packet Length Mean":"packet_size_measurement","Bwd Packet Length Std":"packet_size_measurement","min_seg_size_forward":"packet_size_measurement",
"Down/Up Ratio":"connection_context_ratio",
"Fwd Avg Bytes/Bulk":"packet_size_measurement","Fwd Avg Packets/Bulk":"packet_volume","Bwd Avg Bytes/Bulk":"packet_size_measurement","Bwd Avg Packets/Bulk":"packet_volume",
"Subflow Fwd Bytes":"packet_size_measurement","Subflow Bwd Bytes":"packet_size_measurement",
"Active Mean":"packet_interarrival_time","Active Std":"packet_interarrival_time","Active Max":"packet_interarrival_time","Active Min":"packet_interarrival_time",
"Idle Mean":"packet_interarrival_time","Idle Std":"packet_interarrival_time","Idle Max":"packet_interarrival_time","Idle Min":"packet_interarrival_time",
}
time_features={n.strip() for n in cols if n.strip().startswith(("Flow IAT","Fwd IAT","Bwd IAT","Active ","Idle "))}
entries={}
for name in cols:
 key=name.strip()
 desc,unit,meaning=descriptions.get(key,(f"CICFlowMeter-derived field {name}; interpretation follows the field name and its feature family.","dataset_specific","Dataset-specific flow metadata"))
 if key in level1:
  sem,src,srcunit,transform=level1[key]
  mapping={"level":1,"target_semantic_id":sem,"target_source_feature":src,"relationship":"equivalent","confidence":"high","unit_compatible":True,"type_compatible":True,"declared_transform":transform,"evidence":f"Clear semantic equivalent to source feature {src}; unit/type compatibility checked after declared conversion."}
 elif key in groups:
  mapping={"level":2,"semantic_group":groups[key],"relationship":"group_only","confidence":"medium","unit_compatible":True,"type_compatible":True,"declared_transform":"microseconds_to_seconds" if key in time_features else None,"evidence":f"Related {groups[key]} descriptor only; not an equivalent feature. CICFlowMeter description supports group-level association; no source feature identity is asserted."}
 else:
  mapping={"level":3,"relationship":"dataset_specific","confidence":"unassessed","unit_compatible":False,"type_compatible":False,"declared_transform":None,"evidence":"No clear source semantic equivalent or approved group-level match; retained as dataset-specific."}
 transform=None
 if key in time_features: transform="microseconds_to_seconds"
 elif key=="Flow Duration": transform="microseconds_to_seconds"
 elif key=="Protocol": transform="iana_protocol_to_name"
 entries[name]={"description":desc,"unit":unit,"network_meaning":meaning,"data_type":"symbolic" if key in {"Protocol","Flow ID","Src IP","Source IP","Destination IP","Dst IP","Timestamp","SimillarHTTP"} else ("binary" if key=="Inbound" else "continuous"),"mapping":mapping,"value_transform":transform}
metadata={"schema_version":1,"dataset":"CIC-DDoS2019","source_url":"https://www.unb.ca/cic/datasets/ddos-2019.html","feature_documentation":["https://github.com/jordan178169/CICFlowMeter/blob/master/ReadMe.txt","https://github.com/ahlashkari/CICFlowMeter/blob/master/src/main/java/cic/cs/unb/ca/jnetpcap/FlowFeature.java"],"transform_definitions":{"microseconds_to_seconds":"Divide time values by 1,000,000; CICFlowMeter duration/IAT/active/idle measurements are in microseconds.","iana_protocol_to_name":"Map IANA IP protocol numbers to names (1=icmp, 6=tcp, 17=udp, 47=gre, 50=esp, 58=ipv6-icmp); preserve other values as protocol_<number>."},"features":entries}
(ROOT/"config/cic_feature_metadata.yaml").write_text(yaml.safe_dump(metadata,sort_keys=False,allow_unicode=True),encoding="utf-8")
mapping={"schema_version":1,"dataset":"CIC-DDoS2019","mapping_scope":"metadata-only; created before second-run predictions; no labels or model performance consulted","source_common_features":["flow_duration","transport_protocol","network_service","source_to_destination_bytes","destination_to_source_bytes"],"missing_source_common_features":["network_service"],"features":{k:v["mapping"] for k,v in entries.items()}}
(ROOT/"config/cic_feature_mapping.yaml").write_text(yaml.safe_dump(mapping,sort_keys=False,allow_unicode=True),encoding="utf-8")
for p in (ROOT/"config/cic_feature_metadata.yaml",ROOT/"config/cic_feature_mapping.yaml"):
 print(p.name,hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size)
from collections import Counter
print("LEVEL_COUNTS",Counter(str(v["mapping"]["level"]) for v in entries.values()))
print("FEATURE_COUNT",len(entries),"LEVEL1",[(k,v["mapping"].get("target_semantic_id")) for k,v in entries.items() if v["mapping"]["level"]==1])
print("LEVEL2",sum(v["mapping"]["level"]==2 for v in entries.values()),"LEVEL3",sum(v["mapping"]["level"]==3 for v in entries.values()))
