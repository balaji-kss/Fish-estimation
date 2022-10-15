import pandas as pd
df=pd.read_csv("/home/balaji/Documents/code/RSL/Fish/bucket_detection/annotate/Harmony_12/31 singleview_summary.csv")
df=df[df.Species!="VERTEBRATES, UNCLASSIFIED"]
df=df.sort_values(by="frame")
df.to_csv("only_fish_116.csv")