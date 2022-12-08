# import tator
# #2068112,2068120,2067865
# api = tator.get_api(host='https://cloud.tator.io', token="8b2613b080cb53de709cb4665ea229236a00d66d")
# medias = api.get_media_list(85)
# for media in medias:
#     if(media.id in [2916992,2916795]):
#         out_path = f"/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/{media.id}.mp4"
#         print(f"Downloading {media.name}...")
#         for progress in tator.util.download_media(api, media, out_path):
#             print(f"Download progress: {progress}%")

import pandas as pd
df=pd.read_csv("/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/Harmony_5/29 singleview_summary.csv")
for i, g in df.groupby('media_id'):
     g.to_csv('/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/Harmony_5/{}.csv'.format(i), index=False)
     # print(i)
