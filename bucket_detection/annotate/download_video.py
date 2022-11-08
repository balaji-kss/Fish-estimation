import tator
api = tator.get_api(host='https://cloud.tator.io', token="8b2613b080cb53de709cb4665ea229236a00d66d")
medias = api.get_media_list(85)
for media in medias:
    if(media.id==2067840 or media.id==2067844):
        out_path = f"/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/{media.name}"
        print(f"Downloading {media.name}...")
        for progress in tator.util.download_media(api, media, out_path):
            print(f"Download progress: {progress}%")